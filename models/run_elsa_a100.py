import os, sys, math, time, json, argparse, statistics, random, subprocess, importlib.util, textwrap, shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import pandas as pd
except Exception:
    pd = None

# -------------------- Utilities --------------------
def set_deterministic(seed=123):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def giB(x_bytes): return float(x_bytes) / (1024.0**3)

def device_info():
    dev = torch.cuda.get_device_properties(0)
    return {
        "name": dev.name,
        "total_gb": round(dev.total_memory/(1024**3),2),
        "sm_count": dev.multi_processor_count,
        "major": dev.major, "minor": dev.minor,
        "driver": torch.version.cuda,
        "pytorch": torch.__version__,
    }

def ensure_dirs():
    Path("results").mkdir(exist_ok=True)
    Path("figs").mkdir(exist_ok=True)

def reset_cuda_peak():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def percentile(vals, p):
    if not vals: return None
    vals_sorted = sorted(vals)
    k = max(0, min(len(vals_sorted)-1, int(round((p/100.0)*(len(vals_sorted)-1)))))
    return float(vals_sorted[k])

def median_ms(vals): return float(statistics.median(vals)) if vals else float("nan")

# -------------------- Kernel Loader --------------------
def load_kernels(kernel_path: str):
    spec = importlib.util.spec_from_file_location("elsa_kernels", kernel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    syms = {}
    # Optional symbols
    syms["CAN_triton_fp32"] = getattr(mod, "CAN_triton_fp32", None)
    syms["CAN_triton"]      = getattr(mod, "CAN_triton", None)
    syms["CAN_triton_mem"]  = getattr(mod, "CAN_triton_mem", None)
    syms["CAN_pytorch"]     = getattr(mod, "CAN_pytorch", None)
    return syms

# -------------------- Implementations --------------------
def sdpa_baseline(q, k, v, is_causal=False, backend="mem_efficient"):
    # backend in {"math","mem_efficient","flash"}; we only use "math" or "mem_efficient"
    if backend == "math":
        ctx = torch.backends.cuda.sdp_kernel(enable_math=True, enable_mem_efficient=False, enable_flash=False)
    else:
        ctx = torch.backends.cuda.sdp_kernel(enable_math=False, enable_mem_efficient=True, enable_flash=False)
    with ctx:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal)
    return out

def make_inputs(B,H,N,D,dtype=torch.float32,device="cuda",scale=True):
    q = torch.randn(B,H,N,D, device=device, dtype=dtype)
    k = torch.randn(B,H,N,D, device=device, dtype=dtype)
    v = torch.randn(B,H,N,D, device=device, dtype=dtype)
    scl = (1.0 / math.sqrt(D)) if scale else 1.0
    return q,k,v,scl

def measure_once(fn, q,k,v, warmup=10, trials=50):
    torch.cuda.synchronize()
    reset_cuda_peak()
    # warmup
    for _ in range(warmup): 
        _ = fn(q,k,v); torch.cuda.synchronize()
    # timed
    lat = []
    for _ in range(trials):
        t0 = time.perf_counter()
        _ = fn(q,k,v)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        lat.append( (t1-t0)*1e3 )
    peak_gb = giB(torch.cuda.max_memory_allocated())
    return {
        "lat_ms_med": median_ms(lat),
        "lat_ms_p5": percentile(lat,5),
        "lat_ms_p95": percentile(lat,95),
        "peak_gb": peak_gb
    }

# Policy toggles
def set_strict_fp32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

def set_turbo_tf32():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["NVIDIA_TF32_OVERRIDE"] = "1"

# -------------------- Experiment Runners --------------------
def run_fp32_long_seq(kernels, lengths, B=1, H=8, D=64, warmup=10, trials=50):
    # ELSA-Strict-FP32 vs Baseline-FP32-memEff
    results = []
    set_strict_fp32()
    assert kernels["CAN_triton_fp32"] is not None, "Missing CAN_triton_fp32 in your kernel file."
    def elsa_fp32(q,k,v):
        return kernels["CAN_triton_fp32"].apply(q.float(), k.float(), v.float(), 1.0/math.sqrt(q.shape[-1]))
    def baseline_fp32(q,k,v):
        return sdpa_baseline(q.float(), k.float(), v.float(), is_causal=False, backend="mem_efficient")
    for N in lengths:
        for name,fn,dtype in [
            ("ELSA-Strict-FP32", elsa_fp32, torch.float32),
            ("Baseline-FP32-MemEff", baseline_fp32, torch.float32),
        ]:
            q,k,v,_ = make_inputs(B,H,N,D,dtype=dtype)
            met = measure_once(fn, q,k,v, warmup=warmup, trials=trials)
            tokens = B*N
            met.update({
                "method": name, "B":B,"H":H,"D":D,"N":N,
                "tokens_per_s": (tokens / (met["lat_ms_med"]/1e3))
            })
            results.append(met)
    return results

def run_mixed_and_fp16(kernels, lengths, B=1, H=8, D=64, warmup=10, trials=50):
    # Mixed-Triton (FP16 matmuls; FP32 accum) vs pure FP16 baseline
    assert kernels["CAN_triton"] is not None, "Missing CAN_triton in your kernel file."
    def elsa_mixed(q,k,v):
        # CAN_triton: (q,k,v, scale, qk_norm_weights=None, is_causal=False)
        return kernels["CAN_triton"].apply(q.half(), k.half(), v.half(), 1.0/math.sqrt(q.shape[-1]))
    def baseline_fp16(q,k,v):
        return sdpa_baseline(q.half(), k.half(), v.half(), is_causal=False, backend="mem_efficient")
    results = []
    set_turbo_tf32()  # OK for matmuls; accumulators we keep in PyTorch ops as FP32 where applicable
    for N in lengths:
        for name,fn,dtype in [
            ("ELSA-Mixed-Triton(FP16-QK/PV,FP32-acc)", elsa_mixed, torch.float16),
            ("Baseline-FP16-MemEff", baseline_fp16, torch.float16),
        ]:
            q,k,v,_ = make_inputs(B,H,N,D,dtype=dtype)
            met = measure_once(fn, q,k,v, warmup=warmup, trials=trials)
            tokens = B*N
            met.update({
                "method": name, "B":B,"H":H,"D":D,"N":N,
                "tokens_per_s": (tokens / (met["lat_ms_med"]/1e3))
            })
            results.append(met)
    return results

def run_ultra_lean(kernels, N=4096, etas=(1.0,0.5,0.25), B=1, H=8, D=64, warmup=10, trials=50):
    # Leverage the memory-lean Triton kernel and scale BLOCK_N via ELSA_MEM_ETA.
    assert kernels["CAN_triton_mem"] is not None, "Missing CAN_triton_mem in kernel file."
    def make_fn():
        def _fn(q, k, v):
            # CAN_triton_mem(q, k, v, scale, is_causal=False)
            return kernels["CAN_triton_mem"].apply(q.float(), k.float(), v.float(), 1.0 / math.sqrt(q.shape[-1]), False)
        return _fn

    results = []
    set_strict_fp32()
    prev_eta = os.environ.get("ELSA_MEM_ETA")
    for eta in etas:
        os.environ["ELSA_MEM_ETA"] = f"{eta:.4f}"
        q, k, v, _ = make_inputs(B, H, N, D, dtype=torch.float32)
        fn = make_fn()
        met = measure_once(fn, q, k, v, warmup=warmup, trials=trials)
        met.update({"method": f"ELSA-FP32-UL(eta={eta:.2f})", "B": B, "H": H, "D": D, "N": N, "eta": eta})
        results.append(met)

    if prev_eta is None:
        os.environ.pop("ELSA_MEM_ETA", None)
    else:
        os.environ["ELSA_MEM_ETA"] = prev_eta
    return results

def run_varlen_effective_tp(kernels, lengths_dist, samples=200, H=8, D=64, warmup=5, trials=20):
    # Effective throughput: processed non-padding tokens / wall-clock
    assert kernels["CAN_triton_fp32"] is not None
    set_strict_fp32()
    # Generate a batch of (B_i, N_i) with sum B fixed by 1 per sample
    seqs = [random.choices(list(lengths_dist.keys()), weights=list(lengths_dist.values()))[0] for _ in range(samples)]
    total_tokens = sum(seqs)
    # Build one-by-one to avoid padding bias in kernels that don't accept mask
    def elsa_fp32(q,k,v): return kernels["CAN_triton_fp32"].apply(q.float(), k.float(), v.float(), 1.0/math.sqrt(q.shape[-1]))
    lat = []
    for N in seqs:
        q,k,v,_ = make_inputs(B=1,H=H,N=N,D=D,dtype=torch.float32)
        m = measure_once(elsa_fp32, q,k,v, warmup=warmup, trials=trials)
        lat.append(m["lat_ms_med"])
    total_ms = sum(lat)
    eff_tps = total_tokens / (total_ms/1e3)
    return {"samples": samples, "total_tokens": total_tokens, "sum_latency_ms": total_ms, "effective_tokens_per_s": eff_tps}

# -------------------- Report & Plot --------------------
def df_or_list_to_csv(name, rows):
    if pd is None:
        # Minimal CSV writer
        import csv
        keys = sorted(list({k for r in rows for k in r.keys()}))
        with open(name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows: w.writerow(r)
    else:
        import pandas as _pd
        _pd.DataFrame(rows).to_csv(name, index=False)

def plot_latency_vs_len(rows, title, fig_path):
    import pandas as _pd
    df = _pd.DataFrame(rows)
    pivot = df.pivot_table(index="N", columns="method", values="lat_ms_med", aggfunc="median")
    ax = pivot.plot(marker="o")
    ax.set_xlabel("Sequence length N"); ax.set_ylabel("Latency (ms)"); ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(fig_path); plt.close()

def plot_pareto(rows, fig_path):
    import pandas as _pd
    df = _pd.DataFrame(rows)
    ax = plt.figure().gca()
    for eta, sub in df.groupby("eta"):
        ax.scatter(sub["peak_gb"], sub["lat_ms_med"], label=f"eta={eta:.2f}")
    ax.set_xlabel("Peak VRAM (GB)"); ax.set_ylabel("Latency (ms)"); ax.set_title("Ultra-Lean Pareto (eta vs. time/memory)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(fig_path); plt.close()

def write_report(hwinfo, fp32_rows, mix_rows, ul_rows, varlen_res):
    rp = Path("results/report.md")
    rp.write_text(textwrap.dedent(f"""
    # ELSA on A100 — 自動化實驗報告

    **硬體/軟體**：{json.dumps(hwinfo)}

    ## FP32 長序列（B=1,H=8,D=64）
    - 方法：ELSA-Strict-FP32 vs Baseline-FP32-MemEff
    - 度量：Latency(ms, median/p5/p95)、Peak VRAM(GB)、Tokens/s
    - 圖：`figs/latency_vs_len_fp32.png`

    ## Mixed Precision / FP16（B=1,H=8,D=64）
    - 方法：ELSA-Mixed-Triton(FP16-QK/PV,FP32-acc) vs Baseline-FP16-MemEff
    - 圖：`figs/latency_vs_len_mixed.png`

    ## Ultra‑Lean 記憶體旋鈕
    - eta ∈ {1.0,0.5,0.25}；排程以 chunk size 近似雙通/部分重算。
    - 圖：`figs/pareto_ultra_lean.png`

    ## Variable‑length Effective Throughput
    - 定義：有效吞吐＝非 padding token 數 / 總時間（避免 padding 偏誤）
    - 樣本：{varlen_res['samples']}，總 token：{varlen_res['total_tokens']}, 有效吞吐：{varlen_res['effective_tokens_per_s']:.2f} tok/s

    ## 附註
    - Strict‑FP32：已在腳本中禁用 TF32（`torch.backends.cuda.matmul.allow_tf32=False` 等）；如裝有 Nsight Compute，可執行 `check_strict_fp32.sh`，計數器 HMMA/TensorCore 皆應為 0。
    - 單位統一為 ms / tokens/s / GB；數據以 median 為主，並提供 p5/p95 觀察抖動。
*** End Patch
    """).strip()+"\n")

# -------------------- Nsight Compute helper --------------------
def write_ncu_script():
    sh = Path("check_strict_fp32.sh")
    sh.write_text(textwrap.dedent("""\
#!/usr/bin/env bash
set -euo pipefail
if ! command -v ncu &>/dev/null; then
  echo "[Skip] Nsight Compute (ncu) not found."
  exit 0
fi
bash -lc 'python run_elsa_a100.py --only ncu_probe --lengths 4096 --warmup 5 --trials 5'
bash -lc 'ncu --target-processes all --force-overwrite --set full \
  --metrics smsp__inst_executed.pipe_tensor.sum,smsp__sass_thread_inst_executed_hmma.sum \
  --csv -o results/ncu_strict_fp32.csv \
  python - <<PY
import torch, math, os, importlib.util
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
q=torch.randn(1,8,4096,64,device="cuda",dtype=torch.float32)
k=torch.randn(1,8,4096,64,device="cuda",dtype=torch.float32)
v=torch.randn(1,8,4096,64,device="cuda",dtype=torch.float32)
spec=importlib.util.spec_from_file_location("k", os.environ.get("KERNEL_FILE","elsa_kernels.py"))
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
y=mod.CAN_triton_fp32.apply(q,k,v,1.0/math.sqrt(64))
torch.cuda.synchronize()
PY'
echo "[Done] See results/ncu_strict_fp32.csv"
"""))
    sh.chmod(0o755)

# -------------------- Main --------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kernel-file", type=str, required=False, default=os.environ.get("KERNEL_FILE","PATH_TO_SIC_TRITON"))
    p.add_argument("--lengths", type=str, default="512,1024,2048,4096,8192,16384")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--full", action="store_true", help="run full sweep incl. plots and report")
    p.add_argument("--only", type=str, default="", help="subset: fp32|mixed|ul|varlen|ncu_probe")
    args = p.parse_args()

    ensure_dirs(); set_deterministic(args.seed)
    assert torch.cuda.is_available(), "CUDA not available"
    info = device_info()
    print("[Device]", info)

    # Load kernels
    assert os.path.exists(args.kernel_file), f"Kernel file not found: {args.kernel_file}"
    os.environ["KERNEL_FILE"] = args.kernel_file
    kernels = load_kernels(args.kernel_file)

    lengths = [int(x) for x in args.lengths.split(",") if x]

    # Quick probes
    if args.only == "ncu_probe":
        # build a tiny graph to let ncu attach; nothing else to do
        q,k,v,_ = make_inputs(1,8,1024,64,dtype=torch.float32)
        if kernels["CAN_triton_fp32"] is None: 
            print("Missing CAN_triton_fp32 for ncu probe."); return
        _ = kernels["CAN_triton_fp32"].apply(q,k,v,1.0/math.sqrt(64))
        torch.cuda.synchronize(); print("ncu_probe done."); return

    # Run experiments
    rows_fp32 = rows_mixed = rows_ul = None
    if args.full or args.only in ("","fp32"):
        print("[Run] FP32 long sequence sweep...")
        rows_fp32 = run_fp32_long_seq(kernels, lengths)
        df_or_list_to_csv("results/fp32_long_seq.csv", rows_fp32)
        plot_latency_vs_len(rows_fp32, "FP32 Long Sequence", "figs/latency_vs_len_fp32.png")
    if args.full or args.only in ("","mixed"):
        print("[Run] Mixed & FP16...")
        rows_mixed = run_mixed_and_fp16(kernels, lengths)
        df_or_list_to_csv("results/mixed_fp16.csv", rows_mixed)
        plot_latency_vs_len(rows_mixed, "Mixed Precision / FP16", "figs/latency_vs_len_mixed.png")
    if args.full or args.only in ("","ul"):
        print("[Run] Ultra-Lean sweep...")
        rows_ul = run_ultra_lean(kernels, N=4096, etas=(1.0,0.5,0.25))
        df_or_list_to_csv("results/ultra_lean.csv", rows_ul)
        plot_pareto(rows_ul, "figs/pareto_ultra_lean.png")
    if args.full or args.only in ("","varlen"):
        print("[Run] Variable-length effective throughput...")
        varlen_res = run_varlen_effective_tp(kernels, {256:0.1,512:0.2,1024:0.3,2048:0.2,4096:0.2}, samples=50)
        with open("results/varlen.json","w") as f: json.dump(varlen_res, f, indent=2)
    else:
        varlen_res = {"samples":0,"total_tokens":0,"effective_tokens_per_s":0.0}

    # Report
    write_ncu_script()
    if args.full:
        write_report(info, rows_fp32 or [], rows_mixed or [], rows_ul or [], varlen_res)
        print("[OK] Results in ./results and ./figs")
        for pth in ["results/fp32_long_seq.csv","results/mixed_fp16.csv","results/ultra_lean.csv","results/varlen.json","results/report.md"]:
            if os.path.exists(pth): print(" -", pth)

if __name__ == "__main__":
    main()
