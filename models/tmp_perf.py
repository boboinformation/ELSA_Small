import os
import torch
from elsa_cuda.elsa_cuda.elsa import elsa_forward


def measure_once(fn, warm=10, iters=50):
    torch.cuda.synchronize()
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def run_case(shape, config):
    B, H, N, D = shape
    block_n, warps, stages, vec, tf32 = config
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    os.environ["ELSACUDA_BLOCK_N"] = str(block_n)
    os.environ["ELSACUDA_NUM_WARPS"] = str(warps)
    os.environ["ELSACUDA_NUM_STAGES"] = str(stages)
    os.environ["ELSACUDA_VEC"] = str(vec)
    os.environ["ELSACUDA_FORCE_TF32"] = str(int(tf32))

    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    fused = elsa_forward(q, k, v, algo="cuda_new")
    diff = (ref - fused).abs().max().item()

    def ref_call():
        torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

    def fused_call():
        elsa_forward(q, k, v, algo="cuda_new")

    ref_ms = measure_once(ref_call)
    fused_ms = measure_once(fused_call)
    return ref_ms, fused_ms, diff


if __name__ == "__main__":
    shapes = [
        (1, 6, 1024, 80),
        (1, 6, 2048, 96),
        (4, 12, 2048, 96),
    ]
    config = (128, 4, 2, 4, 0)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    results = []
    for shape in shapes:
        ref_ms, fused_ms, diff = run_case(shape, config)
        speedup = ref_ms / fused_ms if fused_ms > 0 else float("nan")
        print(f"shape={shape}, sdpa_math={ref_ms:.3f} ms, cuda_new={fused_ms:.3f} ms, diff={diff:.2e}, speedup={speedup:.3f}x")
        results.append((shape, ref_ms, fused_ms, diff, speedup))
    torch.save(results, "tmp_perf.pt")
