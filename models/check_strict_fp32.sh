#!/usr/bin/env bash
set -euo pipefail
if ! command -v ncu &>/dev/null; then
  echo "[Skip] Nsight Compute (ncu) not found."
  exit 0
fi
bash -lc 'python run_elsa_a100.py --only ncu_probe --lengths 4096 --warmup 5 --trials 5'
bash -lc 'ncu --target-processes all --force-overwrite --set full   --metrics smsp__inst_executed.pipe_tensor.sum,smsp__sass_thread_inst_executed_hmma.sum   --csv -o results/ncu_strict_fp32.csv   python - <<PY
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
