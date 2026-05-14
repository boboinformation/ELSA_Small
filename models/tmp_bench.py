import os
import torch
from torch.backends.cuda import SDPBackend
from torch.nn.attention import sdpa_kernel
from elsa_cuda.elsa_cuda.elsa import elsa_forward

def get_math_ctx():
    try:
        return sdpa_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False)
    except TypeError:
        return sdpa_kernel(backends=SDPBackend.MATH, set_priority=True)

def time_once(func, reps=10):
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(reps):
        func()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps

def run_case(shape, config):
    B,H,N,D = shape
    block_n, warps, stages, vec, tf32 = config
    torch.manual_seed(0)
    q = torch.randn(B,H,N,D, device='cuda', dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    os.environ['ELSACUDA_BLOCK_N']=str(block_n)
    os.environ['ELSACUDA_NUM_WARPS']=str(warps)
    os.environ['ELSACUDA_NUM_STAGES']=str(stages)
    os.environ['ELSACUDA_VEC']=str(vec)
    os.environ['ELSACUDA_FORCE_TF32']=str(int(tf32))
    ctx = get_math_ctx()
    with ctx:
        ref = torch.nn.functional.scaled_dot_product_attention(q,k,v, dropout_p=0.0)
    cand = elsa_forward(q,k,v, algo='cuda_new')
    diff = (ref - cand).abs().max().item()
    def ref_call():
        with ctx:
            torch.nn.functional.scaled_dot_product_attention(q,k,v, dropout_p=0.0)
    def cand_call():
        elsa_forward(q,k,v, algo='cuda_new')
    ref_ms = time_once(ref_call, reps=5)
    cand_ms = time_once(cand_call, reps=5)
    return ref_ms, cand_ms, diff

if __name__=='__main__':
    shape=(1,6,1024,80)
    config=(128,4,2,4,0)
    ref_ms, cand_ms, diff = run_case(shape, config)
    print(f"shape={shape} ref={ref_ms:.3f}ms cand={cand_ms:.3f}ms diff={diff:.2e} speedup={ref_ms/cand_ms:.3f}x")
