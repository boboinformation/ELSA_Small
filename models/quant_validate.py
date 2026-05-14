#!/usr/bin/env python3
"""Precision comparison — FP32 (PyTorch) / FP16 / INT8 (TensorRT).

Flow:
  fp32 : FP32 checkpoint → PyTorch inference → validate Top-1/5 + bench latency
  fp16 : FP32 checkpoint → ONNX → TRT FP16 engine → validate + bench
  int8 : FP32 checkpoint → ONNX → TRT INT8 engine (calibrate) → validate + bench

Usage:
  # FP32 baseline (no TRT)
  python models/quant_validate.py \\
      --model sic_small_patch4_256_int_win \\
      --checkpoint /path/to/model_best.pth.tar \\
      --data-dir /path/to/ILSVRC2012 \\
      --quant-mode fp32

  # FP16 TRT
  python models/quant_validate.py \\
      --model sic_small_patch4_256_int_win \\
      --checkpoint /path/to/model_best.pth.tar \\
      --data-dir /path/to/ILSVRC2012 \\
      --quant-mode fp16

  # elsa_swin needs model-kwargs
  python models/quant_validate.py \\
      --model elsa_small_window8_256 \\
      --model-kwargs elsa_backend=pytorch \\
      --quant-mode fp16 ...
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# TRT is only needed for fp16/int8 modes — import lazily to avoid errors on fp32-only runs
try:
    import tensorrt as trt
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    _TRT_AVAILABLE = True
except ImportError:
    _TRT_AVAILABLE = False


# ── data loading ──────────────────────────────────────────────────────────────

def make_loader(data_dir: str, split: str, img_size: int, batch_size: int, workers: int,
                crop_pct: float = 0.9, interpolation: str = "bicubic"):
    import timm.data as td
    dataset = td.create_dataset("", root=data_dir, split=split, is_training=False)
    return td.create_loader(
        dataset,
        input_size=(3, img_size, img_size),
        batch_size=batch_size,
        is_training=False,
        use_prefetcher=False,
        num_workers=workers,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=interpolation,
        crop_pct=crop_pct,
    )


def _resolve_data_config(model_name: str, model_kwargs: str) -> dict:
    """從 model 的 pretrained_cfg 讀取正確的前處理設定。"""
    import timm
    import timm.data as td
    kwargs: dict = {}
    if model_kwargs:
        for kv in model_kwargs.split(","):
            k, v = kv.strip().split("=", 1)
            kwargs[k.strip()] = v.strip()
    tmp = timm.create_model(model_name, pretrained=False, **kwargs)
    cfg = td.resolve_data_config({}, model=tmp)
    del tmp
    return cfg


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(model_name: str, checkpoint: str, model_kwargs: str, num_classes: int = 1000):
    import timm
    kwargs: dict = {}
    if model_kwargs:
        for kv in model_kwargs.split(","):
            k, v = kv.strip().split("=", 1)
            kwargs[k.strip()] = v.strip()

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes, **kwargs)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state.get("model", state))
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


# ── ONNX export (CPU — avoids CUDA context conflict with TRT) ─────────────────

def export_onnx(model: torch.nn.Module, onnx_path: str, img_size: int) -> None:
    if Path(onnx_path).exists():
        print(f"[Quant] Reusing ONNX: {onnx_path}")
        return
    print(f"[Quant] Exporting ONNX → {onnx_path}")
    model = model.cpu()
    dummy = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, onnx_path,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
    print("[Quant] ONNX export done.")


# ── INT8 calibrator (torch tensors, no pycuda) ────────────────────────────────

class _Int8Calibrator(trt.IInt8EntropyCalibrator2):
    """Feed a subset of the val set to TRT INT8 calibration."""

    def __init__(self, loader, calib_batches: int, cache_path: str):
        super().__init__()
        self._it = iter(loader)
        self._n = calib_batches
        self._cache = cache_path
        self._count = 0
        self._batch_size = loader.batch_size
        self._buf: torch.Tensor | None = None

    def get_batch_size(self) -> int:
        return self._batch_size

    def get_batch(self, names):
        if self._count >= self._n:
            return None
        try:
            imgs, _ = next(self._it)
        except StopIteration:
            return None
        self._buf = imgs.contiguous().cuda()
        self._count += 1
        print(f"[Quant] Calibrating batch {self._count}/{self._n} ...", end="\r")
        return [self._buf.data_ptr()]

    def read_calibration_cache(self):
        if os.path.exists(self._cache):
            with open(self._cache, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self._cache, "wb") as f:
            f.write(cache)
        print(f"\n[Quant] Calibration cache saved: {self._cache}")


# ── TRT engine build ──────────────────────────────────────────────────────────

def build_engine(
    onnx_path: str,
    engine_path: str,
    quant_mode: str,
    img_size: int,
    val_batch: int,
    calib_loader=None,
    calib_batches: int = 8,
    calib_cache: str = "",
) -> bytes:
    if Path(engine_path).exists():
        print(f"[Quant] Reusing TRT engine: {engine_path}")
        with open(engine_path, "rb") as f:
            return f.read()

    print(f"[Quant] Building TRT engine ({quant_mode}) ...")
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  [ERR] {parser.get_error(i)}")
            raise RuntimeError("ONNX parsing failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input",
        min=(1,        3, img_size, img_size),
        opt=(val_batch, 3, img_size, img_size),
        max=(val_batch, 3, img_size, img_size),
    )
    config.add_optimization_profile(profile)

    if quant_mode == "fp16":
        if not builder.platform_has_fast_fp16:
            print("[WARN] GPU has no fast FP16 path.")
        config.set_flag(trt.BuilderFlag.FP16)

    elif quant_mode == "int8":
        if not builder.platform_has_fast_int8:
            print("[WARN] GPU has no fast INT8 path.")
        config.set_flag(trt.BuilderFlag.INT8)
        if calib_loader is None:
            raise ValueError("--data-dir is required for INT8 calibration")
        config.int8_calibrator = _Int8Calibrator(calib_loader, calib_batches, calib_cache)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("TRT engine build returned None — check ONNX and GPU support")

    raw = bytes(engine_bytes)
    with open(engine_path, "wb") as f:
        f.write(raw)
    print(f"[Quant] Engine saved: {engine_path}")
    return raw


# ── PyTorch runner (no TRT, supports fp32/fp16) ───────────────────────────────

_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
}

class PyTorchRunner:
    """Direct PyTorch inference at a specified dtype — no TRT quantization."""

    def __init__(self, model: torch.nn.Module, dtype: str = "fp32"):
        torch_dtype = _DTYPE_MAP.get(dtype, torch.float32)
        self.model = model.cuda().to(torch_dtype).eval()
        self.torch_dtype = torch_dtype

    def infer(self, images: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(images, np.float32)).cuda().to(self.torch_dtype)
        with torch.no_grad():
            return self.model(x).float().cpu().numpy()


# ── TRT runner (torch tensors, no pycuda) ─────────────────────────────────────

class TRTRunner:
    """Execute a TRT engine using torch CUDA tensors as buffers."""

    def __init__(self, engine_bytes: bytes):
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.ctx = self.engine.create_execution_context()

        self.in_name = self.out_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.in_name = name
            else:
                self.out_name = name

        self.num_classes = int(self.engine.get_tensor_shape(self.out_name)[-1])

    def infer(self, images: np.ndarray) -> np.ndarray:
        """images: float32 NCHW numpy. Returns logits (N, num_classes)."""
        B, C, H, W = images.shape
        in_t  = torch.from_numpy(np.ascontiguousarray(images, np.float32)).cuda()
        out_t = torch.empty((B, self.num_classes), dtype=torch.float32, device="cuda")

        self.ctx.set_input_shape(self.in_name, (B, C, H, W))
        self.ctx.set_tensor_address(self.in_name,  in_t.data_ptr())
        self.ctx.set_tensor_address(self.out_name, out_t.data_ptr())
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()

        return out_t.cpu().numpy()


# ── accuracy validation ───────────────────────────────────────────────────────

def validate_accuracy(runner: TRTRunner, loader) -> tuple[float, float]:
    top1 = top5 = total = 0
    for images, labels in loader:
        logits = runner.infer(images.numpy().astype(np.float32))
        preds  = np.argsort(logits, axis=1)[:, ::-1]
        lbl    = labels.numpy()
        top1  += int((preds[:, 0] == lbl).sum())
        top5  += int(np.any(preds[:, :5] == lbl[:, None], axis=1).sum())
        total += len(lbl)
        print(f"[Quant] Validate {total} samples ...", end="\r")
    print()
    return 100.0 * top1 / total, 100.0 * top5 / total


# ── latency benchmark (torch CUDA events) ─────────────────────────────────────

def run_benchmark(
    runner: TRTRunner,
    img_size: int,
    bench_batch: int,
    warmup: int,
    bench_iter: int,
) -> tuple[float, float, float]:
    dummy = np.random.randn(bench_batch, 3, img_size, img_size).astype(np.float32)

    print(f"[Quant] Warming up ({warmup} iters) ...")
    for _ in range(warmup):
        runner.infer(dummy)

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)
    latencies = []

    print(f"[Quant] Benchmarking ({bench_iter} iters) ...")
    for _ in range(bench_iter):
        start_ev.record()
        runner.infer(dummy)
        end_ev.record()
        torch.cuda.synchronize()
        latencies.append(start_ev.elapsed_time(end_ev))

    arr     = np.array(latencies)
    mean_ms = float(arr.mean())
    std_ms  = float(arr.std())
    fps     = bench_batch / (mean_ms / 1000.0)
    return mean_ms, std_ms, fps


# ── results CSV ───────────────────────────────────────────────────────────────

def save_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_header = not Path(path).exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"[Quant] Results → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Precision comparison: fp32/fp16/int8 validate + bench")
    p.add_argument("--model",            required=True)
    p.add_argument("--checkpoint",       default="",
                   help="Checkpoint path. Required if ONNX/engine not yet cached.")
    p.add_argument("--model-kwargs",     default="",
                   help="Comma-separated k=v pairs, e.g. elsa_backend=pytorch")
    p.add_argument("--num-classes",      type=int, default=1000)
    p.add_argument("--data-dir",         default="")
    p.add_argument("--split",            default="val")
    p.add_argument("--ckpt-dtype",       default="fp32", choices=["fp32", "fp16"],
                   help="Dtype the checkpoint weights are stored in")
    p.add_argument("--infer-dtype",      default="fp16", choices=["fp32", "fp16", "int8"],
                   help="Target inference dtype. Strategy is auto-determined from ckpt-dtype + infer-dtype")
    p.add_argument("--task",             default="both",
                   choices=["validate", "bench", "both"])
    p.add_argument("--img-size",         type=int, default=256)
    p.add_argument("--batch-size",       type=int, default=64,
                   help="Validation batch size")
    p.add_argument("--bench-batch-size", type=int, default=1,
                   help="Benchmark (latency) batch size")
    p.add_argument("--workers",          type=int, default=4)
    p.add_argument("--calib-batches",    type=int, default=8,
                   help="INT8 calibration batches (calib_batches × batch_size images)")
    p.add_argument("--engine-dir",
                   default=str(ROOT / "script" / "Engine_cache"))
    p.add_argument("--results-file",     default="")
    p.add_argument("--warmup",           type=int, default=50)
    p.add_argument("--bench-iter",       type=int, default=200)
    return p.parse_args()


def determine_strategy(ckpt_dtype: str, infer_dtype: str) -> str:
    """Auto-determine execution strategy from checkpoint dtype and target inference dtype.

    Returns one of:
      pytorch_fp32  — load checkpoint, run PyTorch FP32 inference (no quantization)
      pytorch_fp16  — load checkpoint, cast to fp16, run PyTorch FP16 inference (no TRT)
      trt_fp16      — FP32 checkpoint → ONNX → TRT FP16 engine (precision reduction via TRT)
      trt_int8      — FP32 checkpoint → ONNX → TRT INT8 engine (calibration required)
    """
    if infer_dtype == "int8":
        return "trt_int8"           # INT8 always needs TRT regardless of ckpt dtype
    if infer_dtype == "fp16":
        if ckpt_dtype == "fp32":
            return "trt_fp16"       # precision reduction: quantize via TRT
        return "pytorch_fp16"       # same or higher ckpt precision: direct inference
    # infer_dtype == "fp32"
    return "pytorch_fp32"           # fp32 output: always PyTorch, no quantization needed


def main():
    args = parse_args()
    do_val   = args.task in ("validate", "both")
    do_bench = args.task in ("bench",    "both")

    if do_val and not args.data_dir:
        raise ValueError("--data-dir is required for validation")

    Path(args.engine_dir).mkdir(parents=True, exist_ok=True)

    # 從 model pretrained_cfg 取前處理設定，與 validate.py 保持一致
    _dcfg        = _resolve_data_config(args.model, args.model_kwargs)
    _crop_pct    = _dcfg.get("crop_pct",      0.9)
    _interp      = _dcfg.get("interpolation", "bicubic")
    print(f"[Quant] data_config: crop_pct={_crop_pct}  interpolation={_interp}")

    strategy    = determine_strategy(args.ckpt_dtype, args.infer_dtype)
    stem        = f"{args.model}_{args.ckpt_dtype}to{args.infer_dtype}_bs{args.batch_size}_img{args.img_size}"
    onnx_path   = os.path.join(args.engine_dir, f"{args.model}_{args.ckpt_dtype}.onnx")
    engine_path = os.path.join(args.engine_dir, f"{stem}.trt")
    calib_cache = os.path.join(args.engine_dir, f"{args.model}_int8_calib.cache")

    print(f"[Quant] ckpt={args.ckpt_dtype} → infer={args.infer_dtype} → strategy={strategy}")

    # ── PyTorch direct inference (no TRT) ────────────────────────────────────
    if strategy.startswith("pytorch"):
        if not args.checkpoint:
            raise ValueError("--checkpoint is required")
        model = load_model(args.model, args.checkpoint, args.model_kwargs, args.num_classes)
        infer_dtype = strategy.split("_")[1]   # "fp32" or "fp16"
        runner = PyTorchRunner(model, dtype=infer_dtype)

    # ── TRT quantization path ─────────────────────────────────────────────────
    else:
        if not _TRT_AVAILABLE:
            raise RuntimeError("TensorRT not installed. Install tensorrt-cu12.")

        trt_mode = "fp16" if strategy == "trt_fp16" else "int8"

        # ONNX export (always in fp32 for TRT, cast ckpt if needed)
        if not Path(onnx_path).exists():
            if not args.checkpoint:
                raise ValueError(
                    f"ONNX not found at {onnx_path}. "
                    "Provide --checkpoint to export ONNX on first run."
                )
            print(f"[Quant] Loading checkpoint (ckpt_dtype={args.ckpt_dtype}) for ONNX export ...")
            model = load_model(args.model, args.checkpoint, args.model_kwargs, args.num_classes)
            model = model.float()   # TRT always needs fp32 ONNX
            export_onnx(model, onnx_path, args.img_size)
            del model

        # INT8 calibration loader
        calib_loader = None
        if trt_mode == "int8" and not Path(calib_cache).exists():
            if not args.data_dir:
                raise ValueError("--data-dir is required for INT8 calibration")
            calib_loader = make_loader(
                args.data_dir, args.split, args.img_size,
                args.batch_size, args.workers,
                crop_pct=_crop_pct, interpolation=_interp,
            )

        engine_bytes = build_engine(
            onnx_path, engine_path, trt_mode,
            img_size=args.img_size, val_batch=args.batch_size,
            calib_loader=calib_loader, calib_batches=args.calib_batches,
            calib_cache=calib_cache,
        )
        runner = TRTRunner(engine_bytes)

    rows: list[dict] = []

    # ── validate ─────────────────────────────────────────────────────────────
    if do_val:
        val_loader = make_loader(
            args.data_dir, args.split, args.img_size,
            args.batch_size, args.workers,
            crop_pct=_crop_pct, interpolation=_interp,
        )
        top1, top5 = validate_accuracy(runner, val_loader)
        print(f"[Quant] Top-1: {top1:.3f}%  Top-5: {top5:.3f}%")
        rows.append(dict(
            model=args.model, ckpt_dtype=args.ckpt_dtype, infer_dtype=args.infer_dtype,
            strategy=strategy, task="validate",
            batch_size=args.batch_size, img_size=args.img_size,
            top1=f"{top1:.4f}", top5=f"{top5:.4f}",
            latency_mean_ms="", latency_std_ms="", throughput_fps="",
        ))

    # ── benchmark ────────────────────────────────────────────────────────────
    if do_bench:
        mean_ms, std_ms, fps = run_benchmark(
            runner, args.img_size, args.bench_batch_size,
            args.warmup, args.bench_iter,
        )
        print(f"[Quant] Latency: {mean_ms:.3f} ± {std_ms:.3f} ms  "
              f"({fps:.1f} fps, batch={args.bench_batch_size})")
        rows.append(dict(
            model=args.model, ckpt_dtype=args.ckpt_dtype, infer_dtype=args.infer_dtype,
            strategy=strategy, task="bench",
            batch_size=args.bench_batch_size, img_size=args.img_size,
            top1="", top5="",
            latency_mean_ms=f"{mean_ms:.4f}",
            latency_std_ms=f"{std_ms:.4f}",
            throughput_fps=f"{fps:.2f}",
        ))

    if args.results_file and rows:
        save_csv(args.results_file, rows)


if __name__ == "__main__":
    main()
