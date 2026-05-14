"""Unified entrypoints for selecting ELSA kernel variants."""
from __future__ import annotations

import importlib
import importlib.util
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F


_ROOT = Path(__file__).resolve().parents[2]
_STABLE_SIC_PATH = (
    _ROOT
    / "timm"
    / "elsa_cuda"
    / "versions"
    / "original_20251021_195305"
    / "elsa_cuda"
    / "versions"
    / "original_20251021_195305"
    / "sic_triton.py"
)


@dataclass(frozen=True)
class ElsaEntryPoints:
    name: str
    fp16: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    fp32_strict: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    fp32_turbo: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    fp32_baseline: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]
    note: str


# Clean release defaults to the locked stable ViT/attn-only kernels.
# Future streaming kernels must be opted into explicitly.
_DEFAULT_VARIANT = os.environ.get("ELSA_ENTRYPOINT_VARIANT", "attn_only_stable")
_STABLE_MODULE = None
_FUTURE_MODULE = None
_STRICT_REF_MODULE = None


def set_default_entrypoint_variant(name: str) -> None:
    global _DEFAULT_VARIANT
    _DEFAULT_VARIANT = name


def get_default_entrypoint_variant() -> str:
    return _DEFAULT_VARIANT


def _load_stable_module():
    global _STABLE_MODULE
    if _STABLE_MODULE is not None:
        return _STABLE_MODULE
    if not _STABLE_SIC_PATH.is_file():
        raise FileNotFoundError(f"Stable sic_triton path not found: {_STABLE_SIC_PATH}")
    spec = importlib.util.spec_from_file_location("_elsa_attn_only_stable", _STABLE_SIC_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load stable module at {_STABLE_SIC_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _STABLE_MODULE = module
    return module


def _load_future_module():
    global _FUTURE_MODULE
    if _FUTURE_MODULE is not None:
        return _FUTURE_MODULE
    _FUTURE_MODULE = importlib.import_module("timm.models.elsa_triton")
    return _FUTURE_MODULE


def _load_strict_ref_module():
    global _STRICT_REF_MODULE
    if _STRICT_REF_MODULE is not None:
        return _STRICT_REF_MODULE
    _STRICT_REF_MODULE = importlib.import_module("timm.models.elsa_strict_ref")
    return _STRICT_REF_MODULE


def _sdpa_fallback(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias, is_causal: bool) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias, is_causal=is_causal, dropout_p=0.0)


def _stable_fp16(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, bias=None, is_causal: bool = False) -> torch.Tensor:
    mod = _load_stable_module()
    if bias is not None or is_causal:
        return _sdpa_fallback(q, k, v, bias, is_causal)
    # Locked stable kernel is inference-only for this path (backward returns None).
    # Route training to differentiable Triton autograd path to avoid silent fake speedups.
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        future = _load_future_module()
        scale = 1.0 / math.sqrt(q.shape[-1])
        return future.ELSA_triton.apply(
            q.to(torch.float16),
            k.to(torch.float16),
            v.to(torch.float16),
            scale,
            None,
            is_causal,
        ).to(q.dtype)
    scale = 1.0 / math.sqrt(q.shape[-1])
    q_t = q.to(torch.float16)
    k_t = k.to(torch.float16)
    v_t = v.to(torch.float16)
    out = mod.CAN_triton.apply(q_t, k_t, v_t, scale, None, is_causal)
    return out.to(q.dtype)


def _stable_fp32_new(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, bias=None, is_causal: bool = False) -> torch.Tensor:
    impl = os.environ.get("ELSA_STABLE_FP32_IMPL", "new").strip().lower()
    if impl in {"strict", "strict_core", "strict_ref"}:
        mod = _load_stable_module()
        return mod.can_triton_strict_core_fp32(q, k, v, is_causal=is_causal, bias=bias)
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        future = _load_future_module()
        scale = 1.0 / math.sqrt(q.shape[-1])
        return future.ELSA_triton_fp32.apply(
            q.to(torch.float32),
            k.to(torch.float32),
            v.to(torch.float32),
            scale,
        ).to(q.dtype)
    mod = _load_stable_module()
    return mod.can_triton_new_fp32(q, k, v, is_causal=is_causal, bias=bias)


def _stable_fp32_baseline(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, bias=None, is_causal: bool = False) -> torch.Tensor:
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        future = _load_future_module()
        scale = 1.0 / math.sqrt(q.shape[-1])
        return future.ELSA_triton_fp32.apply(
            q.to(torch.float32),
            k.to(torch.float32),
            v.to(torch.float32),
            scale,
        ).to(q.dtype)
    mod = _load_stable_module()
    return mod.can_triton_baseline_fp32(q, k, v, is_causal=is_causal, bias=bias)


def _future_fp16(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, bias=None, is_causal: bool = False) -> torch.Tensor:
    mod = _load_future_module()
    return mod.elsa_triton_new(q, k, v, is_causal=is_causal, bias=bias, precision="fp16")


def _future_fp32_strict(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, bias=None, is_causal: bool = False) -> torch.Tensor:
    mod = _load_future_module()
    return mod.elsa_triton_new(q, k, v, is_causal=is_causal, bias=bias, precision="fp32")


def _future_fp32_turbo(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, bias=None, is_causal: bool = False) -> torch.Tensor:
    mod = _load_future_module()
    return mod.elsa_triton_new(q, k, v, is_causal=is_causal, bias=bias, precision="tf32")


def _future_fp32_baseline(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, bias=None, is_causal: bool = False) -> torch.Tensor:
    mod = _load_future_module()
    return mod.elsa_triton_baseline_fp32(q, k, v, is_causal=is_causal, bias=bias)


def _strict_ref_block_n(seq_len: int) -> int:
    override = os.environ.get("ELSA_STRICT_REF_BLOCK_N")
    if override is not None:
        try:
            return max(16, int(override))
        except ValueError:
            return 512
    try:
        mod = _load_strict_ref_module()
        return int(mod.default_strict_ref_block_n(int(seq_len), training=False))
    except Exception:
        return 512 if int(seq_len) <= 1024 else 2048


def _strict_ref_fp32(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, bias=None, is_causal: bool = False) -> torch.Tensor:
    mod = _load_stable_module()
    return mod.can_triton_strict_core_fp32(q, k, v, is_causal=is_causal, bias=bias)


def _strict_ref_fp16(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, bias=None, is_causal: bool = False) -> torch.Tensor:
    mod = _load_stable_module()
    strict_fp16 = getattr(mod, "can_triton_strict_core_fp16", None)
    if strict_fp16 is None:
        raise RuntimeError("strict_core_ref fp16 entrypoint unavailable in stable sic_triton module.")
    return strict_fp16(q, k, v, is_causal=is_causal, bias=bias)


def get_entrypoints(variant: Optional[str] = None) -> ElsaEntryPoints:
    name = (variant or _DEFAULT_VARIANT).lower()
    if name in {"attn_only_stable", "stable"}:
        return ElsaEntryPoints(
            name="attn_only_stable",
            fp16=_stable_fp16,
            fp32_strict=_stable_fp32_new,
            fp32_turbo=_stable_fp32_new,
            fp32_baseline=_stable_fp32_baseline,
            note="Locked sic_triton baseline; FP32 uses can_triton_new_fp32.",
        )
    if name in {"future_exp", "exp", "streaming"}:
        return ElsaEntryPoints(
            name="future_exp",
            fp16=_future_fp16,
            fp32_strict=_future_fp32_strict,
            fp32_turbo=_future_fp32_turbo,
            fp32_baseline=_future_fp32_baseline,
            note="Streaming-capable kernels from timm.models.elsa_triton.",
        )
    if name in {"strict_core_ref", "strict_ref", "strict_core"}:
        return ElsaEntryPoints(
            name="strict_core_ref",
            fp16=_strict_ref_fp16,
            fp32_strict=_strict_ref_fp32,
            fp32_turbo=_strict_ref_fp32,
            fp32_baseline=_stable_fp32_baseline,
            note="Strict-core path: explicit monoid summaries + Hillis-Steele scan for fp32; fp16 uses strict q-block scan with fp16/bf16 I/O storage.",
        )
    raise ValueError(f"Unknown entrypoint variant '{variant}'.")
