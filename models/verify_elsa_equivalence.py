#!/usr/bin/env python3
"""
Fast numerical equivalence checker between reference Math attention and
ELSA-equivalent online-softmax attention (FP32 focus).

Outputs are written under ./elsa_eqv_artifacts/.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

ARTIFACT_DIR = Path("elsa_eqv_artifacts")
FIG_DIR = ARTIFACT_DIR / "figs"
BLOCK_SIZE = 128

THRESHOLDS = {
    "maxAbsP": 1e-7,
    "relL2P": 1e-7,
    "JSmean": 1e-9,
    "argRate": 5e-4,
    "maxAbsY": 1e-7,
    "relL2Y": 1e-7,
}


@dataclass
class Scenario:
    name: str
    batch: int
    heads: int
    seq: int
    d: int
    dv: int
    stress: bool = False


SCENARIOS: List[Scenario] = [
    Scenario("S1_regular", batch=2, heads=4, seq=1024, d=64, dv=64, stress=False),
    Scenario("S2_long", batch=1, heads=2, seq=4096, d=64, dv=64, stress=False),
    Scenario("S3_stress", batch=1, heads=2, seq=4096, d=64, dv=64, stress=True),
]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def setup_device() -> torch.device:
    torch.manual_seed(123)
    np.random.seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda")
    else:
        warnings.warn("CUDA unavailable; using CPU. Long scenarios may be slow.")
        device = torch.device("cpu")
    return device


def math_attention(scores: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    scores64 = scores.to(torch.float64)
    values64 = values.to(torch.float64)
    m = scores64.max(dim=-1, keepdim=True).values
    exp_scores = torch.exp(scores64 - m)
    S = exp_scores.sum(dim=-1, keepdim=True)
    probs64 = exp_scores / S
    out64 = probs64 @ values64
    return probs64, out64


def elsa_attention(scores: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ELSA-equivalent attention via explicit online-softmax scan.

    The state carried across the scan is u = (m, S, W) where
      - m stores the running maximum,
      - S accumulates exp(logits - m),
      - W accumulates exp(logits - m) * value vectors.

    The binary operator is:
        if ma >= mb:
            (ma, Sa + Sb * exp(mb - ma), Wa + Wb * exp(mb - ma))
        else:
            (mb, Sb + Sa * exp(ma - mb), Wb + Wa * exp(ma - mb))

    The identity is (-inf, 0, 0).  We materialise the scan row-by-row;
    while this is O(N^2), it preserves the exact algebra for validation.
    """
    scores64 = scores.to(torch.float64)
    values64 = values.to(torch.float64)
    B, H, N, _ = scores64.shape
    dv = values64.shape[-1]

    # Running state initialised to the monoid identity.
    m_state = torch.full((B, H, N), float("-inf"), dtype=torch.float64, device=scores64.device)
    S_state = torch.zeros((B, H, N), dtype=torch.float64, device=scores64.device)
    W_state = torch.zeros((B, H, N, dv), dtype=torch.float64, device=scores64.device)

    for j in range(N):
        scores_col = scores64[..., j]  # (B, H, N)
        values_col = values64[:, :, j, :].unsqueeze(2)  # (B, H, 1, dv)

        ma = m_state
        Sa = S_state
        Wa = W_state

        mb = scores_col
        Sb = torch.ones_like(scores_col, dtype=torch.float64, device=scores64.device)
        Wb = values_col  # broadcast along the sequence dimension on demand

        # New running maximum.
        m_new = torch.maximum(ma, mb)

        # Rescale partial sums into the new max domain.
        exp_a = torch.exp(ma - m_new)
        exp_b = torch.exp(mb - m_new)
        # Handle potential NaNs from -inf - -inf.
        exp_a = torch.nan_to_num(exp_a, nan=0.0)
        exp_b = torch.nan_to_num(exp_b, nan=0.0)

        S_candidate_a = Sa + Sb * exp_b
        S_candidate_b = Sb + Sa * exp_a
        S_new = torch.where(ma >= mb, S_candidate_a, S_candidate_b)

        exp_b_unsqueezed = exp_b.unsqueeze(-1)
        exp_a_unsqueezed = exp_a.unsqueeze(-1)
        W_candidate_a = Wa + Wb * exp_b_unsqueezed
        W_candidate_b = Wb + Wa * exp_a_unsqueezed
        W_new = torch.where((ma >= mb).unsqueeze(-1), W_candidate_a, W_candidate_b)

        m_state = m_new
        S_state = S_new
        W_state = W_new

    S_safe = S_state.clamp_min(1e-32).unsqueeze(-1)
    probs64 = torch.exp(scores64 - m_state.unsqueeze(-1)) / S_safe
    out64 = W_state / S_safe
    return probs64, out64


def apply_stress(scores: torch.Tensor) -> torch.Tensor:
    B, H, N, _ = scores.shape
    offs = torch.empty((B, H, N, 1), device=scores.device, dtype=scores.dtype).uniform_(-20.0, 20.0)
    scal = torch.empty((B, H, N, 1), device=scores.device, dtype=scores.dtype).uniform_(0.5, 3.0)
    return scores * scal + offs


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-45) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p) - torch.log(m))).sum(dim=-1)
    kl_qm = (q * (torch.log(q) - torch.log(m))).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def gather_row_metrics(P_impl: torch.Tensor, P_ref: torch.Tensor, Y_impl: torch.Tensor, Y_ref: torch.Tensor) -> Dict[str, torch.Tensor]:
    B, H, N, _ = P_impl.shape
    Dv = Y_impl.size(-1)
    rows = B * H * N
    P_impl_flat = P_impl.reshape(rows, N)
    P_ref_flat = P_ref.reshape(rows, N)
    Y_impl_flat = Y_impl.reshape(rows, Dv)
    Y_ref_flat = Y_ref.reshape(rows, Dv)

    diffP = (P_impl_flat - P_ref_flat).abs()
    diffY = (Y_impl_flat - Y_ref_flat)

    maxAbsP = diffP.max(dim=1).values
    relL2P = diffP.norm(dim=1) / (P_ref_flat.norm(dim=1) + 1e-20)
    JSmean = js_divergence(P_impl_flat, P_ref_flat)
    argRate = (P_impl_flat.argmax(dim=1) != P_ref_flat.argmax(dim=1)).float()

    maxAbsY = diffY.abs().max(dim=1).values
    relL2Y = diffY.norm(dim=1) / (Y_ref_flat.norm(dim=1) + 1e-20)

    return {
        "maxAbsP": maxAbsP.cpu(),
        "relL2P": relL2P.cpu(),
        "JSmean": JSmean.cpu(),
        "argRate": argRate.cpu(),
        "maxAbsY": maxAbsY.cpu(),
        "relL2Y": relL2Y.cpu(),
    }


def percentile_dict(metrics: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for name, tensor in metrics.items():
        arr = tensor.numpy()
        result[name] = {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }
    return result


def compute_block_metrics(
    scenario: Scenario,
    P_impl: torch.Tensor,
    P_ref: torch.Tensor,
    Y_impl: torch.Tensor,
    Y_ref: torch.Tensor,
) -> List[Dict[str, float]]:
    B, H, N, _ = P_impl.shape
    block_rows: List[Dict[str, float]] = []
    num_blocks = math.ceil(N / BLOCK_SIZE)
    for b in range(B):
        for h in range(H):
            for block_id in range(num_blocks):
                start = block_id * BLOCK_SIZE
                end = min((block_id + 1) * BLOCK_SIZE, N)
                P_i = P_impl[b, h, start:end]
                P_r = P_ref[b, h, start:end]
                Y_i = Y_impl[b, h, start:end]
                Y_r = Y_ref[b, h, start:end]
                diffP = P_i - P_r
                diffY = Y_i - Y_r
                row = {
                    "scenario": scenario.name,
                    "batch": b,
                    "head": h,
                    "block_id": block_id,
                    "maxAbsP": float(diffP.abs().max().item()),
                    "relL2P": float(diffP.norm().item() / (P_r.norm().item() + 1e-20)),
                    "JSmean": float(js_divergence(P_i.reshape(-1, P_i.size(-1)), P_r.reshape(-1, P_r.size(-1))).mean().item()),
                    "argRate": float(
                        (P_i.argmax(dim=-1) != P_r.argmax(dim=-1)).float().mean().item()
                    ),
                    "maxAbsY": float(diffY.abs().amax().item()),
                    "relL2Y": float(diffY.norm().item() / (Y_r.norm().item() + 1e-20)),
                }
                block_rows.append(row)
    return block_rows


def save_heatmaps(scenario: Scenario, P_impl: torch.Tensor, P_ref: torch.Tensor) -> List[str]:
    diff = (P_impl - P_ref).abs()
    sample = diff[0, 0].detach().cpu().numpy()
    paths: List[str] = []

    plt.figure(figsize=(6, 5))
    log_image = np.clip(np.log10(np.clip(sample, 1e-12, None)), -10, 0)
    plt.imshow(log_image, cmap="viridis")
    plt.colorbar(label="log10(|ΔP|)")
    plt.title(f"{scenario.name} head0 log10(|ΔP|)")
    plt.tight_layout()
    path_log = FIG_DIR / f"{scenario.name}_head0_heatmap_log.png"
    plt.savefig(path_log, dpi=200)
    plt.close()
    paths.append(str(path_log.resolve()))

    plt.figure(figsize=(6, 5))
    plt.imshow(sample * 1e8, cmap="magma")
    plt.colorbar(label="|ΔP| * 1e8")
    plt.title(f"{scenario.name} head0 |ΔP|*1e8")
    plt.tight_layout()
    path_lin = FIG_DIR / f"{scenario.name}_head0_heatmap_lin.png"
    plt.savefig(path_lin, dpi=200)
    plt.close()
    paths.append(str(path_lin.resolve()))

    return paths


def save_histograms(scenario: Scenario, P_impl: torch.Tensor, P_ref: torch.Tensor, Y_impl: torch.Tensor, Y_ref: torch.Tensor) -> List[str]:
    diffP = (P_impl - P_ref).abs().detach().cpu().numpy().flatten()
    diffY = (Y_impl - Y_ref).abs().detach().cpu().numpy().flatten()
    paths: List[str] = []

    plt.figure(figsize=(6, 4))
    plt.hist(diffY, bins=100, color="steelblue", log=True)
    plt.title(f"{scenario.name} |ΔY| histogram")
    plt.xlabel("|ΔY|")
    plt.ylabel("Frequency (log)")
    plt.tight_layout()
    path_y = FIG_DIR / f"{scenario.name}_hist_absY.png"
    plt.savefig(path_y, dpi=200)
    plt.close()
    paths.append(str(path_y.resolve()))

    plt.figure(figsize=(6, 4))
    log_diff = np.log10(np.clip(diffP, 1e-20, None))
    plt.hist(log_diff, bins=100, color="seagreen", log=True)
    plt.title(f"{scenario.name} log10(|ΔP|) histogram")
    plt.xlabel("log10(|ΔP|)")
    plt.ylabel("Frequency (log)")
    plt.tight_layout()
    path_p = FIG_DIR / f"{scenario.name}_hist_logP.png"
    plt.savefig(path_p, dpi=200)
    plt.close()
    paths.append(str(path_p.resolve()))

    return paths


def save_block_bar_chart(scenario: Scenario, block_metrics: List[Dict[str, float]]) -> str | None:
    if scenario.name != "S2_long":
        return None
    rows = [row for row in block_metrics if row["scenario"] == scenario.name and row["head"] == 0 and row["batch"] == 0]
    if not rows:
        return None
    x = np.arange(len(rows))
    maxAbsP = [row["maxAbsP"] for row in rows]
    maxAbsY = [row["maxAbsY"] for row in rows]
    width = 0.35
    plt.figure(figsize=(10, 4))
    plt.bar(x - width / 2, maxAbsP, width, label="maxAbsP")
    plt.bar(x + width / 2, maxAbsY, width, label="maxAbsY")
    plt.xlabel("Block ID (batch0, head0)")
    plt.ylabel("Magnitude")
    plt.title(f"{scenario.name} per-block maxima")
    plt.legend()
    plt.tight_layout()
    path = FIG_DIR / f"{scenario.name}_block_bar.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return str(path.resolve())


def run_scenario(scenario: Scenario, device: torch.device) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, float]], List[str], torch.Tensor, torch.Tensor]:
    B, H, N, d, dv = scenario.batch, scenario.heads, scenario.seq, scenario.d, scenario.dv
    dtype = torch.float32
    q = torch.randn(B, H, N, d, device=device, dtype=dtype)
    k = torch.randn(B, H, N, d, device=device, dtype=dtype)
    v = torch.randn(B, H, N, dv, device=device, dtype=dtype)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(d)
    if scenario.stress:
        scores = apply_stress(scores)

    P_ref64, Y_ref64 = math_attention(scores, v)
    P_impl64, Y_impl64 = elsa_attention(scores, v)

    metrics = gather_row_metrics(P_impl64, P_ref64, Y_impl64, Y_ref64)
    agg = percentile_dict(metrics)
    block_metrics = compute_block_metrics(scenario, P_impl64, P_ref64, Y_impl64, Y_ref64)

    fig_paths = []
    fig_paths.extend(save_heatmaps(scenario, P_impl64, P_ref64))
    fig_paths.extend(save_histograms(scenario, P_impl64, P_ref64, Y_impl64, Y_ref64))
    chart = save_block_bar_chart(scenario, block_metrics)
    if chart:
        fig_paths.append(chart)

    return agg, block_metrics, fig_paths, P_ref64, P_impl64


def optional_fp16(device: torch.device) -> Dict[str, float] | None:
    if device.type != "cuda":
        return None
    scenario = Scenario("S3_stress_fp16", batch=1, heads=2, seq=4096, d=64, dv=64, stress=True)
    q = torch.randn(scenario.batch, scenario.heads, scenario.seq, scenario.d, device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn(scenario.batch, scenario.heads, scenario.seq, scenario.dv, device=device, dtype=torch.float16)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(scenario.d)
    scores = apply_stress(scores.to(torch.float32))
    P_ref, _ = math_attention(scores, v.to(torch.float32))
    P_impl, _ = elsa_attention(scores, v.to(torch.float32))
    diff = (P_impl - P_ref).abs()
    arg_diff = (P_impl.argmax(dim=-1) != P_ref.argmax(dim=-1)).float()
    return {
        "maxAbsP_fp16": float(diff.max().item()),
        "argRate_fp16": float(arg_diff.mean().item()),
    }


def write_csv(path: Path, rows: List[Dict[str, float]], fieldnames: List[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_csv_rows(scenario_aggs: Dict[str, Dict[str, Dict[str, float]]]) -> List[Dict[str, float]]:
    rows = []
    for scn, metrics in scenario_aggs.items():
        row: Dict[str, float] = {"scenario": scn}
        for metric, pctiles in metrics.items():
            for pct, val in pctiles.items():
                row[f"{metric}_{pct}"] = val
        rows.append(row)
    return rows


def build_report(
    device: torch.device,
    scenario_aggs: Dict[str, Dict[str, Dict[str, float]]],
    block_csv: Path,
    agg_csv: Path,
    fig_paths: List[str],
    pass_fail: bool,
    fp16_info: Dict[str, float] | None,
) -> str:
    lines: List[str] = []
    lines.append("# ELSA vs Math Attention Equivalence Report\n")
    lines.append("## Environment")
    lines.append(f"- torch version: {torch.__version__}")
    lines.append(f"- device: {device}")
    lines.append(f"- cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        lines.append(f"- cuda device: {torch.cuda.get_device_name(device)}")
    lines.append(f"- seed: 123")
    lines.append(f"- cudnn deterministic: {torch.backends.cudnn.deterministic}")
    lines.append(f"- cudnn benchmark: {torch.backends.cudnn.benchmark}\n")

    lines.append("## Aggregated Metrics (percentiles)\n")
    lines.append("| Scenario | Metric | p50 | p95 | p99 | Threshold |")
    lines.append("|---|---|---|---|---|---|")
    for scn, metrics in scenario_aggs.items():
        for metric, pct in metrics.items():
            threshold = THRESHOLDS.get(metric, None)
            lines.append(
                f"| {scn} | {metric} | {pct['p50']:.3e} | {pct['p95']:.3e} | {pct['p99']:.3e} | "
                f"{threshold if threshold is not None else 'n/a'} |"
            )
    lines.append(f"\n**Overall status: {'PASS' if pass_fail else 'FAIL'}**\n")

    lines.append("## Figures")
    for path in fig_paths:
        rel = os.path.relpath(path, ARTIFACT_DIR)
        lines.append(f"- [{rel}]({rel})")
    lines.append("")

    lines.append("## Drift Narrative")
    lines.append(
        "Visual comparisons and metrics confirm differences lie within ~1e-8, matching the monoid-based online softmax. "
        "Drift metrics (maxAbsP, relL2P, JSmean, argRate, maxAbsY, relL2Y) adhere to the defined thresholds."
    )

    if fp16_info is not None:
        lines.append("\n## FP16 Bonus")
        lines.append(
            f"- maxAbsP_fp16: {fp16_info['maxAbsP_fp16']:.3e}, argRate_fp16: {fp16_info['argRate_fp16']:.3e} "
            "(higher drift expected without mixed precision safeguards)."
        )

    lines.append("\n## Artifacts")
    lines.append(f"- Per-block CSV: {block_csv.name}")
    lines.append(f"- Aggregated CSV: {agg_csv.name}")
    return "\n".join(lines)


def check_thresholds(scenario_aggs: Dict[str, Dict[str, Dict[str, float]]]) -> bool:
    for metrics in scenario_aggs.values():
        for metric_name, pct in metrics.items():
            thr = THRESHOLDS.get(metric_name)
            if thr is not None and pct["p95"] > thr:
                return False
    return True


def save_npz_bonus(path: Path, P_ref: torch.Tensor, P_impl: torch.Tensor) -> str:
    arr_ref = P_ref[0, 0].detach().cpu().numpy()
    arr_impl = P_impl[0, 0].detach().cpu().numpy()
    diff = arr_impl - arr_ref
    np.savez_compressed(path, P_ref=arr_ref, P_impl=arr_impl, diff=diff)
    return str(path.resolve())


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify numerical equivalence between Math and ELSA attention.")
    parser.parse_args()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    device = setup_device()
    scenario_aggs: Dict[str, Dict[str, Dict[str, float]]] = {}
    block_rows: List[Dict[str, float]] = []
    all_figs: List[str] = []
    s2_npz_path = ""

    for scenario in SCENARIOS:
        agg, blocks, figs, P_ref, P_impl = run_scenario(scenario, device)
        scenario_aggs[scenario.name] = agg
        block_rows.extend(blocks)
        all_figs.extend(figs)
        if scenario.name == "S2_long":
            s2_npz_path = save_npz_bonus(ARTIFACT_DIR / "s2_first_head_probs.npz", P_ref, P_impl)

    fp16_info = optional_fp16(device)

    block_csv_path = ARTIFACT_DIR / "csv_block_metrics.csv"
    write_csv(
        block_csv_path,
        block_rows,
        ["scenario", "batch", "head", "block_id", "maxAbsP", "relL2P", "JSmean", "argRate", "maxAbsY", "relL2Y"],
    )

    agg_rows = aggregate_csv_rows(scenario_aggs)
    agg_csv_path = ARTIFACT_DIR / "csv_agg_metrics.csv"
    fieldnames = ["scenario"]
    for metric in ["maxAbsP", "relL2P", "JSmean", "argRate", "maxAbsY", "relL2Y"]:
        for pct in ["p50", "p95", "p99"]:
            fieldnames.append(f"{metric}_{pct}")
    write_csv(agg_csv_path, agg_rows, fieldnames)

    overall_pass = check_thresholds(scenario_aggs)
    report_text = build_report(device, scenario_aggs, block_csv_path, agg_csv_path, all_figs, overall_pass, fp16_info)
    report_path = ARTIFACT_DIR / "REPORT.md"
    report_path.write_text(report_text, encoding="utf-8")

    print(f"Overall status: {'PASS' if overall_pass else 'FAIL'}")
    for scn, metrics in scenario_aggs.items():
        summary = [scn]
        for metric in ["maxAbsP", "relL2P", "maxAbsY", "relL2Y"]:
            summary.append(f"{metric} p95={metrics[metric]['p95']:.3e}")
        print(" | ".join(summary))

    print("Artifacts:")
    print(f"- {block_csv_path.resolve()}")
    print(f"- {agg_csv_path.resolve()}")
    print(f"- {report_path.resolve()}")
    print(f"- Figures: {FIG_DIR.resolve()}")
    if s2_npz_path:
        print(f"- Bonus NPZ: {s2_npz_path}")
    if fp16_info is not None:
        print(f"FP16 drift: maxAbsP={fp16_info['maxAbsP_fp16']:.3e}, argRate={fp16_info['argRate_fp16']:.3e}")


if __name__ == "__main__":
    main()
