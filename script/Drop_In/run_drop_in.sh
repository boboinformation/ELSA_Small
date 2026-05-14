#!/usr/bin/env bash
# Drop_In — 驗證 ELSA 作為 drop-in replacement 的準確率與推理速度（不需訓練）。
#
# ---- 預設 checkpoint（各 family 已內建）----------------------
#   elsa_swin : /raid/jess/SICNet/swinv2_small_patch4_window8_256.pth
#               (SwinV2-S pretrained，非 attention 層可直接載入)
#   elsa_vit  : /raid/jess/SICNet/deit_small_patch16_224.pth
#               (DeiT-S non-distilled pretrained，key 結構與 ELSA ViT 完全相容)
#   sic_int   : /raid/jess/SICNet/out_sic_small_int/20250527-172928-.../model_best.pth.tar
#               (完整訓練 300 epochs | top1=80.844%)
#
#   注意：elsa_swin / elsa_vit 用的是標準 attention 訓練的 weights，
#         attention 層 key 不匹配會 skip（timm partial load），
#         其餘層（patch embed、norm、MLP、head）正常載入。
#
# ---- 常用範例 ------------------------------------------------
#   ※ 從 script/Drop_In/ 目錄執行；從 ELSA_Small/ 根目錄則用
#     bash script/Drop_In/run_drop_in.sh
#
#   validate + bench elsa_swin:
#     bash run_drop_in.sh
#
#   validate + bench elsa_vit:
#     RUN_FAMILY=elsa_vit bash run_drop_in.sh
#
#   validate + bench sic_int:
#     RUN_FAMILY=sic_int bash run_drop_in.sh
#
#   只跑 bench（不需要 data-dir）:
#     RUN_TASK=bench bash run_drop_in.sh
#
#   只跑 validate:
#     RUN_TASK=validate bash run_drop_in.sh
#
#   dry run:
#     RUN_DRY_RUN=1 bash run_drop_in.sh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_SH="/home/pojen/miniconda3/etc/profile.d/conda.sh"

# ---- Drop_In 固定設定 ----------------------------------------
RUN_BACKEND="pytorch"
RUN_DTYPE="fp32"

# ---- 可覆寫的預設值 ------------------------------------------
RUN_TASK="${RUN_TASK:-both}"               # validate | bench | both
RUN_FAMILY="${RUN_FAMILY:-elsa_swin}"      # elsa_swin | elsa_vit | sic_int
RUN_MODEL="${RUN_MODEL:-}"
RUN_CHECKPOINT="${RUN_CHECKPOINT:-}"       # 不設 → pretrained=True (HF)
RUN_DEVICE="${RUN_DEVICE:-cuda:0}"
if [[ "${RUN_FAMILY}" == "elsa_vit" ]]; then
  RUN_IMG_SIZE="${RUN_IMG_SIZE:-224}"
else
  RUN_IMG_SIZE="${RUN_IMG_SIZE:-256}"
fi
RUN_VAL_BATCH="${RUN_VAL_BATCH:-64}"
RUN_BENCH_BATCH="${RUN_BENCH_BATCH:-1}"
RUN_WARMUP="${RUN_WARMUP:-5}"
RUN_TRIALS="${RUN_TRIALS:-20}"
RUN_WORKERS="${RUN_WORKERS:-4}"
RUN_DATA_DIR="${RUN_DATA_DIR:-/home/pojen/project/ELSA/ViT/ViT-pytorch/data/ILSVRC2012}"
RUN_SPLIT="${RUN_SPLIT:-val}"
RUN_DRY_RUN="${RUN_DRY_RUN:-0}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${ROOT}/logs/Drop_In}"
_stamp="$(date +%Y%m%d_%H%M%S)"
_run_dir="${_stamp}_${RUN_FAMILY}"
RUN_OUT="${RUN_OUT:-${ROOT}/results/Drop_In/${_run_dir}/drop_in_${RUN_FAMILY}.csv}"

# ---- 設定預設 model -----------------------------------------
if [[ -z "${RUN_MODEL}" ]]; then
  if [[ "${RUN_FAMILY}" == "sic_int" ]]; then
    RUN_MODEL="sic_small_patch4_256_int_win"
  elif [[ "${RUN_FAMILY}" == "elsa_vit" ]]; then
    RUN_MODEL="elsa_small_patch16_224"
  else
    RUN_MODEL="elsa_small_window8_256"
  fi
fi

# ---- 各 family 預設 checkpoint ------------------------------
if [[ -z "${RUN_CHECKPOINT}" ]]; then
  case "${RUN_FAMILY}" in
    elsa_swin)
      RUN_CHECKPOINT="/raid/jess/SICNet/swinv2_small_patch4_window8_256.pth" ;;
    elsa_vit)
      RUN_CHECKPOINT="/raid/jess/SICNet/deit_small_patch16_224.pth" ;;
    sic_int)
      RUN_CHECKPOINT="/raid/jess/SICNet/out_sic_small_int/20250527-172928-sic_small_patch4_256_int_win-256/model_best.pth.tar" ;;
  esac
fi

# ---- checkpoint 檢查 ----------------------------------------
if [[ -n "${RUN_CHECKPOINT}" && ! -f "${RUN_CHECKPOINT}" ]]; then
  echo "[ERR] checkpoint not found: ${RUN_CHECKPOINT}" >&2; exit 1
fi

# ---- conda 環境 ---------------------------------------------
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[ERR] conda init script not found: ${CONDA_SH}" >&2; exit 1
fi
set +u; source "${CONDA_SH}"; conda activate sicnet; set -u
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# ---- model-kwargs (elsa_swin 需要 elsa_backend) -------------
model_kwargs_args=()
if [[ "${RUN_FAMILY}" == "elsa_swin" ]]; then
  model_kwargs_args=(--model-kwargs "elsa_backend=${RUN_BACKEND}")
fi

# ---- checkpoint args ----------------------------------------
checkpoint_args=(--checkpoint "${RUN_CHECKPOINT}")

# ---- validate -----------------------------------------------
if [[ "${RUN_TASK}" == "validate" || "${RUN_TASK}" == "both" ]]; then
  _val_out="${ROOT}/results/Drop_In/${_run_dir}/drop_in_val_${RUN_FAMILY}_${RUN_MODEL}.csv"
  val_cmd=(
    python "${ROOT}/models/validate.py"
    --data-dir   "${RUN_DATA_DIR}"
    --split      "${RUN_SPLIT}"
    --model      "${RUN_MODEL}"
    --batch-size "${RUN_VAL_BATCH}"
    --img-size   "${RUN_IMG_SIZE}"
    --workers    "${RUN_WORKERS}"
    --device     "${RUN_DEVICE}"
    --model-dtype "float32"
    --results-file "${_val_out}"
    "${model_kwargs_args[@]}"
    "${checkpoint_args[@]}"
  )
fi

# ---- bench --------------------------------------------------
if [[ "${RUN_TASK}" == "bench" || "${RUN_TASK}" == "both" ]]; then
  bench_cmd=(
    python "${ROOT}/models/benchmark.py"
    --model          "${RUN_MODEL}"
    --bench          inference
    --amp-dtype      "${RUN_DTYPE}"
    --batch-size     "${RUN_BENCH_BATCH}"
    --img-size       "${RUN_IMG_SIZE}"
    --num-warm-iter  "${RUN_WARMUP}"
    --num-bench-iter "${RUN_TRIALS}"
    --device         "${RUN_DEVICE}"
    --results-file   "${RUN_OUT}"
    "${model_kwargs_args[@]}"
  )
fi

# ---- 執行 ---------------------------------------------------
ckpt_label="${RUN_CHECKPOINT:-pretrained(HF)}"
echo "[Drop_In] task=${RUN_TASK} family=${RUN_FAMILY} model=${RUN_MODEL} device=${RUN_DEVICE}" >&2
echo "[Drop_In] checkpoint=${ckpt_label}" >&2

if [[ "${RUN_DRY_RUN}" == "1" ]]; then
  [[ "${RUN_TASK}" != "bench" ]] && echo "[Drop_In] val command:   ${val_cmd[*]}" >&2
  [[ "${RUN_TASK}" != "validate" ]] && echo "[Drop_In] bench command: ${bench_cmd[*]}" >&2
  exit 0
fi

mkdir -p "${RUN_LOG_DIR}" "${ROOT}/results/Drop_In/${_run_dir}"
stamp="$(date +%Y%m%d_%H%M%S)"
log="${RUN_LOG_DIR}/${stamp}_${RUN_TASK}_${RUN_FAMILY}_${RUN_MODEL}.log"
echo "[Drop_In] log=${log}" >&2

{
  echo "[Drop_In] task=${RUN_TASK} family=${RUN_FAMILY} model=${RUN_MODEL}"
  echo "[Drop_In] checkpoint=${ckpt_label}"

  if [[ "${RUN_TASK}" == "validate" || "${RUN_TASK}" == "both" ]]; then
    echo "[Drop_In] === validate ==="
    echo "[Drop_In] command: ${val_cmd[*]}"
    "${val_cmd[@]}"
  fi

  if [[ "${RUN_TASK}" == "bench" || "${RUN_TASK}" == "both" ]]; then
    echo "[Drop_In] === bench ==="
    echo "[Drop_In] command: ${bench_cmd[*]}"
    "${bench_cmd[@]}"
  fi
} 2>&1 | tee "${log}"
exit "${PIPESTATUS[0]}"
