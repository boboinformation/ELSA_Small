#!/usr/bin/env bash
# ReTrain — PyTorch backend, fp32 precision (no Triton kernel).
#
# ---- 資料與 checkpoint 路徑 ----------------------------------
#   ImageNet  : /raid/ilsvrc  (train/ + val/)
#
#   sic_int (model_best):
#     /raid/jess/SICNet/out_sic_small_int/20250527-172928-sic_small_patch4_256_int_win-256/model_best.pth.tar
#
#   elsa_swin pretrained init (SwinV2):
#     /raid/jess/SICNet/swinv2_small_patch4_window8_256.pth   (elsa_small_*)
#     /raid/jess/SICNet/swinv2_tiny_patch4_window8_256.pth    (elsa_tiny_*)
#
# ---- 常用範例 ------------------------------------------------
#   ※ 從 script/ReTrain/ 目錄執行；從 ELSA_Small/ 根目錄則用
#     bash script/ReTrain/run_retrain.sh
#
#   bench (預設):
#     bash run_retrain.sh
#
#   validate sic_int:
#     RUN_TASK=validate RUN_FAMILY=sic_int \
#     RUN_CHECKPOINT=/raid/jess/SICNet/out_sic_small_int/20250527-172928-sic_small_patch4_256_int_win-256/model_best.pth.tar \
#     bash run_retrain.sh
#
#   validate elsa_swin (無 checkpoint，從 HF 抓預訓練):
#     RUN_TASK=validate bash run_retrain.sh
#
#   train elsa_swin from scratch:
#     RUN_CHANNELS_LAST=1 RUN_TRAIN_BATCH=64 RUN_WORKERS=8 RUN_TASK=train \
#     RUN_DEVICE=cuda:1 bash run_retrain.sh
#     # LR 預設已換算為 max_lr≈6.25e-5（官方 batch=1024 線性換算到 batch=64）
#     # 若要手動覆蓋：加 RUN_LR=<value>
#
#   train elsa_vit from scratch:
#     RUN_CHANNELS_LAST=1 RUN_TRAIN_BATCH=300 RUN_WORKERS=8 RUN_TASK=train \
#     RUN_FAMILY=elsa_vit RUN_DEVICE=cuda:0 bash run_retrain.sh
#
#   bench elsa_vit (img 224):
#     RUN_FAMILY=elsa_vit bash run_retrain.sh
#
#   train sic_int from scratch:
#     RUN_TRAIN_BATCH=100 RUN_WORKERS=8 RUN_TASK=train RUN_FAMILY=sic_int bash run_retrain.sh
#
#   dry run:
#     RUN_DRY_RUN=1 bash run_retrain.sh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_SH="/home/pojen/miniconda3/etc/profile.d/conda.sh"

# ---- ReTrain 固定設定 ----------------------------------------
RUN_BACKEND="pytorch"
RUN_DTYPE="fp32"

# ---- 可覆寫的預設值 ------------------------------------------
RUN_TASK="${RUN_TASK:-bench}"
RUN_FAMILY="${RUN_FAMILY:-elsa_swin}"      # elsa_swin | elsa_vit | sic_int
RUN_MODEL="${RUN_MODEL:-}"
RUN_CHECKPOINT="${RUN_CHECKPOINT:-}"
RUN_INITIAL_CHECKPOINT="${RUN_INITIAL_CHECKPOINT:-}"
RUN_DEVICE="${RUN_DEVICE:-cuda:0}"
if [[ "${RUN_FAMILY}" == "elsa_vit" ]]; then
  RUN_IMG_SIZE="${RUN_IMG_SIZE:-224}"
else
  RUN_IMG_SIZE="${RUN_IMG_SIZE:-256}"
fi
RUN_BATCH="${RUN_BATCH:-1}"
RUN_VAL_BATCH="${RUN_VAL_BATCH:-64}"
RUN_TRAIN_BATCH="${RUN_TRAIN_BATCH:-64}"
RUN_WARMUP="${RUN_WARMUP:-5}"
RUN_TRIALS="${RUN_TRIALS:-20}"
RUN_WORKERS="${RUN_WORKERS:-4}"
RUN_CHANNELS_LAST="${RUN_CHANNELS_LAST:-0}"
RUN_LR="${RUN_LR:-}"                          # 直接指定 max LR（設了就覆蓋 RUN_LR_BASE 的計算）
# LR_BASE 預設依 family 不同：Swin 官方 batch=1024 用 1e-3，換算到 batch=64 約 6.25e-5；
# 以 sqrt scaling 反推 lr_base = 6.25e-5 / sqrt(64/256) = 1.25e-4
if [[ "${RUN_FAMILY}" == "elsa_swin" ]]; then
  RUN_LR_BASE="${RUN_LR_BASE:-0.000125}"     # → max_lr ≈ 6.25e-5 at batch=64
else
  RUN_LR_BASE="${RUN_LR_BASE:-0.001}"        # elsa_vit / sic_int 沿用原本設定
fi
RUN_MIN_LR="${RUN_MIN_LR:-1e-5}"              # cosine decay 的下限
RUN_WEIGHT_DECAY="${RUN_WEIGHT_DECAY:-0.05}"
RUN_WARMUP_EPOCHS="${RUN_WARMUP_EPOCHS:-20}"
RUN_DATA_DIR="${RUN_DATA_DIR:-/home/pojen/project/ELSA/ViT/ViT-pytorch/data/ILSVRC2012}"
RUN_SPLIT="${RUN_SPLIT:-val}"
RUN_EPOCHS="${RUN_EPOCHS:-300}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-/raid/pojen/ELSA/ELSA_Small/output/train}"
RUN_DRY_RUN="${RUN_DRY_RUN:-0}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${ROOT}/logs/ReTrain}"
_stamp="$(date +%Y%m%d_%H%M%S)"
_run_dir="${_stamp}_${RUN_FAMILY}"
RUN_OUT="${RUN_OUT:-${ROOT}/results/ReTrain/${_run_dir}/retrain_${RUN_FAMILY}_${RUN_DTYPE}.csv}"

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

# ---- conda 環境 ---------------------------------------------
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[ERR] conda init script not found: ${CONDA_SH}" >&2; exit 1
fi
set +u; source "${CONDA_SH}"; conda activate sicnet; set -u
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# ---- 防止重複執行 (train 才鎖) --------------------------------
LOCKFILE="/tmp/retrain_${RUN_FAMILY}_${RUN_MODEL}.lock"
if [[ "${RUN_TASK}" == "train" && "${RUN_DRY_RUN}" != "1" ]]; then
  if [[ -f "${LOCKFILE}" ]]; then
    pid="$(cat "${LOCKFILE}")"
    if kill -0 "${pid}" 2>/dev/null; then
      echo "[ERR] 訓練已在執行中 (PID=${pid})，請勿重複啟動。" >&2
      echo "[ERR] 若要強制重跑，先執行：rm ${LOCKFILE}" >&2
      exit 1
    fi
  fi
  echo "$$" > "${LOCKFILE}"
  trap "rm -f '${LOCKFILE}'" EXIT
fi

# ---- checkpoint helper --------------------------------------
_checkpoint_args() {
  local var="$1"
  if [[ -n "${var}" && "${var}" != "none" ]]; then
    if [[ -f "${var}" ]]; then
      echo "--checkpoint ${var}"
    else
      echo "[WARN] checkpoint not found, skipping: ${var}" >&2
    fi
  fi
}

# ---- 組合指令 -----------------------------------------------
_model_kwargs_for_family() {
  [[ "${RUN_FAMILY}" == "elsa_swin" ]] && echo "--model-kwargs elsa_backend=${RUN_BACKEND}"
}

if [[ "${RUN_TASK}" == "bench" ]]; then
  bench_model_kwargs=()
  if [[ "${RUN_FAMILY}" == "elsa_swin" ]]; then
    bench_model_kwargs=(--model-kwargs "elsa_backend=${RUN_BACKEND}")
  fi
  cmd=(
    python "${ROOT}/models/benchmark.py"
    --model    "${RUN_MODEL}"
    --bench    inference
    --amp-dtype "${RUN_DTYPE}"
    --batch-size "${RUN_BATCH}"
    --img-size "${RUN_IMG_SIZE}"
    --num-warm-iter "${RUN_WARMUP}"
    --num-bench-iter "${RUN_TRIALS}"
    --device   "${RUN_DEVICE}"
    --results-file "${RUN_OUT}"
    "${bench_model_kwargs[@]}"
  )

elif [[ "${RUN_TASK}" == "validate" ]]; then
  checkpoint_args=()
  if [[ -n "${RUN_CHECKPOINT}" && "${RUN_CHECKPOINT}" != "none" ]]; then
    [[ -f "${RUN_CHECKPOINT}" ]] && checkpoint_args=(--checkpoint "${RUN_CHECKPOINT}") \
      || echo "[WARN] checkpoint not found, skipping: ${RUN_CHECKPOINT}" >&2
  fi
  _val_out="${ROOT}/results/ReTrain/${_run_dir}/retrain_val_${RUN_FAMILY}_${RUN_MODEL}_${RUN_DTYPE}.csv"
  model_kwargs_args=()
  [[ "${RUN_FAMILY}" == "elsa_swin" ]] && model_kwargs_args=(--model-kwargs "elsa_backend=${RUN_BACKEND}")
  cmd=(
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

elif [[ "${RUN_TASK}" == "train" ]]; then
  # 每次 train 加時間戳，獨立目錄，不互相覆蓋
  _stamp="$(date +%Y%m%d_%H%M%S)"
  _experiment="${RUN_FAMILY}_${RUN_MODEL}_retrain_${_stamp}"
  initial_ckpt_args=()
  if [[ -n "${RUN_INITIAL_CHECKPOINT}" ]]; then
    [[ -f "${RUN_INITIAL_CHECKPOINT}" ]] && initial_ckpt_args=(--initial-checkpoint "${RUN_INITIAL_CHECKPOINT}") \
      || echo "[WARN] initial checkpoint not found: ${RUN_INITIAL_CHECKPOINT}" >&2
  fi
  model_kwargs_args=()
  [[ "${RUN_FAMILY}" == "elsa_swin" ]] && model_kwargs_args=(--model-kwargs "elsa_backend=${RUN_BACKEND}")
  channels_last_args=()
  [[ "${RUN_CHANNELS_LAST}" == "1" ]] && channels_last_args=(--channels-last)
  lr_args=()
  if [[ -n "${RUN_LR}" ]]; then
    lr_args=(--lr "${RUN_LR}")
  else
    lr_args=(--lr-base "${RUN_LR_BASE}")
  fi
  cmd=(
    python "${ROOT}/models/train.py"
    --data-dir              "${RUN_DATA_DIR}"
    --train-split           train
    --val-split             "${RUN_SPLIT}"
    --model                 "${RUN_MODEL}"
    --batch-size            "${RUN_TRAIN_BATCH}"
    --validation-batch-size "${RUN_VAL_BATCH}"
    --img-size              "${RUN_IMG_SIZE}"
    --workers               "${RUN_WORKERS}"
    --device                "${RUN_DEVICE}"
    --epochs                "${RUN_EPOCHS}"
    --output                "${RUN_OUTPUT_DIR}"
    --experiment            "${_experiment}"
    --opt adamw --weight-decay "${RUN_WEIGHT_DECAY}"
    --warmup-epochs "${RUN_WARMUP_EPOCHS}" --sched cosine --min-lr "${RUN_MIN_LR}"
    --mixup 0.8 --cutmix 1.0 --smoothing 0.1
    "${lr_args[@]}"
    "${model_kwargs_args[@]}"
    "${channels_last_args[@]}"
    "${initial_ckpt_args[@]}"
  )
else
  echo "[ERR] RUN_TASK must be bench | validate | train, got: ${RUN_TASK}" >&2; exit 2
fi

# ---- 執行 ---------------------------------------------------
echo "[ReTrain] task=${RUN_TASK} family=${RUN_FAMILY} model=${RUN_MODEL} backend=${RUN_BACKEND} dtype=${RUN_DTYPE} device=${RUN_DEVICE}" >&2
echo "[ReTrain] data=${RUN_DATA_DIR}" >&2
echo "[ReTrain] command: ${cmd[*]}" >&2

if [[ "${RUN_DRY_RUN}" == "1" ]]; then exit 0; fi

mkdir -p "${RUN_LOG_DIR}" "${ROOT}/results/ReTrain/${_run_dir}"
stamp="$(date +%Y%m%d_%H%M%S)"
log="${RUN_LOG_DIR}/${stamp}_${RUN_TASK}_${RUN_FAMILY}_${RUN_MODEL}.log"
echo "[ReTrain] log=${log}" >&2

{ echo "[ReTrain] task=${RUN_TASK} family=${RUN_FAMILY} model=${RUN_MODEL} backend=${RUN_BACKEND} dtype=${RUN_DTYPE}"
  echo "[ReTrain] data=${RUN_DATA_DIR}"
  echo "[ReTrain] command: ${cmd[*]}"
  "${cmd[@]}"
} 2>&1 | grep -v "^🚨" | tee "${log}"
exit "${PIPESTATUS[0]}"
