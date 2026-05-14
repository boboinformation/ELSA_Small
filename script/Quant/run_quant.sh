#!/usr/bin/env bash
# Quant — precision comparison: specify checkpoint dtype + target inference dtype,
#         strategy (PyTorch direct / TRT FP16 / TRT INT8) is auto-determined.
#
# ---- 預設 checkpoint（各 family 已內建，不需手動設）-----------
#   elsa_swin : /raid/jess/SICNet/swinv2_small_patch4_window8_256.pth
#   elsa_vit  : /raid/jess/SICNet/deit_small_patch16_224.pth
#   sic_int   : /raid/jess/SICNet/out_sic_small_int/20250527-172928-.../model_best.pth.tar
#               300 epochs | top1=80.844% | top5=95.422%
#
# ---- 常用範例 ------------------------------------------------
#   從 script/Quant/ 目錄執行：
#
#   FP32 → FP16 (預設):
#     bash run_quant.sh                          # elsa_swin
#     RUN_FAMILY=elsa_vit bash run_quant.sh
#     RUN_FAMILY=sic_int bash run_quant.sh
#
#   FP32 → INT8:
#     RUN_INFER_DTYPE=int8 bash run_quant.sh
#     RUN_INFER_DTYPE=int8 RUN_FAMILY=elsa_vit bash run_quant.sh
#     RUN_INFER_DTYPE=int8 RUN_FAMILY=sic_int bash run_quant.sh
#
#   FP32 → FP32 (PyTorch baseline, no TRT, validate + bench):
#     RUN_INFER_DTYPE=fp32 bash run_quant.sh
#     RUN_INFER_DTYPE=fp32 RUN_FAMILY=elsa_vit bash run_quant.sh
#     RUN_INFER_DTYPE=fp32 RUN_FAMILY=sic_int bash run_quant.sh
#
#   FP32 → FP32 validate only:
#     RUN_INFER_DTYPE=fp32 RUN_TASK=validate bash run_quant.sh
#     RUN_INFER_DTYPE=fp32 RUN_TASK=validate RUN_FAMILY=elsa_vit bash run_quant.sh
#     RUN_INFER_DTYPE=fp32 RUN_TASK=validate RUN_FAMILY=sic_int bash run_quant.sh
#
#   Only benchmark (no validation, no data-dir needed):
#     RUN_TASK=bench bash run_quant.sh
#     RUN_TASK=bench RUN_FAMILY=elsa_vit bash run_quant.sh
#     RUN_TASK=bench RUN_FAMILY=sic_int bash run_quant.sh
#
#   dry run:
#     RUN_DRY_RUN=1 bash run_quant.sh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_SH="/home/pojen/miniconda3/etc/profile.d/conda.sh"

# ---- 可覆寫的預設值 ------------------------------------------
RUN_CKPT_DTYPE="${RUN_CKPT_DTYPE:-fp32}"      # fp32 | fp16  — dtype the checkpoint is stored in
RUN_INFER_DTYPE="${RUN_INFER_DTYPE:-fp16}"     # fp32 | fp16 | int8  — target inference dtype
                                                # strategy is auto-determined:
                                                #   fp32→fp32 : pytorch_fp32 (no quantization)
                                                #   fp32→fp16 : trt_fp16     (TRT quantization)
                                                #   fp16→fp16 : pytorch_fp16 (no quantization)
                                                #   fp16→fp32 : pytorch_fp32 (upcast, no quantization)
                                                #   any →int8 : trt_int8     (TRT INT8)
RUN_TASK="${RUN_TASK:-both}"                   # validate | bench | both
RUN_FAMILY="${RUN_FAMILY:-elsa_swin}"          # elsa_swin | elsa_vit | sic_int
RUN_MODEL="${RUN_MODEL:-}"
RUN_CHECKPOINT="${RUN_CHECKPOINT:-}"
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
if [[ "${RUN_FAMILY}" == "elsa_vit" ]]; then
  RUN_IMG_SIZE="${RUN_IMG_SIZE:-224}"
else
  RUN_IMG_SIZE="${RUN_IMG_SIZE:-256}"
fi
RUN_VAL_BATCH="${RUN_VAL_BATCH:-64}"
RUN_BENCH_BATCH="${RUN_BENCH_BATCH:-1}"
RUN_WORKERS="${RUN_WORKERS:-4}"
RUN_DATA_DIR="${RUN_DATA_DIR:-/home/pojen/project/ELSA/ViT/ViT-pytorch/data/ILSVRC2012}"
RUN_SPLIT="${RUN_SPLIT:-val}"
RUN_CALIB_BATCHES="${RUN_CALIB_BATCHES:-8}"
RUN_WARMUP="${RUN_WARMUP:-50}"
RUN_BENCH_ITER="${RUN_BENCH_ITER:-200}"
RUN_ENGINE_DIR="${RUN_ENGINE_DIR:-${ROOT}/script/Engine_cache}"
RUN_DRY_RUN="${RUN_DRY_RUN:-0}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${ROOT}/logs/Quant}"
_stamp="$(date +%Y%m%d_%H%M%S)"
_run_dir="${_stamp}_${RUN_FAMILY}_${RUN_CKPT_DTYPE}to${RUN_INFER_DTYPE}"
RUN_OUT="${RUN_OUT:-${ROOT}/results/Quant/${_run_dir}/quant_${RUN_FAMILY}_${RUN_CKPT_DTYPE}to${RUN_INFER_DTYPE}.csv}"

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

# ---- checkpoint 檢查 ----------------------------------------
if [[ -n "${RUN_CHECKPOINT}" && ! -f "${RUN_CHECKPOINT}" ]]; then
  echo "[ERR] checkpoint not found: ${RUN_CHECKPOINT}" >&2
  exit 1
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
  model_kwargs_args=(--model-kwargs "elsa_backend=pytorch")
fi

# ---- 組合指令 -----------------------------------------------
cmd=(
  python "${ROOT}/models/quant_validate.py"
  --model          "${RUN_MODEL}"
  --checkpoint     "${RUN_CHECKPOINT}"
  --ckpt-dtype     "${RUN_CKPT_DTYPE}"
  --infer-dtype    "${RUN_INFER_DTYPE}"
  --task           "${RUN_TASK}"
  --img-size       "${RUN_IMG_SIZE}"
  --batch-size     "${RUN_VAL_BATCH}"
  --bench-batch-size "${RUN_BENCH_BATCH}"
  --workers        "${RUN_WORKERS}"
  --calib-batches  "${RUN_CALIB_BATCHES}"
  --engine-dir     "${RUN_ENGINE_DIR}"
  --warmup         "${RUN_WARMUP}"
  --bench-iter     "${RUN_BENCH_ITER}"
  --results-file   "${RUN_OUT}"
  "${model_kwargs_args[@]}"
)

# validate / both 才需要 data-dir 和 split
if [[ "${RUN_TASK}" != "bench" ]]; then
  cmd+=(--data-dir "${RUN_DATA_DIR}" --split "${RUN_SPLIT}")
fi

# ---- 執行 ---------------------------------------------------
echo "[Quant] ckpt=${RUN_CKPT_DTYPE} → infer=${RUN_INFER_DTYPE} | task=${RUN_TASK} family=${RUN_FAMILY} model=${RUN_MODEL}" >&2
echo "[Quant] checkpoint=${RUN_CHECKPOINT}" >&2
echo "[Quant] engine_dir=${RUN_ENGINE_DIR}" >&2
echo "[Quant] command: ${cmd[*]}" >&2

if [[ "${RUN_DRY_RUN}" == "1" ]]; then exit 0; fi

mkdir -p "${RUN_LOG_DIR}" "${ROOT}/results/Quant/${_run_dir}" "${RUN_ENGINE_DIR}"
stamp="$(date +%Y%m%d_%H%M%S)"
log="${RUN_LOG_DIR}/${stamp}_${RUN_TASK}_${RUN_FAMILY}_${RUN_MODEL}_${RUN_CKPT_DTYPE}to${RUN_INFER_DTYPE}.log"
echo "[Quant] log=${log}" >&2

{ echo "[Quant] ckpt=${RUN_CKPT_DTYPE} → infer=${RUN_INFER_DTYPE} | task=${RUN_TASK} family=${RUN_FAMILY} model=${RUN_MODEL}"
  echo "[Quant] checkpoint=${RUN_CHECKPOINT}"
  echo "[Quant] command: ${cmd[*]}"
  "${cmd[@]}"
} 2>&1 | grep -v "^🚨" | tee "${log}"
exit "${PIPESTATUS[0]}"
