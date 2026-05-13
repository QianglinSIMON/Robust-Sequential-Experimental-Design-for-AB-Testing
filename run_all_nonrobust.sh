#!/usr/bin/env bash
# Serially run 3 scripts:
# 1) Sequential_DP_nonrobust.py
# 2) Training_Q_last_net_nonrobust_nonscal.py
# 3) Training_Q_net_iteration_nonrobust_nonscal.py

set -euo pipefail

########################################
# Basic config (edit here)
########################################
GPU_ID=${GPU_ID:-3}
SEED=${SEED:-2025}
N_EXP=${N_EXP:-20}

# objective type string used by your pipeline
TYPE_OBJ=${TYPE_OBJ:-"var"}

# sizes for dataset & backward training
B=${B:-8000}
B_SHARE=${B_SHARE:-8000}
B_ITER=${B_ITER:-8000}

# chunk size for dataset building (Step 1)
BK_BATCH_SIZE=${BK_BATCH_SIZE:-256}

# DP_seq hyperparameter used by Sequential_DP_nonrobust
NU_FACTOR=${NU_FACTOR:-0.005}

# -------------------------
# Step 2 training hparams
# -------------------------
LAST_EPOCHS=${LAST_EPOCHS:-2000}
LAST_BATCH_SIZE=${LAST_BATCH_SIZE:-2048}
LAST_PATIENCE=${LAST_PATIENCE:-150}
LAST_LR=${LAST_LR:-1e-2}
LAST_TEST_RATIO=${LAST_TEST_RATIO:-0.2}

# -------------------------
# Step 3 backward training hparams
# -------------------------
BACK_EPOCHS=${BACK_EPOCHS:-2000}
BACK_BATCH_SIZE=${BACK_BATCH_SIZE:-2048}
BACK_PATIENCE=${BACK_PATIENCE:-200}
BACK_LR=${BACK_LR:-1e-3}
BACK_WEIGHT_DECAY=${BACK_WEIGHT_DECAY:-1e-5}

# -------------------------
# output / log dirs
# -------------------------
OUT_DIR_STEP1=${OUT_DIR_STEP1:-"."}
OUT_DIR_STEP2=${OUT_DIR_STEP2:-"plots_output_${TYPE_OBJ}"}
OUT_DIR_STEP3=${OUT_DIR_STEP3:-"plots_output_${TYPE_OBJ}"}

LOG_DIR=${LOG_DIR:-"logs"}
mkdir -p "${LOG_DIR}"

# file prefixes (aligned with your examples)
DATA_PREFIX=${DATA_PREFIX:-"Qval_dataset"}
MODEL_PREFIX_LAST=${MODEL_PREFIX_LAST:-"best_model_Q"}   # Step 2 out_prefix
MODEL_PREFIX_BACK=${MODEL_PREFIX_BACK:-"best_model_Q"}   # Step 3 model_prefix

LOG_PREFIX_STEP1=${LOG_PREFIX_STEP1:-"qval_build"}
LOG_PREFIX_STEP2=${LOG_PREFIX_STEP2:-"train_Q"}
LOG_PREFIX_STEP3=${LOG_PREFIX_STEP3:-"backward_train"}

TS=$(date +"%Y%m%d_%H%M%S")

echo "========================================"
echo "Start chained run"
echo "GPU      : ${GPU_ID}"
echo "SEED     : ${SEED}"
echo "N_EXP    : ${N_EXP}  (real horizon T = N_EXP + 1)"
echo "TYPE_OBJ : ${TYPE_OBJ}"
echo "B        : ${B}"
echo "B_SHARE  : ${B_SHARE}"
echo "B_ITER   : ${B_ITER}"
echo "========================================"
echo

########################################
# Step 1: build Q-value dataset
########################################
echo "[$(date)] Step 1/3: Sequential_DP_nonrobust.py"
echo "--------------------------------------------------------"

STEP1_LOG="${LOG_DIR}/${LOG_PREFIX_STEP1}_${TYPE_OBJ}_nexp${N_EXP}_gpu${GPU_ID}_${TS}.stdout.log"

python Sequential_DP_nonrobust.py \
  --gpu "${GPU_ID}" \
  --bk_batch_size "${BK_BATCH_SIZE}" \
  --B "${B}" \
  --B_share "${B_SHARE}" \
  --n_exp "${N_EXP}" \
  --seed "${SEED}" \
  --nu_factor "${NU_FACTOR}" \
  --type_obj "${TYPE_OBJ}" \
  --out_prefix "${DATA_PREFIX}" \
  --out_dir "${OUT_DIR_STEP1}" \
  > "${STEP1_LOG}" 2>&1

echo "[$(date)] Step 1 done. Log: ${STEP1_LOG}"
echo

########################################
# Step 2: train last-day Q network
########################################
echo "[$(date)] Step 2/3: Training_Q_last_net_nonrobust_nonscal.py"
echo "--------------------------------------------------------"

STEP2_LOG="${LOG_DIR}/${LOG_PREFIX_STEP2}_${TYPE_OBJ}_nexp${N_EXP}_gpu${GPU_ID}_${TS}.stdout.log"

python Training_Q_last_net_nonrobust_nonscal.py \
  --gpu "${GPU_ID}" \
  --n_exp "${N_EXP}" \
  --type_obj "${TYPE_OBJ}" \
  --epochs "${LAST_EPOCHS}" \
  --batch_size "${LAST_BATCH_SIZE}" \
  --patience "${LAST_PATIENCE}" \
  --lr "${LAST_LR}" \
  --test_ratio "${LAST_TEST_RATIO}" \
  --out_dir "${OUT_DIR_STEP2}" \
  --out_prefix "${MODEL_PREFIX_LAST}" \
  --log_dir "${LOG_DIR}" \
  > "${STEP2_LOG}" 2>&1

echo "[$(date)] Step 2 done. Log: ${STEP2_LOG}"
echo

########################################
# Step 3: backward/iterative training
########################################
echo "[$(date)] Step 3/3: Training_Q_net_iteration_nonrobust_nonscal.py"
echo "--------------------------------------------------------"

STEP3_LOG="${LOG_DIR}/${LOG_PREFIX_STEP3}_${TYPE_OBJ}_nexp${N_EXP}_gpu${GPU_ID}_${TS}.stdout.log"

python Training_Q_net_iteration_nonrobust_nonscal.py \
  --gpu "${GPU_ID}" \
  --n_exp "${N_EXP}" \
  --obj_type "${TYPE_OBJ}" \
  --B_share "${B_SHARE}" \
  --B_iter "${B_ITER}" \
  --epochs "${BACK_EPOCHS}" \
  --batch_size "${BACK_BATCH_SIZE}" \
  --patience "${BACK_PATIENCE}" \
  --lr "${BACK_LR}" \
  --weight_decay "${BACK_WEIGHT_DECAY}" \
  --out_dir "${OUT_DIR_STEP3}" \
  --model_prefix "${MODEL_PREFIX_BACK}" \
  --data_prefix "${DATA_PREFIX}" \
  --log_dir "${LOG_DIR}" \
  --log_prefix "${LOG_PREFIX_STEP3}" \
  > "${STEP3_LOG}" 2>&1

echo "[$(date)] Step 3 done. Log: ${STEP3_LOG}"
echo

echo "========================================"
echo "All 3 steps finished at $(date)"
echo "Logs:"
echo "  1) ${STEP1_LOG}"
echo "  2) ${STEP2_LOG}"
echo "  3) ${STEP3_LOG}"
echo "========================================"