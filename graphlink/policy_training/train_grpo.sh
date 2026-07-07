#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${BASE_MODEL:=Qwen/Qwen2.5-Coder-7B-Instruct}"
: "${TRAIN_PARQUET:=${REPO_ROOT}/outputs/policy_training/train.parquet}"
: "${VAL_PARQUET:=${REPO_ROOT}/outputs/policy_training/val.parquet}"
: "${EXAMPLES_ROOT:=${REPO_ROOT}/data/examples_lite}"
: "${SAVE_DIR:=${REPO_ROOT}/outputs/policy_training/checkpoints}"
: "${EXPERIMENT_NAME:=graphlink_policy_grpo}"
: "${N_TRAJ:=8}"
: "${TOTAL_EPOCHS:=1}"
: "${LR:=1e-6}"
: "${KL_LOSS_COEF:=0.0001}"
: "${TRAIN_BATCH_SIZE:=16}"
: "${PROMPT_LEN:=16384}"
: "${RESPONSE_LEN:=1024}"
: "${ROLLOUT_NAME:=vllm}"
: "${TENSOR_MODEL_PARALLEL_SIZE:=1}"
: "${GPU_MEMORY_UTILIZATION:=0.5}"
: "${VERL_MODULE:=verl.trainer.main_ppo}"

REWARD_PY="${SCRIPT_DIR}/schema_filtering_reward.py"
REWARD_NAME="compute_score"
mkdir -p "${SAVE_DIR}/${EXPERIMENT_NAME}"

if command -v ray >/dev/null 2>&1; then
  ray stop --force || true
fi

CMD=(
  python3 -m "${VERL_MODULE}"
  algorithm.adv_estimator=grpo
  data.train_files="${TRAIN_PARQUET}"
  data.val_files="${VAL_PARQUET}"
  data.train_batch_size="${TRAIN_BATCH_SIZE}"
  data.max_prompt_length="${PROMPT_LEN}"
  data.max_response_length="${RESPONSE_LEN}"
  data.filter_overlong_prompts=True
  data.truncation=left
  actor_rollout_ref.model.path="${BASE_MODEL}"
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.actor.optim.lr="${LR}"
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.name="${ROLLOUT_NAME}"
  actor_rollout_ref.rollout.n="${N_TRAJ}"
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_k=50
  actor_rollout_ref.rollout.top_p=0.7
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}"
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  reward_model.enable=False
  custom_reward_function.path="${REWARD_PY}"
  custom_reward_function.name="${REWARD_NAME}"
  custom_reward_function.reward_kwargs.examples_root="${EXAMPLES_ROOT}"
  trainer.val_before_train=false
  trainer.test_freq=-1
  trainer.default_hdfs_dir=null
  trainer.save_freq=10
  trainer.total_epochs="${TOTAL_EPOCHS}"
  trainer.logger="['console']"
  trainer.project_name=graphlink_policy_training
  trainer.experiment_name="${EXPERIMENT_NAME}"
  trainer.default_local_dir="${SAVE_DIR}/${EXPERIMENT_NAME}"
)

LOG_FILE="${SAVE_DIR}/${EXPERIMENT_NAME}/training.log"
echo "Running GraphLink policy GRPO training"
printf ' %q' "${CMD[@]}"; echo
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
