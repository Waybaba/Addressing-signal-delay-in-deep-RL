#!/bin/bash
# SLURM training script for SAC-RNN experiment - Grid search over environments, delays, and seeds
# Parameters (loop order: env.name -> env.delay -> seed):
#   - env.name: HalfCheetah-v5, Ant-v5, Walker2d-v5, Hopper-v5 (4 values)
#   - env.delay: 4, 8, 16 (3 values)
#   - seed: 0, 1, 2 (3 values)
# Total combinations: 4 * 3 * 3 = 36

#SBATCH --job-name=grid-search
#SBATCH --array=0-35%40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=output/slurm_logs/%x/%A/%a.out
#SBATCH --error=output/slurm_logs/%x/%A/%a.err

### Set environment variables
export DISPLAY=:1
export PYTHONUNBUFFERED=1

### VARIABLES
WANDB_PROJECT="signal-delay-rl"
WANDB_GROUP="sac_rnn_grid_search_v1"
CONDA_ENV="address"
PRJ_DIR="${HOME}/code/tmp/Addressing-signal-delay-in-deep-RL"

### Parameter arrays (order: env.name -> env.delay -> seed)
ENV_NAMES=("HalfCheetah-v5" "Ant-v5" "Walker2d-v5" "Hopper-v5")
# DELAYS=(0 4 8 12 16 20)  # old values
DELAYS=(4 8 16)
# SEEDS=(0 1 2 3 4)  # old values
SEEDS=(0 1 2)

# Calculate indices from task ID
# Total combinations: 4 * 3 * 3 = 36
# Loop order: env.name (outermost) -> env.delay -> seed (innermost)
# env_name_idx = task_id / (3 * 3) = task_id / 9
# delay_idx = (task_id / 3) % 3
# seed_idx = task_id % 3

TASK_ID=$SLURM_ARRAY_TASK_ID
ENV_NAME_IDX=$((TASK_ID / 9))
DELAY_IDX=$(((TASK_ID / 3) % 3))
SEED_IDX=$((TASK_ID % 3))

# Get parameter values
ENV_NAME=${ENV_NAMES[$ENV_NAME_IDX]}
DELAY=${DELAYS[$DELAY_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# Generate run name
RUN_NAME="${ENV_NAME}_delay${DELAY}_seed${SEED}"

### RUN
cd ${PRJ_DIR} || exit 1
source ~/.bashrc
conda activate ${CONDA_ENV}

echo "Starting SAC-RNN experiment - Grid search"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Parameters:"
echo "  - experiment: sac_rnn (fixed)"
echo "  - env.name: $ENV_NAME (idx: $ENV_NAME_IDX)"
echo "  - env.delay: $DELAY (idx: $DELAY_IDX)"
echo "  - seed: $SEED (idx: $SEED_IDX)"
echo "  - run_name: $RUN_NAME"
echo ""

python src/entry.py \
  experiment=cat_mlp \
  env.name="${ENV_NAME}" \
  +env.d_touch_ratio=0.5 \
  +env.touch_rbase=1 \
  +env.goal_rratio=10 \
  env.delay="${DELAY}" \
  env.save_minari=false \
  start_timesteps=10000 \
  trainer.episode_per_test=10 \
  trainer.max_epoch=1000 \
  trainer.step_per_epoch=5000 \
  trainer.batch_size=32 \
  trainer.batch_seq_len=64 \
  global_cfg.critic_input.obs_type=oracle \
  global_cfg.critic_input.history_merge_method=none \
  global_cfg.actor_input.obs_type=normal \
  global_cfg.actor_input.history_merge_method=cat_mlp \
  seed="${SEED}" \
  wandb.mode=online \
  wandb.buf_dir=false \
  wandb.group="${WANDB_GROUP}"
