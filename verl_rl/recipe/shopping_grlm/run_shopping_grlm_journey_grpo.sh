#!/bin/bash
set -e

# ============================================================================
# Shopping GRLM Journey GRPO Training — Standard Sampling
# ============================================================================
# Journey task: predict shopping journeys from user events / profile.
# Uses standard verl main_ppo entry point with vLLM sampling (no beam search,
# no CoT, no two-stage, no custom trainer/worker/rollout).
#
# Reward: format * (0.2*IF + 0.3*diversity + 0.5*relevance) * volume_factor
# Thinking mode: disabled (enable_thinking=False for Qwen3.5)
#
# Data (from ShoppingGenRec/s8_build_journey_rl_data.py):
#   prompt  : JSON chat messages [{"role":"user","content":"..."}]
#   answer  : ground-truth journey JSON (used for relevance reward)
#   extra columns: required_journey_count, min_products_per_journey,
#                  gt_journey_count, gt_total_products, task_type, user_id
# ============================================================================

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
REPO_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")
cd "$REPO_ROOT"

# ============================================================================
# Cluster Configuration
# ============================================================================
export N_NODES=${N_NODES:-1}
export N_GPUS=${N_GPUS:-2}

echo "Using configuration: N_NODES=$N_NODES, N_GPUS=$N_GPUS"

# ============================================================================
# Model & Data Configuration
# ============================================================================
export BASE_MODEL=${BASE_MODEL:-"/scratch/AzureBlobStorage_CODE/scratch/workspaceblobstore/users/wangying/LlamaFactory/saves/journeyv4_step1_le4096_ckpt4768/lora_journey_v4_step2_v1sample/sft_4gpus_lr2e-5_batch8_gradacc2_lorarank64_cut32768_enableligerkernel_true_neatpacking_false_flashattn_fa2_enablethinkingfalse_epoch3.0/checkpoint-1425-merged"}
export DATA_DIR=${DATA_DIR:-"/data/xiaoyukou/GRLM/ShoppingGenRec/rl_data/journey"}
export TRAIN_FILES=${TRAIN_FILES:-"$DATA_DIR/train.parquet"}
export VAL_FILES=${VAL_FILES:-"$DATA_DIR/test.parquet"}

export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}
export VLLM_ATTENTION_BACKEND=XFORMERS

# ============================================================================
# Training Hyperparameters
# ============================================================================
export LEARNING_RATE=${LEARNING_RATE:-1e-6}
export KL_LOSS_COEF=${KL_LOSS_COEF:-0.005}
export TEMPERATURE=${TEMPERATURE:-0.9}

# ============================================================================
# Batch Size Configuration
# ============================================================================
export USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}
export MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-65536}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}

# ============================================================================
# Rollout Configuration — Standard Sampling (no beam search)
# ============================================================================
# Number of samples per prompt for GRPO advantage computation
export ROLLOUT_N=${ROLLOUT_N:-8}

# Journey JSON can be very long — 8192 tokens max
export RESPONSE_LENGTH=${RESPONSE_LENGTH:-8192}

# ============================================================================
# Output Configuration
# ============================================================================
export PROJECT_NAME=${PROJECT_NAME:-"Shopping_GRLM_Journey_RL"}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-"grpo_journey_n8_rel05"}
export OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/shopping_grlm_journey"}
export WANDB_MODE=${WANDB_MODE:-offline}

# ============================================================================
# Checkpoint & Logging Configuration
# ============================================================================
export SAVE_FREQ=${SAVE_FREQ:-50}
export MAX_CKPT_TO_KEEP=${MAX_CKPT_TO_KEEP:-3}
export TEST_FREQ=${TEST_FREQ:-80}
export LOGGER_BACKEND=${LOGGER_BACKEND:-"[console,file]"}

# ============================================================================
# Network Configuration
# ============================================================================
export TCP_NIC=$(ifconfig 2>/dev/null | grep -B1 " "$(hostname -i 2>/dev/null)" " | grep -o "^\w*" || echo "eth0")
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}

# ============================================================================
# Print Configuration
# ============================================================================
echo "==================================="
echo "Shopping GRLM Journey GRPO Training"
echo "==================================="
echo "Model: $BASE_MODEL"
echo "Data:  $TRAIN_FILES"
echo "Cluster: $N_NODES nodes x $N_GPUS GPUs"
echo "Batch Size: $TRAIN_BATCH_SIZE"
echo "GRPO n: $ROLLOUT_N"
echo "Learning Rate: $LEARNING_RATE"
echo "Temperature: $TEMPERATURE"
echo "Max Response Length: $RESPONSE_LENGTH"
echo "Mode: Standard vLLM sampling"
echo "Reward: format*(0.2*IF + 0.3*Div + 0.5*Rel)*vol"
echo "Save Freq: $SAVE_FREQ steps"
echo "Max Checkpoints: $MAX_CKPT_TO_KEEP"
echo "Output Dir: $OUTPUT_DIR"
echo "==================================="

# ============================================================================
# Run — uses verl's standard main_ppo entry point
# ============================================================================
mkdir -p logs

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.max_prompt_length=20000 \
    data.prompt_key='prompt' \
    data.shuffle=True \
    data.max_response_length=$RESPONSE_LENGTH \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path=$SCRIPT_DIR/shopping_grlm_journey_recipe.py \
    data.custom_cls.name=ShoppingGrlmJourneyDataset \
    data.reward_fn_key='data_source' \
    ++data.data_source_key='data_source' \
    custom_reward_function.path=$SCRIPT_DIR/shopping_grlm_journey_recipe.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_TOKENS_PER_GPU \
    actor_rollout_ref.rollout.max_num_seqs=2048 \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$N_NODES \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$OUTPUT_DIR/ckpts \
    trainer.total_epochs=2 \
    trainer.val_before_train=True \
    trainer.logger=$LOGGER_BACKEND \
    ++trainer.max_actor_ckpt_to_keep=$MAX_CKPT_TO_KEEP \
    ++trainer.log_val_generations=10 \
    ++trainer.validation_data_dir=$OUTPUT_DIR/val_generations \
    ++trainer.rollout_data_dir=$OUTPUT_DIR/rollout_generations \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    ++critic.enable=False \
    ++actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    ++actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    $@
