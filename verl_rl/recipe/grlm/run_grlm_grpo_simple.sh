#!/bin/bash
set -e

# ============================================================================
# GRLM GRPO Training - Single-Stage Beam Search (No CoT)
# ============================================================================
# This script uses single-stage beam search for GRLM models.
# GRLM was not trained with CoT, so we skip the CoT sampling stage.
#
# Key differences from run_grlm_grpo.sh (two-stage):
# - Single-stage: Direct beam search on prompt (no CoT + prefix)
# - Uses rollout.name=single_stage_beam
# - No think mode, direct generation
#
# Data format (from s7_build_rl_data.py):
# - prompt: instruction + input + output (ends at itemN-2)
# - ground_truth: valid_ground_truth (itemN-1)
# - test_ground_truth: held out for s5_beauty_eval.py
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
export BASE_MODEL=${BASE_MODEL:-"/data/xiaoyukou/LLaMA-Factory/saves/grlm/indomain_beauty"}
export DATA_DIR=${DATA_DIR:-"/data/xiaoyukou/GRLM/in_domain/beauty/rl_data"}
export TRAIN_FILES=${TRAIN_FILES:-"$DATA_DIR/train.parquet"}
export VAL_FILES=${VAL_FILES:-"$DATA_DIR/test.parquet"}

export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}
export VLLM_ATTENTION_BACKEND=XFORMERS

# ============================================================================
# Training Hyperparameters
# ============================================================================
export LEARNING_RATE=${LEARNING_RATE:-2e-6}
export KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
export TEMPERATURE=${TEMPERATURE:-1}

# ============================================================================
# Batch Size Configuration
# ============================================================================
export USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}
export MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-40960}
export TRAIN_BATCH_SIZE=$((N_GPUS * N_NODES))

# ============================================================================
# Rollout Configuration - Single-Stage Beam Search
# ============================================================================
# ROLLOUT_N=1 because we use beam search to get multiple candidates
export ROLLOUT_N=${ROLLOUT_N:-1}

# Beam search width - how many candidates to generate per prompt
export BEAM_SIZE=${BEAM_SIZE:-32}

# Response length for GRLM recommendation task
# "Item text ID: [w1, w2, w3, w4, w5] Title: xxx" is about 30-50 tokens
export RESPONSE_LENGTH=${RESPONSE_LENGTH:-64}

# Number of tokens to generate in beam search (GRLM item format)
# "[w1, w2, w3, w4, w5]" is about 15-20 tokens
export BEAM_MAX_TOKENS=${BEAM_MAX_TOKENS:-64}

# ============================================================================
# Output Configuration
# ============================================================================
export PROJECT_NAME=${PROJECT_NAME:-"GRLM_RL"}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-"grpo_simple_beam"}
export OUTPUT_DIR=${OUTPUT_DIR:-"./output"}
export WANDB_MODE=${WANDB_MODE:-offline}

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
echo "GRLM GRPO Training (Single-Stage Beam Search)"
echo "==================================="
echo "Model: $BASE_MODEL"
echo "Data: $TRAIN_FILES"
echo "Cluster: $N_NODES nodes x $N_GPUS GPUs"
echo "Batch Size: $TRAIN_BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Beam Size: $BEAM_SIZE"
echo "Beam Max Tokens: $BEAM_MAX_TOKENS"
echo "Mode: Single-stage beam search (no CoT)"
echo "==================================="

# ============================================================================
# Script Execution
# ============================================================================
mkdir -p logs

python3 -u -m recipe.grlm.main_grlm_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.max_prompt_length=10240 \
    data.prompt_key='prompt' \
    data.shuffle=True \
    data.max_response_length=$RESPONSE_LENGTH \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.custom_cls.path=$SCRIPT_DIR/grlm_recipe_simple.py \
    data.custom_cls.name=GrlmDataset \
    data.reward_fn_key='source' \
    ++data.data_source_key='source' \
    ++data.enable_think=False \
    ++data.enable_nonthink=False \
    ++data.use_force_prefix=False \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.use_remove_padding=True \
    custom_reward_function.path=$SCRIPT_DIR/grlm_recipe_simple.py \
    custom_reward_function.name=compute_score \
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
    actor_rollout_ref.rollout.name=single_stage_beam \
    ++actor_rollout_ref.rollout.backend=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    ++actor_rollout_ref.rollout.max_length=$RESPONSE_LENGTH \
    ++actor_rollout_ref.rollout.beam_size=$BEAM_SIZE \
    ++actor_rollout_ref.rollout.beam_max_tokens=$BEAM_MAX_TOKENS \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.max_logprobs=320 \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=$N_NODES \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$OUTPUT_DIR/ckpts \
    trainer.total_epochs=2 \
    trainer.val_before_train=True \
    ++trainer.log_val_generations=10 \
    ++trainer.validation_data_dir=$OUTPUT_DIR/val_generations \
    ++trainer.rollout_data_dir=$OUTPUT_DIR/rollout_generations \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    ++critic.enable=False \
    ++actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    ++actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    $@
