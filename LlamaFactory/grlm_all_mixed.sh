#!/bin/bash
# GRLM All Datasets Mixed Training Script

model_path=../hf_qwen3_2507_4b
output_dir=../sft_ckpts/grlm_all_mixed


DATASETS="grlm_indomain_beauty,grlm_indomain_sports,grlm_indomain_toys,grlm_crossdomain_cloth_sport,grlm_crossdomain_electronic_phone"

# ============================================================================
# 数据采样配置 (可选)
# ============================================================================
# 方式1: 使用 interleave_probs 指定各数据集的采样概率
# 例如: 0.3,0.2,0.2,0.15,0.15 表示 beauty 30%, sports 20%, toys 20%, cloth_sport 15%, electronic_phone 15%
# 注意: 概率之和应为1.0，且顺序与 DATASETS 中的数据集顺序一致
INTERLEAVE_PROBS="0.25,0.2,0.2,0.2,0.15"

MIX_STRATEGY="interleave_over"

# ============================================================================
deepspeed --num_gpus 8 \
src/train.py \
--deepspeed examples/deepspeed/ds_z2_config.json \
--stage sft \
--model_name_or_path $model_path \
--do_train \
--dataset $DATASETS \
--mix_strategy $MIX_STRATEGY \
--interleave_probs $INTERLEAVE_PROBS \
--template qwen3 \
--finetuning_type full \
--output_dir $output_dir \
--overwrite_cache \
--save_total_limit 1 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8 \
--lr_scheduler_type cosine \
--logging_steps 10 \
--save_steps 200 \
--learning_rate 1e-4 \
--num_train_epochs 3.0 \
--plot_loss \
--bf16

