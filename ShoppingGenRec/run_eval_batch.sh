#!/bin/bash
set -e
cd /scratch/workspaceblobstore/users/xiaoyukou/GRLM/ShoppingGenRec

echo "=== Waiting for PID 757839 to finish ==="
while kill -0 757839 2>/dev/null; do sleep 30; done
echo "=== PID 757839 done, starting batch evals ==="

echo "=== [1/3] checkpoint-final, min_products=15 ==="
python -u s5_2_journey_eval_split_task.py \
  --model_path /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Results/qwen3-5-9b_lora_v4/merged_checkpoint_final \
  --output_dir /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/eval_results/qwen3-5-9b_lora_v4_checkpoint-final_15/ \
  --min_products_override 15 \
  2>&1 | tee logs/s5_ckpt_final_15.out

echo "=== [2/3] checkpoint-475, min_products=15 ==="
python -u s5_2_journey_eval_split_task.py \
  --model_path /scratch/AzureBlobStorage_CODE/scratch/workspaceblobstore/users/wangying/LlamaFactory/saves/journeyv4_step1_le4096_ckpt4768/lora_journey_v4_step2_v1sample/sft_4gpus_lr2e-5_batch8_gradacc2_lorarank64_cut32768_enableligerkernel_true_neatpacking_false_flashattn_fa2_enablethinkingfalse/checkpoint-475-merged \
  --output_dir /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/eval_results/v4_ying_9B_checkpoint-475_15/ \
  --min_products_override 15 \
  2>&1 | tee logs/s5_ckpt_475_15.out

echo "=== [3/3] checkpoint-800, min_products=15 ==="
python -u s5_2_journey_eval_split_task.py \
  --model_path /scratch/AzureBlobStorage_CODE/scratch/workspaceblobstore/users/wangying/LlamaFactory/saves/journeyv4_step1_le4096_ckpt4768/lora_journey_v4_step2_v1sample/sft_4gpus_lr2e-5_batch8_gradacc2_lorarank64_cut32768_enableligerkernel_true_neatpacking_false_flashattn_fa2_enablethinkingfalse_epoch3.0/checkpoint-800-merged \
  --output_dir /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/eval_results/v4_ying_9B_checkpoint-800_15/ \
  --min_products_override 15 \
  2>&1 | tee logs/s5_ckpt_800_15.out

echo "=== All 3 evals done! ==="
