model_path=../hf_qwen3_2507_4b
output_dir=../sft_ckpts/grlm_crossdomain_electronic_phone

deepspeed --num_gpus 8 \
src/train.py \
--deepspeed examples/deepspeed/ds_z2_config.json \
--stage sft \
--model_name_or_path $model_path \
--do_train \
--dataset grlm_crossdomain_electronic_phone \
--template qwen3 \
--finetuning_type full \
--output_dir  $output_dir \
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