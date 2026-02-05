CUDA_VISIBLE_DEVICES=0,1 python s0_init_emb.py

CUDA_VISIBLE_DEVICES=0,1 nohup python -u s1_init_sum.py > ../logs/s1_init_sum.out 2>&1 &

python s2_build_id2meta.py

python s3_build_meta2tid_sft_data.py

python s4_build_rec_sft_data.py

python s4_build_collaborative_sft_data.py

CUDA_VISIBLE_DEVICES=0,1 nohup bash examples/GRLM/grlm_indomain_beauty.sh > ./logs/grlm_indomain_beauty_sft.out 2>&1 &

CUDA_VISIBLE_DEVICES=0,1 nohup python -u s5_beauty_eval.py > ../logs/s5_beauty_eval.out 2>&1 &

bash deploy_env.sh

CUDA_VISIBLE_DEVICES=0,1 nohup bash ./recipe/grlm/run_grlm_grpo.sh > ../logs/run_grlm_grpo.out 2>&1 &
