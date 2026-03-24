CUDA_VISIBLE_DEVICES=0,1 nohup python -u s1_init_sum.py > ../logs/s1_init_sum.out 2>&1 &

cd EasyR1/
CUDA_VISIBLE_DEVICES=0,1 nohup bash examples/grlm_beauty_grpo.sh > ./logs/grlm_beauty_grpo.out 2>&1 &

cd verl_rl/
CUDA_VISIBLE_DEVICES=0,1 nohup bash ./recipe/grlm/run_grlm_grpo_simple.sh > ../logs/run_grlm_grpo.out 2>&1 &

cd GRLM/ShoppingGenRec/
nohup python -u s0_init_emb.py > logs/s0_init_emb.out 2>&1 &
nohup python -u preprocess_raw_data/pre_s1_construct_shopping_profile.py > logs/pre_s1.out 2>&1 &
nohup python -u s4_journey_eval.py > logs/s4_eval.out 2>&1 &
