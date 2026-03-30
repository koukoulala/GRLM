CUDA_VISIBLE_DEVICES=0,1 nohup python -u s1_init_sum.py > ../logs/s1_init_sum.out 2>&1 &

cd EasyR1/
CUDA_VISIBLE_DEVICES=0,1 nohup bash examples/grlm_beauty_grpo.sh > ./logs/grlm_beauty_grpo.out 2>&1 &

cd verl_rl/
CUDA_VISIBLE_DEVICES=0,1 nohup bash ./recipe/grlm/run_grlm_grpo_simple.sh > ../logs/run_grlm_grpo.out 2>&1 &

cd GRLM/ShoppingGenRec/
nohup python -u s0_init_emb.py > logs/s0_init_emb.out 2>&1 &
nohup python -u preprocess_raw_data/pre_s1_construct_shopping_profile.py > logs/pre_s1.out 2>&1 &
nohup python -u s4_journey_eval.py > logs/s4_eval.out 2>&1 &
nohup python -u cook_data/step3_eval_ranker_results.py > logs/step3.out 2>&1 &
nohup python -u preprocess_raw_data/pre_s2_construct_shopping_journey.py > logs/pre_s2.out 2>&1 &
nohup python -u s1_generate_tid.py --prompt_results_dir="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260324/processed/prompts/" > logs/s1_merge.out 2>&1 &
nohup python -u s2_6_build_journey_sft_data.py --task=event2journey > logs/s2_6.out 2>&1 &
nohup python -u s3_merge_sft_data.py > logs/s3.out 2>&1 &

nohup bash -c 'for i in 1 2 3 4 5 6; do echo "=== Chunk $i start $(date) ==="; python -u cook_data/step0_generate_journey.py --input_file /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/CookData/ShoppingJourney_Input_80K_${i}.tsv; echo "=== Chunk $i done $(date) ==="; done' > logs/step0.out 2>&1 &
nohup bash -c 'for i in 1 2 3 4 5 6; do echo "=== Chunk $i start $(date) ==="; python -u preprocess_raw_data/pre_s1_construct_shopping_profile.py --input_file /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/CookData/ShoppingJourney_Input_80K_${i}.tsv; echo "=== Chunk $i done $(date) ==="; done' > logs/pre_s1.out 2>&1 &
nohup bash -c 'for i in 53 54 55 56 57; do echo "=== Chunk $i start $(date) ==="; python -u s1_generate_tid.py --prompts_input_file /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260324/processed/prompts/item_prompts_${i}.tsv --token_file="./resources/tokens_shopping.txt" --copilot_workers=20; echo "=== Chunk $i done $(date) ==="; done' > logs/s1_continue.out 2>&1 &
nohup bash -c 'for i in 8; do echo "=== Chunk $i start $(date) ==="; python -u cook_data