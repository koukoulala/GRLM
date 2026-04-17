CUDA_VISIBLE_DEVICES=0,1 nohup python -u s1_init_sum.py > ../logs/s1_init_sum.out 2>&1 &

cd EasyR1/
CUDA_VISIBLE_DEVICES=0,1 nohup bash examples/grlm_beauty_grpo.sh > ./logs/grlm_beauty_grpo.out 2>&1 &

cd verl_rl/
CUDA_VISIBLE_DEVICES=0,1 nohup bash ./recipe/grlm/run_grlm_grpo_simple.sh > ../logs/run_grlm_grpo.out 2>&1 &

cd GRLM/ShoppingGenRec/
nohup python -u preprocess_raw_data/pre_s0_combine_item_data.py > logs/pre_s0.out 2>&1 &
nohup python -u s0_init_emb.py > logs/s0_init_emb.out 2>&1 &
nohup python -u cook_data/step0_0_build_input.py > logs/step0_0_build.out 2>&1 &
nohup python -u cook_data/step0_1_merge_inputs.py > logs/step0_1_merge.out 2>&1 &
nohup python -u cook_data/step0_2_generate_journey.py --max_users=60000 > logs/step0_split.out 2>&1 &
nohup python -u cook_data/step3_eval_ranker_results.py --input_folder=/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/1225_0325/JourneyWithProfile/ > logs/step3_profile.out 2>&1 &
nohup python -u preprocess_raw_data/pre_s1_construct_shopping_profile.py > logs/pre_s1.out 2>&1 &
nohup python -u s5_journey_eval.py > logs/s5_eval.out 2>&1 &
nohup python -u cook_data/step3_eval_ranker_results.py > logs/step3.out 2>&1 &
nohup python -u preprocess_raw_data/pre_s2_construct_shopping_journey.py > logs/pre_s2.out 2>&1 &
nohup python -u s1_generate_tid.py --prompt_results_dir="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260324/processed/prompts/" > logs/s1_merge.out 2>&1 &
nohup python -u s4_merge_sft_data.py > logs/s4.out 2>&1 &

nohup bash -c 'for i in 1 3; do echo "=== Chunk $i start $(date) ==="; python -u cook_data/step0_generate_journey.py --input_file /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/1225_0325/CookData/Step1_ShoppingJourney_HisLarge_100_200K_80K_${i}.tsv; echo "=== Chunk $i done $(date) ==="; done' > logs/step0.out 2>&1 &
nohup bash -c 'for i in 8; do echo "=== Chunk $i start $(date) ==="; python -u cook_data/step1_extract_query_and_infer.py --input_file /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/1225_0325/CookData/ShoppingJourney_Input_80K_${i}_results.tsv --gpu_count 2; echo "=== Chunk $i done $(date) ==="; done' > logs/step1.out 2>&1 &
nohup bash -c 'for i in 1 2 3 4 5 6; do echo "=== Chunk $i start $(date) ==="; python -u preprocess_raw_data/pre_s1_construct_shopping_profile.py --input_file /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/CookData/ShoppingJourney_Input_80K_${i}.tsv; echo "=== Chunk $i done $(date) ==="; done' > logs/pre_s1.out 2>&1 &
nohup bash -c 'for i in 53 54 55 56 57; do echo "=== Chunk $i start $(date) ==="; python -u s1_generate_tid.py --prompts_input_file /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260324/processed/prompts/item_prompts_${i}.tsv --token_file="./resources/tokens_shopping.txt" --copilot_workers=20; echo "=== Chunk $i done $(date) ==="; done' > logs/s1_continue.out 2>&1 &
nohup bash -c 'for i in 1 10 13 7 8 9; do echo "=== Chunk $i start $(date) ==="; python -u cook_data/step2_ann_search_and_output.py --input_file /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/CookData/ShoppingJourney_Input_80K_${i}_results.tsv; echo "=== Chunk $i done $(date) ==="; done' > logs/step2.out 2>&1 &

nohup bash -c 'while kill -0 1623855 2>/dev/null; do sleep 60; done; echo "=== step2.8_9 done, starting s1 rerun $(date) ==="; python -u s1_generate_tid.py --prompts_input_file /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260324/processed/prompts/item_rerun_other_prompts.tsv' > logs/s1_rerun_other.out 2>&1 &
nohup bash -c 'while kill -0 1827277 2>/dev/null; do sleep 60; done; echo "=== s1 rerun done, starting s1 continue $(date) ==="; for i in 53 54 55 56 57; do echo "=== Chunk $i start $(date) ==="; python -u s1_generate_tid.py --prompts_input_file /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260324/processed/prompts/item_prompts_${i}.tsv --token_file="./resources/tokens_shopping.txt" --copilot_workers=20; echo "=== Chunk $i done $(date) ==="; done' > logs/s1_continue.out 2>&1 &
nohup bash -c 'while kill -0 2701785 2>/dev/null; do sleep 60; done; echo "=== starting s1 42-52 $(date) ==="; for i in 42 43 44 45 46 47 48 49 50 51 52; do echo "=== Chunk $i start $(date) ==="; python -u s1_generate_tid.py --prompts_input_file /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260324/processed/prompts/item_prompts_${i}.tsv --token_file="./resources/tokens_shopping.txt" --copilot_workers=20; echo "=== Chunk $i done $(date) ==="; done' > logs/s1_42_52.out 2>&1 &
nohup bash -c 'while kill -0 2375264 2>/dev/null; do sleep 60; done; echo "=== step2.8_1 done, starting step2.8_2 $(date) ==="; python -u cook_data/step2.8_call_LLM_ranker.py --input_file=/cosmos/local/Aether/_3/xiaoyukou/385b788d-e810-4651-ae90-2791861c9e73@@@-General-_Cosmos_Split_N@@@a54ffcf0@@@3-25-2026_12-52-55_PM/Part1/Part1_574d863a-bdc3-4cc5-b954-4dcffb8959c4 --output_dir=/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/CookData/' > logs/step2.8_2.out 2>&1 &
nohup bash -c 'while kill -0 2559462 2>/dev/null; do sleep 60; done; echo "=== step2.8_1 done, starting step2.8_3 $(date) ==="; python -u cook_data/step2.8_call_LLM_ranker.py --input_file=/cosmos/local/Aether/_3/xiaoyukou/385b788d-e810-4651-ae90-2791861c9e73@@@-General-_Cosmos_Split_N@@@a54ffcf0@@@3-25-2026_12-52-55_PM/Part2/Part2_d351c2f0-53e1-40fb-af96-e9fc0d7a5a12 --output_dir=/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/CookData/' > logs/step2.8_3.out 2>&1 &
nohup bash -c 'while kill -0 2646503 2>/dev/null; do sleep 60; done; echo "=== starting step2.8_10_1 $(date) ==="; python -u cook_data/step2.8_call_LLM_ranker.py --input_file=/cosmos/local/Aether/_3/xiaoyukou/385b788d-e810-4651-ae90-2791861c9e73@@@-General-_Cosmos_Split_N@@@ff90562a@@@3-25-2026_12-48-14_PM/Part0/Part0_c3c9524a-387e-41f4-9c62-e634b9527f06 --output_dir=/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/CookData/ --token_file=./resources/tokens_shopping.txt --num_workers=20' > logs/step2.8_10_1.out 2>&1 &

nohup python -u cook_data/step2.8_call_LLM_ranker.py --input_file=/cosmos/local/Aether/_3/xiaoyukou/385b788d-e810-4651-ae90-2791861c9e73@@@-General-_Cosmos_Split_N@@@a54ffcf0@@@3-25-2026_12-52-55_PM/Part0/Part0_34e72ed2-f81d-4bf1-8de2-76c856bbe655 --output_dir=/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/0128_0301/CookData/  > logs/step2.8_1.out 2>&1 &

# Task 1: event2journey
nohup python -u s3_build_journey_sft_data.py --task event2journey > logs/s3_event2journey.out 2>&1 &

# Task 2: profile2journey
nohup python -u s3_build_journey_sft_data.py --task profile2journey > logs/s3_profile2journey.out 2>&1 &

nohup bash -c 'while kill -0 3260153 2>/dev/null; do sleep 60; done; echo "=== pre_s2 done $(date) ==="; echo "=== s2_6 event2journey start $(date) ==="; python -u s2_6_build_journey_sft_data.py --task event2journey; echo "=== s2_6 event2journey done $(date) ==="; echo "=== s2_6 profile2journey start $(date) ==="; python -u s2_6_build_journey_sft_data.py --task profile2journey; echo "=== s2_6 profile2journey done $(date) ==="; echo "=== s3 merge start $(date) ==="; python -u s3_merge_sft_data.py; echo "=== s3 merge done $(date) ==="' > logs/s2_6_s3_pipeline.out 2>&1 &

nohup python -u s6_compute_stats.py /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/eval_results/s2_ckpt1425_v4_copilot_shopping_homepage_sample/profile2journey_slm_output_evaluations.tsv /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/eval_results/s2_ckpt1425_v4_copilot_shopping_homepage_sample/remapped_thresh_0.80/profile2journey_slm_output_evaluations.tsv > logs/s6_remap_0.8_3.out 2>&1 &
python s6_visualize.py /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/eval_results/qwen3-5-9b_lora_v2_checkpoint_7000/llm_output_results.tsv --output /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/eval_results/qwen3-5-9b_lora_v2_checkpoint_7000/llm_output_results_report.html

nohup python -u cook_data/step2.8_call_LLM_ranker.py --input_file /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/1225_0325/CookData/Step1_200K_EnUs_UserReadableHis_HisLarge50_90K_1_Results_JWP.tsv > logs/step2.8_90K_1.out 2>&1 &
nohup python -u cook_data/step2.8_call_LLM_ranker.py --input_file /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/1225_0325/CookData/Step1_200K_EnUs_UserReadableHis_HisLarge50_90K_2_Results_JWP.tsv > logs/step2.8_90K_2.out 2>&1 &

nohup bash -c 'while kill -0 2800289 2>/dev/null; do sleep 60; done; echo "=== Run 1: match_6 (no reorder) ===" && python -u s5_journey_eval.py --fuzzy_score_threshold 6.0 --reorder_tid_pos -1 --output_dir /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/eval_results/demo_ckpt/match_6/ && echo "=== Run 2: match_7 (no reorder) ===" && python -u s5_journey_eval.py --fuzzy_score_threshold 7.0 --reorder_tid_pos -1 --output_dir /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/eval_results/demo_ckpt/match_7/' > logs/s5_demo_match6_and_match7.out 2>&1 &
source ~/miniconda3/etc/profile.d/conda.sh && conda activate vocab_compress && export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 && nohup python -u resources/term_vocab_compress.py > logs/term_vocab_compress.out 2>&1 &

nohup python -u preprocess_raw_data/pre_s2_construct_shopping_journey.py \
  --task event2journey \
  --merge_tsv_only \
  --prompt_results_dir /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/1225_0325/CookData_merged/ \
  --output_dir /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260407/raw_data/ \
  --user_his_file "" \
  > logs/merge_tsv_event2journey.out 2>&1 &

nohup python -u preprocess_raw_data/pre_s2_construct_shopping_journey.py \
  --task profile2journey \
  --merge_tsv_only \
  --prompt_results_dir /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/1225_0325/JourneyWithProfile/ \
  --output_dir /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260407/raw_data/ \
  --user_his_file /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/1225_0325/Step1_All_EnUs_UserReadableHis.tsv \
  > logs/merge_tsv_profile2journey.out 2>&1 &

nohup python -u s4_merge_sft_data.py --build_test_tsv > logs/s4_build_test.out 2>&1 &
