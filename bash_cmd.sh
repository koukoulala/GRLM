
cd GRLM/ShoppingGenRec/
nohup python -u cook_journey_data/step0_combine_item_data.py > logs/step0_idb.out 2>&1 &
nohup python -u cook_journey_data/step0_combine_item_data.py --skip_filter > logs/step0_pg.out 2>&1 &
nohup bash -c '
while kill -0 452262 2>/dev/null; do sleep 30; done
echo "PID 407871 finished, starting step1..."
cd /scratch/workspaceblobstore/users/xiaoyukou/GRLM/ShoppingGenRec
python -u cook_journey_data/step1_InferIndexEmbAndAnnBuild.py \
  --resume_emb_dir \
    /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260522/raw_data/MatadorEmb_Index \
    /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260525/raw_data/MatadorEmb_Index \
    /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260513/raw_data/MatadorEmb_Index
' > logs/step1_idb.out 2>&1 &
nohup bash -c 'python -u cook_journey_data/step1_InferIndexEmbAndAnnBuild.py \
  --resume_emb_dir \
    /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260522/raw_data/MatadorEmb_Index \
    /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260525/raw_data/MatadorEmb_Index \
    /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260513/raw_data/MatadorEmb_Index \
    /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260528/raw_data_IDB/MatadorEmb_Index
' > logs/step1_pg.out 2>&1 &
nohup python -u cook_journey_data/step1_init_item_emb.py > logs/step1.out 2>&1 &
nohup python -u cook_journey_data/step2_0_filter_user_events.py > logs/step2_0.out 2>&1 &
nohup python -u cook_journey_data/step2_construct_shopping_profile.py > logs/step2.out 2>&1 &
nohup python -u cook_journey_data/step3_generate_journey_query.py --output_dir /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260513/raw_data/chunks --max_users 100000 > logs/step3_100K.out 2>&1 &
nohup python -u cook_journey_data/step3_generate_journey_query.py > logs/step3.out 2>&1 &
nohup python -u cook_journey_data/step3_generate_journey_query.py --resume_checkpoint_dir "" --input_file /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260513/raw_data/chunks/UserEvents_clean_profiles_results_100K_1.tsv --output_dir /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260513/raw_data/chunks > logs/step3_chunk_1.out 2>&1 &
nohup python -u cook_journey_data/step3_generate_journey_query.py --merge_results_dir /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260513/raw_data/chunks > logs/step3_merge.out 2>&1 &
nohup python -u s1_generate_tid.py --export_prompts_only > logs/s1_split.out 2>&1 &
nohup bash -c 'for i in $(seq 14 14); do echo "=== Chunk $i start $(date) ==="; python -u s1_generate_tid.py --prompts_input_file /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260528/processed_IDB_v4/prompts/item_prompts_${i}.tsv; echo "=== Chunk $i done $(date) ==="; done' > logs/s1_14.out 2>&1 &
nohup python -u s1_generate_tid.py --prompt_results_dir="/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260516/processed/prompts/" > logs/s1_merge_20260516.out 2>&1 &
pip install faiss-gpu==1.7.2 orjson==3.10.15 onnxruntime-gpu==1.18.0
nohup python -u cook_journey_data/step4_InferIndexEmbAndAnnBuild.py > logs/step4_20260516.out 2>&1 &
nohup python -u cook_journey_data/step4_InferIndexEmbAndAnnBuild.py --resume_emb_dir /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260513/raw_data/MatadorEmb_Index > logs/step4_20260522.out 2>&1 &
nohup python -u cook_journey_data/step5_InferQueryEmbAndAnnSearch.py > logs/step5.out 2>&1 &
nohup python -u cook_journey_data/step5_InferQueryEmbAndAnnSearch.py --resume --keep_chunks --ann_in_memory > logs/step5_resume.out 2>&1 &
nohup python -u cook_journey_data/step6_call_LLM_ranker.py --debug > logs/step6_debug.out 2>&1 &
nohup python -u cook_journey_data/step6_call_LLM_ranker.py --split_n 900000 > logs/step6_900K.out 2>&1 &
nohup bash -c '
while kill -0 404024 2>/dev/null; do sleep 30; done
echo "PID 404024 finished, starting step6..."
python -u cook_journey_data/step6_call_LLM_ranker.py --merge_dir /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260528/raw_data_IDB/ranker_output_full/' > logs/step6_merge_idb.out 2>&1 &
nohup python -u cook_journey_data/step6_call_LLM_ranker.py --copilot_model gpt-5.4 --input_file /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260528/raw_data_IDB/ranker_output_full/UserEvents_clean_combined_full_journey_with_products_split_009.tsv > logs/step6_idb_009.out 2>&1 &
nohup python -u cook_journey_data/step6_call_LLM_ranker.py --merge_dir /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260516/raw_data/ranker_output_full/ > logs/step6_merge.out 2>&1 &
nohup python -u cook_journey_data/step7_stats.py --input_file /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260516/raw_data/UserEvents_clean_profiles_results_Journey_Results_combined.tsv > logs/step7_stats_after_step3.out 2>&1 &
nohup bash cook_journey_data/run_vip_case_study.sh --tag new --do_sft > logs/vip_pipeline_idb_new.out 2>&1 &
nohup python -u s2_build_meta2tid_sft_data.py > logs/s2.out 2>&1 &
nohup python -u s3_build_journey_sft_data.py --task profile2journey --ranked_journey_file /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260528/vip_case_study_IDB/ranker_output/vip_users_journey_with_products_Ranked.tsv --id2meta_file /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260528/processed_IDB/id2meta_with_norm.json --output_dir /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260528/vip_case_study_IDB/sft_data --use_embedding > logs/s3_profile2journey_idb.log 2>&1 &
nohup python -u s1_generate_tid.py --prompt_file prompts/term_generationV3.md --filter_items_file /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260528/vip_case_study_IDB_new/filter_offer_ids.txt --output_dir /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260528/vip_case_study_IDB_new/processed_v3/ --resume_from_multi_path > logs/s1_generate_v3_vip.out 2>&1 &
nohup python -u s2_0_evaluate_tid.py --id2meta_file /cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/20260528/vip_case_study_IDB_new/processed_v3/id2meta_with_norm.json > logs/eval_v3.out 2>&1 &
nohup bash cook_journey_data/run_tid_ab_test.sh --ver v4 --from s1 --until s2 --runs 3 > logs/pipeline_v4.out 2>&1 &

export LD_LIBRARY_PATH="/scratch/workspaceblobstore/users/xiaoyukou/faiss-gpu-rocm/build/faiss:/home/aiscuser/.local/lib/python3.12/site-packages/faiss:${LD_LIBRARY_PATH}"
nohup python -u s0_init_emb.py > logs/s0_init_emb.out 2>&1 &
nohup bash -c 'source /scratch/workspaceblobstore/users/xiaoyukou/faiss-gpu-rocm/env.sh && python -u s0_init_emb.py' > logs/s0_init_emb.out 2>&1 &
nohup python -u s5_journey_eval.py > logs/s5_eval.out 2>&1 &
nohup python -u cook_data/step3_eval_ranker_results.py > logs/step3.out 2>&1 &
nohup python -u preprocess_raw_data/pre_s2_construct_shopping_journey.py > logs/pre_s2.out 2>&1 &

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
nohup python -u s5_2_journey_eval_split_task.py > logs/s5_eval_full_v4_0408_index.out 2>&1 &
nohup python -u s7_assign_tid_by_similarity.py > logs/s7_assign_tid_full.out 2>&1 &

VLLM_USE_V1=0 python -u /cosmos/local/users/wangn/IDB/TermIdIndex/Code/infer_termid_v3.py --model_path=$ckpt_path --input_path=/cosmos/local/users/wangn/IDB/TermIdIndex/GoidIndex/extend_index_data_en_us_20260408_01_all.tsv  --output_path=/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/EvalData/0408_10M_TermId.tsv --tensor_parallel_size=4 --max_model_len=2048 --max_tokens=128 --batch_size 1024
VLLM_USE_V1=0 nohup python -u s5_0_infer_termid.py --model_path=/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Results/qwen3-5-9b_full_v4_step2/checkpoint-1840/ --input_path=/cosmos/local/users/wangn/IDB/TermIdIndex/GoidIndex/extend_index_data_en_us_20260408_01_all.tsv  --output_path=/cosmos/projects/Recommendations/PartnerData/Pipelines/OneRec/Data/LLMTrainingData/EvalData/full_trained_ckpt_0408_10M_TermId.tsv --tensor_parallel_size=2 --max_model_len=2048 --max_tokens=128 --batch_size 1024 > logs/s5_0_termid.out 2>&1 &

nohup python -u s5_5_ranker_eval.py --instruction_version=v1 --eval_only > logs/s5_5_eval_650.out 2>&1 &
nohup python -u s5_5_ranker_eval.py --instruction_version=v2 > logs/s5_5_eval_v2_1000.out 2>&1 &
