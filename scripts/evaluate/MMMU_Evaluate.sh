export API_KEY="token-abc123"
export BASE_URL="http://localhost:8000/v1"
python3 module/evaluate/MMMU_MC_evaluate_acc.py --data_file ./dataspace/MMMU/ \
 --result_file ./logs//InternVL2-8B-AWQ_MMMU_baseline_wpreFLMR_QwenVL2_whybridRAG_woCoT_random3_wextra_body_wnewprompt_2048_v2_Dev/outputs/final_preds.csv

