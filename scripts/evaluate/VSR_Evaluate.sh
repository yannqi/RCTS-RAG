export API_KEY="token-abc123"
export BASE_URL="http://localhost:8000/v1"
python3 module/evaluate/VSR_MC_evaluate_acc.py --data_file ./dataspace/visual-spatial-reasoning/splits/random/test_options.jsonl \
 --result_file ./logs/Qwen2-VL-2B-Instruct_VSR_MC_random_woRAG_wextra_body_wosystemPrompt/outputs/final_preds.csv
