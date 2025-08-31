export VQA_MODEL="./cache/huggingface/hub/Qwen2-VL-7B-Instruct-GPTQ-Int4"
export API_KEY="token-abc123"
export BASE_URL="http://localhost:8000/v1"
python3 module/evaluate/ScienceQA_evaluate_acc.py --data_file ./dataspace/ScienceQA_DATA/problems.json \
 --result_file ./logs/InternVL2-8B-AWQ_New_ScienceQA_MiniLM-L6-v2_wpreFLMR_wCoT_whybridRAG_wMCTS_wmutual0208_greedy5_repeat5_evaluate_roll8_Top20_Top3_wextra_body//outputs/final_preds.csv
