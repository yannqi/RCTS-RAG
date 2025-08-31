export API_KEY="token-abc123"
export BASE_URL="http://localhost:8000/v1;http://localhost:8001/v1"
python3 module/evaluate/MathV_evaluate.py --data_file ./dataspace/MathVision/data/MathVision_MINI.tsv \
 --result_file ./logs/Math_Ablation/Qwen2-VL-7B-Instruct-GPTQ-Int4_Ablation_new_MathV_wpreFLMR_whybridRAG_wCoT_wMCTS_wonlymutual_greedy5_repeat5_evaluate_roll8_Top20_Top3_wextra_body_2048_testmini_MMMU_Prompt/outputs/final_preds.csv
