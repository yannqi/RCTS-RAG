export VQA_MODEL="./cache/huggingface/hub/Qwen2-VL-7B-Instruct-GPTQ-Int4"
export API_KEY="token-abc123"
export BASE_URL="http://localhost:8000/v1"
python3 module/evaluate/VizWiz_evaluate_acc.py --data_file ./dataspace/VizWiz/vizwiz_val_annotations.json  \
 --result_file ./logs/Qwen2-VL-7B-Instruct-GPTQ-Int4_VizWiz_wtextembedding_wpreFLMR_whybridRAG_woCoT_random3_wextra_body_wnewprompt_1024/outputs/final_preds.csv
