export VQA_MODEL="./cache/huggingface/hub/Qwen2-VL-7B-Instruct-GPTQ-Int4"
export API_KEY="token-abc123"
export BASE_URL="http://localhost:8000/v1"
python3 main_baseline_v2.py --config configs/main/ScienceQA_random.yaml