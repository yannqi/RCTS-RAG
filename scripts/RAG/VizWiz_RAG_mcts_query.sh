export API_KEY="token-abc123"
export BASE_URL="http://localhost:8000/v1;http://localhost:8001/v1;http://localhost:8002/v1"
CUDA_VISIBLE_DEVICES=0,1 python3 main_baseline.py --config configs/main/VizWiz_mctsquery.yaml