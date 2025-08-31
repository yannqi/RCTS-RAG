export API_KEY="token-abc123"
export BASE_URL="http://localhost:8000/v1;http://localhost:8001/v1;http://localhost:8002/v1;http://localhost:8003/v1;http://localhost:8004/v1;http://localhost:8005/v1"
CUDA_VISIBLE_DEVICES=0 python3 main_baseline.py --config configs/main/VSR_hybridquery.yaml