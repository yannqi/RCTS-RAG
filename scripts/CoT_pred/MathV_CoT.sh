#!/bin/bash
export API_KEY="token-abc123"
export BASE_URL="http://localhost:8000/v1;http://localhost:8001/v1;http://localhost:8002/v1"
python3 paper_draw/CoT_pred.py --config configs/CoT_Pred/MathV_config.yaml