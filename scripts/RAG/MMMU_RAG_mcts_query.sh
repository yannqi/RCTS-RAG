#!/bin/bash
export API_KEY="token-abc123"
export BASE_URL="http://localhost:8000/v1"
CUDA_VISIBLE_DEVICES=0,1 python3 main_baseline.py --config configs/main/MMMU_mctsquery.yaml