#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve './cache/huggingface/hub/Qwen2-VL-7B-Instruct-GPTQ-Int4' --api-key token-abc123 --tensor-parallel-size 2 --pipeline-parallel-size 1 --limit-mm-per-prompt "image=12" --trust-remote-code --quantization gptq --gpu_memory_utilization 0.85 --max_model_len=32768 --max-seq-len-to-capture=32768 --port 8000 &

CUDA_VISIBLE_DEVICES=2,3 vllm serve './cache/huggingface/hub/Qwen2-VL-7B-Instruct-GPTQ-Int4' --api-key token-abc123 --tensor-parallel-size 2 --pipeline-parallel-size 1 --limit-mm-per-prompt "image=12" --trust-remote-code --quantization gptq --gpu_memory_utilization 0.85 --max_model_len=32768 --max-seq-len-to-capture=32768 --port 8001 &

CUDA_VISIBLE_DEVICES=4,5 vllm serve './cache/huggingface/hub/Qwen2-VL-7B-Instruct-GPTQ-Int4' --api-key token-abc123 --tensor-parallel-size 2 --pipeline-parallel-size 1 --limit-mm-per-prompt "image=12" --trust-remote-code --quantization gptq --gpu_memory_utilization 0.85 --max_model_len=32768 --max-seq-len-to-capture=32768 --port 8002 &

CUDA_VISIBLE_DEVICES=6,7 vllm serve './cache/huggingface/hub/Qwen2-VL-7B-Instruct-GPTQ-Int4' --api-key token-abc123 --tensor-parallel-size 2 --pipeline-parallel-size 1 --limit-mm-per-prompt "image=12" --trust-remote-code --quantization gptq --gpu_memory_utilization 0.85 --max_model_len=32768 --max-seq-len-to-capture=32768 --port 8003 &


