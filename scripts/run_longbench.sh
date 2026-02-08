#!/bin/bash

# Main reproduction script for CentralKV LongBench experiments

MODELS=("llama2-7b-chat" "llama3-8b-instruct")
METHODS=("centralkv" "h2o" "snapkv")

for model in "${MODELS[@]}"; do
    for method in "${METHODS[@]}"; do
        echo "Running evaluation on $model using $method..."
        
        python benchmarks/longbench/evaluate.py \
            --model $model \
            --method $method \
            --dataset "narrativeqa,qasper,multifieldqa_en,2wikimultihopqa" \
            --budget_ratio 0.2 \
            --output_dir results/$model/$method
            
    done
done





