#!/bin/bash

# SCaTR pipeline: Qwen-1.7B, train=humaneval, test=kodcode

echo "============================================================"
echo "  SCaTR pipeline: Qwen-1.7B  |  train=humaneval  test=kodcode"
echo "  $(date)"
echo "============================================================"

# --- Step 1: Parse Responses ---
echo ""
echo "--- Parsing responses: humaneval ---"
python parse_responses.py \
    --model Qwen/Qwen3-1.7B \
    --dataset humaneval

echo ""
echo "--- Parsing responses: kodcode ---"
python parse_responses.py \
    --model Qwen/Qwen3-1.7B \
    --dataset kodcode

# --- Step 2: Extract Embeddings ---
echo ""
echo "--- Extracting embeddings: humaneval ---"
python extract_embeddings.py \
    --model Qwen/Qwen3-1.7B \
    --dataset humaneval \
    --layers 27 \
    --embedding-types final \
    --num-gpus 8

echo ""
echo "--- Extracting embeddings: kodcode ---"
python extract_embeddings.py \
    --model Qwen/Qwen3-1.7B \
    --dataset kodcode \
    --layers 27 \
    --embedding-types final \
    --num-gpus 8

# --- Step 3: Train SCaTR Classifiers ---
echo ""
echo "--- Running SCaTR: humaneval -> kodcode ---"
python scatr.py \
    --model qwen1_7b \
    --train humaneval --test kodcode \
    --type final --layer 27

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "============================================================"