#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

CHECKPOINT_PATH="/home/czh5/genome/Megatron-LM/outputs"
VOCAB_FILE="/home/czh5/genome/Megatron-LM/dataset/gpt2-vocab.json"
MERGE_FILE="/home/czh5/genome/Megatron-LM/dataset/gpt2-merges.txt"
DATA_PATH="/home/czh5/genome/Megatron-LM/dataset/BookCorpusDataset_text_document"

# ┏━━━━━━━━━━━━━━━━━━━━┓
# ┃ GPT-3 Small:  125M ┃
# ┗━━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="125M"
# NLAYERS=12
# HIDDEN=768
# ATEN_HEADS=12
# GLOBAL_BATCH=128

# ┏━━━━━━━━━━━━━━━━━━━━┓
# ┃ GPT-3: 1.5B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━━┛
MODEL_SIZE="1.5B"
NLAYERS=48
HIDDEN=1536
ATEN_HEADS=24
GLOBAL_BATCH=64

MICRO_BATCH=16
SEQ_LEN=$((2*1024))

GPT_ARGS="
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 4 \
    --sequence-parallel \
    --use-flash-attn \
    --recompute-activations \
    --num-layers $NLAYERS \
    --hidden-size $HIDDEN \
    --num-attention-heads $ATEN_HEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

# torchrun pretrain_gpt.py \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=12346 pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
