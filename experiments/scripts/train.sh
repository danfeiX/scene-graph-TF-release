#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

NET=$1
INFERENCE_ITER=$2
EXP_DIR=$3
GPU_ID=$4

CFG_FILE=experiments/cfgs/sparse_graph.yml
PRETRAINED=data/pretrained/coco_vgg16_faster_rcnn_final.npy

# dataset
ROIDB=VG-SGG
RPNDB=proposals.h5
IMDB=imdb_1024.h5
ITERS=150000

# log
OUTPUT=checkpoints/$EXP_DIR
TF_LOG=checkpoints/$EXP_DIR/tf_logs
rm -rf ${OUTPUT}/logs/
rm -rf ${TF_LOG}
mkdir -p ${OUTPUT}/logs
LOG="$OUTPUT/logs/`date +'%Y-%m-%d_%H-%M-%S'`"

export CUDA_VISIBLE_DEVICES=$GPU_ID

exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu 0 \
  --weights ${PRETRAINED} \
  --imdb ${IMDB} \
  --roidb ${ROIDB} \
  --rpndb ${RPNDB} \
  --iters ${ITERS} \
  --cfg ${CFG_FILE} \
  --network ${NET} \
  --inference_iter ${INFERENCE_ITER} \
  --output ${OUTPUT} \
  --tf_log ${TF_LOG}
