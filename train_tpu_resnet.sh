#!/bin/bash
set -ex
export PYTHONPATH="$(pwd)/models"
#export TPU_NAME="${TPU_NAME:-tpeu0}"
export TPU_NAME="${TPU_NAME:-tpewpew0}"

#data_dir="$1"
#shift 1
#model_dir="$1"
#shift 1

data_dir="gs://gpt-2-poetry/data/imagenet/out"
model_dir="gs://gpt-2-poetry/benchmark/resnet/v3-256"
#model_dir="gs://gpt-2-poetry/benchmark/resnet/v3-8"

#cd models/official/resnet/benchmark
exec python3 "$(pwd)/models/official/resnet/benchmark/resnet_benchmark.py" \
  --tpu="$TPU_NAME" \
  --mode=train \
  --data_dir="$data_dir" \
  --model_dir="$model_dir" \
  --train_batch_size=1024 \
  --train_steps=112590 \
  --iterations_per_loop=1251 \
  "$@"
