#!/bin/sh
set -ex

export PYTHONPATH=`pwd`/models

exec python3 -m pdb -c continue models/official/resnet/resnet_model_test.py "$@"
