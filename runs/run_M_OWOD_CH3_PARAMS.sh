#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3


bash configs/eval/M_OWODB_CH3/HyperParameters.sh