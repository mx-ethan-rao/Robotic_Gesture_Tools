#!/bin/bash
python main.py --action=predict --dataset=${1} --split=${2} --num_epochs=${3} \
               --num_layers_PG=11 \
               --num_layers_R=10 \
               --num_R=3 \
               --features_dim=768

python eval.py --dataset=${1} --split=${2}
