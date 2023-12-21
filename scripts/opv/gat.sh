#!/bin/bash

[ ! -d "exp_results/opv" ] && mkdir -p exp_results/opv

# molecular: 0-gap, 1-homo, 2-lumo, 3-spectral_overlap
# polymer: 4-homo, 5-lumo, 6-gap, 7-optical_lumo
TASK=$1

python -u train_opv.py \
          --method gat \
          --data_dir "pyg_data/opv/GNN_2d" \
          --runs 3 \
          --target $TASK \
          --lr 0.001 \
          --wd 0 \
          --dropout 0.0 \
          --batch_size 64 \
          --epochs 200 \
          > exp_results/opv/gat_task$TASK.log 2>&1
