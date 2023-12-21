#! /bin/bash

[ ! -d "exp_results/ocelot" ] && mkdir -p exp_results/ocelot

# 0-vie, 1-aie, 2-vea, 3-aea, 4-hl, 5-s0s1, 6-s0t1, 7-hr
# 8-cr2, 9-cr1, 10-er, 11-ar1, 12-ar2, 13-lumo, 14-homo
TASK=$1

python -u train_ocelot.py \
    --method mhnn \
    --data_dir "pyg_data/ocelot" \
    --runs 3 \
    --target $TASK \
    --All_num_layers 3 \
    --MLP1_num_layers 2 \
    --MLP2_num_layers 2 \
    --MLP3_num_layers 2 \
    --MLP4_num_layers 2 \
    --output_num_layers 3 \
    --MLP_hidden 256 \
    --output_hidden 128 \
    --aggregate mean \
    --lr 0.0001 \
    --wd 0 \
    --dropout 0.0 \
    --batch_size 64 \
    --epochs 150 \
    > exp_results/ocelot/mhnn_task${TASK}.log 2>&1
