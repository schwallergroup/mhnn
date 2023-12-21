#! /bin/bash

[ ! -d "exp_results/pcqm4mv2" ] && mkdir -p exp_results/pcqm4mv2

python -u train_pcqm4mv2.py \
    --method mhnn \
    --data_dir "pyg_data/pcqm4mv2" \
    --All_num_layers 3 \
    --MLP1_num_layers 2 \
    --MLP2_num_layers 2 \
    --MLP3_num_layers 2 \
    --MLP4_num_layers 2 \
    --output_num_layers 3 \
    --MLP_hidden 512 \
    --output_hidden 256 \
    --aggregate mean \
    --lr 0.0001 \
    --wd 0 \
    --dropout 0.05 \
    --batch_size 256 \
    --epochs 400 \
    > exp_results/pcqm4mv2/mhnn.log 2>&1
