#! /bin/bash

[ ! -d "exp_results/opv" ] && mkdir -p exp_results/opv

# molecular: 0-gap, 1-homo, 2-lumo, 3-spectral_overlap
# polymer: 4-homo, 5-lumo, 6-gap, 7-optical_lumo
TASK=$1

python -u train_opv.py \
    --method mhnn \
    --data_dir "pyg_data/opv/mhnn" \
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
    --min_lr 0.0001 \
    --wd 0 \
    --clip_gnorm 5.0 \
    --dropout 0.0 \
    --batch_size 32 \
    --epochs 400 \
    > exp_results/opv/mhnn_task${TASK}.log 2>&1
