#! /bin/sh

cd ../.. || exit

SIZE="1000 2000 3000 4000 5000 6000 7000"

for i in $SIZE
do
echo -e "\n\ntrain data size = $i\n\n"
python -u train_opv_small.py \
          --method mhnn \
          --data_dir "../data/emhdm_opv_conj" \
          --dsize $i \
          --target 0 \
          --All_num_layers 3 \
          --MLP_num_layers 2 \
          --MLP2_num_layers 2 \
          --MLP3_num_layers 2 \
          --Classifier_num_layers 2 \
          --MLP_hidden 256 \
          --Classifier_hidden 128 \
          --aggregate mean \
          --restart_alpha 0.5 \
          --lr 0.0001 \
          --wd 0.001 \
          --dropout 0.1 \
          --batch_size 16 \
          --epochs 500
done

cd exp/opv || exit
