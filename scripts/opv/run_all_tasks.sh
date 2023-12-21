#! /bin/bash

[ ! -d "exp_results/opv" ] && mkdir -p exp_results/opv

# model: gcn, gin, gat, gatv2, mhnn
MODEL=$1

# Task:
# molecular: 0-gap, 1-homo, 2-lumo, 3-spectral_overlap
# polymer: 4-homo, 5-lumo, 6-gap, 7-optical_lumo

for i in {0..7}
do
echo "Target $i strat"
bash scripts/opv/$MODEL.sh $i
echo "Target $i end"
done