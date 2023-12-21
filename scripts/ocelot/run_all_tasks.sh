#! /bin/bash

[ ! -d "exp_results/ocelot" ] && mkdir -p exp_results/ocelot

# Task:
# 0-vie, 1-aie, 2-vea, 3-aea, 4-hl, 5-s0s1, 6-s0t1, 7-hr
# 8-cr2, 9-cr1, 10-er, 11-ar1, 12-ar2, 13-lumo, 14-homo

for i in {0..14}
do
echo "Target $i strat"
bash scripts/ocelot/train.sh $i
echo "Target $i end"
done
