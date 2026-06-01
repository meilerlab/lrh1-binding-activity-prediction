#!/bin/bash
OUTDIR="out_crossval_repeats"
mkdir -p $OUTDIR

## k-fold cross validation repeats with different seeds
source $(conda info --base)/etc/profile.d/conda.sh
conda activate lrh1models

# generate random seeds
seeds=(42) # seed used for "save" model
for i in {1..9}; do
    seeds+=($RANDOM)
done

# blip-l and alip-l training with seeds, output metrics for plot
for i in "${!seeds[@]}"; do
    seed=${seeds[$i]}
    run=$((i + 1))
    echo "Running repeat $run with seed $seed..."
    python -u train_crossval_save.py --seed $seed --training_mode crossval 2>&1 | tee $OUTDIR/seed_${seed}.txt
done

echo "Done with regular feature set. Results in $OUTDIR/"

## jumble features, also k-fold cross validation repeats with different seeds
for i in "${!seeds[@]}"; do
    seed=${seeds[$i]}
    run=$((i + 1))
    echo "Running repeat $run with seed $seed and jumbled features..."
    python -u train_crossval_save.py --seed $seed --training_mode crossval --jumbled 2>&1 | tee $OUTDIR/seed_${seed}_jumbled.txt
done

echo "Done with jumbled features. Results in $OUTDIR/"


python plot_crossval_metrics.py
