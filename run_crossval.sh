#!/bin/bash

# these are python commands to run grouped cross-validation.
# to execute: conda activate lrh1models , then bash run_crossval.sh 

# use the same seeds for all cross-validation runs 
seedarr=("11006" "19014" "19980" "24137" "24205" "30770" "31084" "42" "5802" "691" "9149") 

# MLP, "all" 
for seed in "${seedarr[@]}"; do
    python train_crossval_save.py --training_mode crossval --seed $seed > out_crossval_repeats_grouped/seed_${seed}.txt
    python train_crossval_save.py --training_mode crossval --seed $seed --jumbled > out_crossval_repeats_grouped/seed_${seed}_jumbled.txt 

done

##############################

# benchmark idx, other models
for seed in "${seedarr[@]}"; do
    python train_crossval_save.py --training_mode crossval --method idx --seed $seed > out_crossval_repeats_grouped/seed_${seed}_idx.txt
    python train_crossval_save.py --training_mode crossval --method logreg --seed $seed > out_crossval_repeats_grouped/seed_${seed}_logreg.txt
    python train_crossval_save.py --training_mode crossval --method xgboost --seed $seed > out_crossval_repeats_grouped/seed_${seed}_xgboost.txt
    python train_crossval_save.py --training_mode crossval --method lightgbm --seed $seed > out_crossval_repeats_grouped/seed_${seed}_lightgbm.txt
    python train_crossval_save.py --training_mode crossval --method rf --seed $seed > out_crossval_repeats_grouped/seed_${seed}_rf.txt     
done

##############################

# data ablation studies
for seed in "${seedarr[@]}"; do
    python train_crossval_save.py --training_mode crossval --seed $seed --ablation base > out_crossval_repeats_grouped/seed_${seed}_base.txt
    python train_crossval_save.py --training_mode crossval --seed $seed --ablation base_rl > out_crossval_repeats_grouped/seed_${seed}_base_rl.txt
    python train_crossval_save.py --training_mode crossval --seed $seed --ablation base_db > out_crossval_repeats_grouped/seed_${seed}_base_db.txt
    python train_crossval_save.py --training_mode crossval --seed $seed --ablation no_full > out_crossval_repeats_grouped/seed_${seed}_no_full.txt  # exclude PL-bound
    python train_crossval_save.py --training_mode crossval --seed $seed --ablation no_4pld > out_crossval_repeats_grouped/seed_${seed}_no_4pld.txt  # exclude apo 
    python train_crossval_save.py --training_mode crossval --seed $seed --ablation no_7tt8 > out_crossval_repeats_grouped/seed_${seed}_no_7tt8.txt  # exclude 7tt8

    ## for construct ablation with just targeted poses 
    python train_crossval_save.py --training_mode crossval --seed $seed --ablation base_rl_no_4pld > out_crossval_repeats_grouped/seed_${seed}_base_rl_no_4pld.txt
    python train_crossval_save.py --training_mode crossval --seed $seed --ablation base_rl_no_7tt8 > out_crossval_repeats_grouped/seed_${seed}_base_rl_no_7tt8.txt
    python train_crossval_save.py --training_mode crossval --seed $seed --ablation base_rl_no_full > out_crossval_repeats_grouped/seed_${seed}_base_rl_no_full.txt  

done
