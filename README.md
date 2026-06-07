# LRH-1 binder and activity prediction

## Structure-guided compound prioritization strategy for virtual screening identifies putative binders for the nuclear receptor LRH-1 

### Overview
This repository contains code for training two separate multi-layer perceptrons, BLiP-L and ALiP-L. These models predict binder and activity likelihood, respectively, of a compound against the target protein LRH-1. The models are trained on data from a wet-lab screen (doi: 10.1021/acschembio.2c00805) and were used in prospective screen. 

### Associated Preprint
Chang-Gonzalez AC, Campbell AN, Bell EW, Blind R, Meiler J. Structure-guided compound prioritization strategy for virtual screening identifies putative binders for the nuclear receptor LRH-1. bioRxiv 2026.doi: 10.64898/2026.06.04.730240

### General uses for this repository
- Select new compounds for testing from pre-computed predictions in **`./out/list_all_vu98k.csv`**
- Execute saved models to predict any compound binding, activity to LRH-1 from a docked pose using **`predict_testset.py`**
- Employ MLP framework to train binder/activity predictors for other protein targets

### Repository Structure
```
.
├── train_crossval_save.py           # Script for training, evaluating prediction models 
├── predict_vu98k_filter.py          # Script for generating predictions from VU98k library 
├── predict_testset.py               # Script for generating predictions for any compound from input CSV
├── utils_define_data_model.py       # MLP definition 
├── blip-l.pth, alip-l.pth           # Trained models
├── environment.yml                  # Conda environment 
├── dock.xml                         # XML script to dock and score protein-ligand models in RosettaLigand
├── out/                             # Pre-computed predictions and filtered lists for VU98k compounds
├── out_crossval_repeat/             # Random seed cross-validation reports
├── additional_scripts/              # Standalone analysis scripts
└── data.tgz                         # Input data: compound libraries (S2k, VU98k), energy terms 
```

### Setup
This project is built using Python 3.10. We recommend using a conda environment to manage dependencies

```
conda env create -f environment.yml
conda activate lrh1models
```

## Usage

### Train and evaluate
Note: **`train_crossval_save.py`** reads input files from ``data.tgz`` 

Cross-validation (default):

```
python train_crossval_save.py
```

Cross-validation, set random seed, report permutation test performance:

```
python train_crossval_save.py --training_mode crossval --seed <insert_seed> --jumbled
```

Save models to **`test-{blip-l,alip-l}.pth`**

```
python train_crossval_save.py --training_mode save --seed <insert_seed>
```

### Inference
Generate predictions from generic compounds list:

```
python predict_testset.py input_dummy_eterms.csv [output_predictions.csv]
```

Generate predictions from Xtal and Mays compound datasets. Script reads input files from ``data.tgz``.

```
python predict_testset_rjw100_series.py
```

#### Steps to generate LRH-1 priority lists. The scripts in this section contain hard-coded file paths to compound structure and docked pose PDBs for computations. Scripts also call files in ``data.tgz``.

```
python predict_vu98k_filter.py

python 1_filter_by_location.py \
       --input_file ./out/list_filter.csv \
       --output_file out/1_list_filter_by_location.csv \
       --distance_cutoff 7.0

python 1.5_mark_confidence.py \
       --input_file ./out/1_list_filter_by_location.csv \
       --output_file ./out/1.5_list_filter_by_location_mark_confidence.csv

python 2_get_shortlist.py \
       out/1.5_list_filter_by_location_mark_confidence.csv \
       lid_root2 50 temp_shortlist_le2_r50.csv \
       --mode fatty_acid_mimetic_search
```

Files in ./out/ folder:

- list_all_vu98k.csv: Energy terms and binding and activity predictions for VU98k compounds
- list_filter.csv: Compound list from VU98k library following scoring, prediction, and chemical filters
- 1.5_list_filter_by_location_mark_confidence.csv: Compound list following structure-based filtering with confidence assignments
- vu98k_selected_for_testing.csv: List of 95 compounds tested in the lab

Files in ./additional_scripts/ :

- bcl_for_mays/: BCL scripts for generating models of compounds in Mays paper (doi: 10.1021/acs.jmedchem.9b00753)  
- *crossval* : Scripts for running cross-validation tests and plotting metrics
- analyze_vu98k_preds.py : enrichment analyses
- calc_tanimoto_paired.py : calculate pairwise tanimoto coefficient
- template_*_contacts* : protein-ligand interaction

### Citation
If you use this code, the models, datasets, or predictions in your research, please cite the corresponding preprint https://doi.org/10.64898/2026.06.04.730240. 