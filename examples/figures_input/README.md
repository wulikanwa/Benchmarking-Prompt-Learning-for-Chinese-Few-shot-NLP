# Optional inputs for full figure reproduction

`plot_all_figures.py` can generate all 20 figures.

It uses `summary.csv` for Figures 4-15 by default.
For Figures 16-20, and for more faithful custom plotting, you can place optional CSV files in `outputs/figures_input/` or another sibling `figures_input/` folder next to your chosen `--out_dir`.

## Expected files

### 1. `prompt_length_ablation.csv`
Columns:
- `prompt_length`
- `score`

Example:
```csv
prompt_length,score
8,0.71
16,0.76
32,0.79
64,0.785
96,0.781
```

### 2. `lora_rank_ablation.csv`
Columns:
- `rank`
- `score`

### 3. `error_taxonomy.csv`
Columns:
- `error_type`
- `count`

### 4. `thucnews_confusion_matrix.csv`
First column should be the true label name, remaining columns should be counts for predicted labels.

Example:
```csv
label,Finance,Realty,Edu,Tech,Sports
Finance,32,4,2,3,1
Realty,5,27,3,2,1
Edu,2,3,29,4,2
Tech,2,1,5,31,3
Sports,1,1,2,4,34
```

### 5. `calibration_curve.csv`
Columns:
- `bin_confidence`
- `full_ft`
- `manual_prompt`
- `ptuning_v2`
- `lora`

### 6. `calibration_summary.csv`
Columns:
- `method`
- `ece`

This is used for the radar chart if you want radar calibration values to come from your own analysis.
