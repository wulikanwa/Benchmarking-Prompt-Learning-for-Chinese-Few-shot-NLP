# Benchmarking Prompt Learning for Chinese Few-shot NLP

A reproducible benchmark implementation for comparing prompt-based and parameter-efficient adaptation methods on Chinese few-shot NLP tasks.

This repository accompanies the report **Benchmarking Prompt Learning for Chinese Few-shot NLP** and provides a unified codebase for dataset preparation, few-shot subset construction, model training, evaluation, result aggregation, and figure generation.

---

## Overview

Few-shot adaptation has become a central problem in Chinese NLP because many downstream tasks operate under limited annotation budgets while still requiring reliable and efficient transfer from pretrained language models. This repository implements a benchmark framework for comparing four representative adaptation paradigms across five Chinese NLP task families:

### Adaptation methods

**Full Fine-Tuning (`full_ft`)**

**Manual Prompting (`manual_prompt`)**

**P-Tuning v2 (`ptuning_v2`)**

**LoRA (`lora`)**

### Task families

**Sentiment classification** (`sentiment`)

**Topic classification** (`topic`)

**Extractive question answering** (`qa`)

**Summarization** (`summarization`)

**Chinese-English translation** (`translation`)

The project is designed around a benchmark workflow rather than a single-task demo. It supports repeated-seed evaluation, few-shot subset sampling, metrics export, and end-to-end figure generation for the report.

---

## What this repository provides

This repository includes:

a unified experiment runner for all supported tasks and methods,

dataset preparation scripts that normalize raw data into a shared JSONL format,

few-shot subset construction for multiple random seeds,

training and evaluation utilities,

benchmark result aggregation into tabular summaries,

plotting utilities for **Figure 1 to Figure 20** in the report,

documentation for data preparation and repository usage.

---

## Repository structure

```text
chinese_fewshot_benchmark_github/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── .gitkeep
│   └── README.md
├── scripts/
│   ├── download_datasets.py
│   ├── prepare_fewshot.py
│   ├── run_benchmark.py
│   ├── aggregate_results.py
│   └── plot_all_figures.py
├── src/
│   └── benchmark/
│       ├── __init__.py
│       ├── prompts.py
│       ├── utils.py
│       ├── metrics.py
│       ├── data.py
│       ├── models.py
│       └── trainer.py
├── examples/
│   └── figures_input/
│       └── README.md
└── configs/
```

---

## File-by-file guide

### Top level

#### `README.md`

Project documentation, setup instructions, data preparation details, benchmark workflow, and reproduction notes.

#### `requirements.txt`

Dependency list for model training, PEFT, datasets, evaluation, and plotting.

#### `.gitignore`

Prevents accidental commits of caches, outputs, temporary files, compiled Python files, and large local artifacts.

#### `data/README.md`

Explains the expected dataset directory layout and JSONL schema for each task.

---

### `scripts/`

#### `scripts/download_datasets.py`

Downloads or prepares the datasets into a unified JSONL format.

Main responsibilities:

fetch supported public datasets when possible,

normalize raw fields to a consistent task-specific schema,

create `data/<task>/train.jsonl`, `dev.jsonl`, and `test.jsonl` files.

#### `scripts/prepare_fewshot.py`

Constructs few-shot subsets from full training data.

Main responsibilities:

per-class sampling for classification tasks,

total-count sampling for QA and generation tasks,

multiple random seeds,

output to `data/fewshot/<task>/seed_x/...`.

#### `scripts/run_benchmark.py`

Main experiment entry point.

Main responsibilities:

load task data,

build prompts and tokenize inputs,

instantiate the requested model and adaptation method,

train and evaluate,

save experiment metrics to `metrics.json`.

#### `scripts/aggregate_results.py`

Collects all `metrics.json` files under `outputs/` and merges them into a single `summary.csv`.

#### `scripts/plot_all_figures.py`

Creates the report figures.

Main responsibilities:

generate conceptual, quantitative, and diagnostic plots,

draw **Figure 1–20**,

consume either `summary.csv` alone or `summary.csv` plus additional figure-specific CSV inputs.

---

### `src/benchmark/`

#### `prompts.py`

Task-specific prompt templates and label verbalizers.

#### `utils.py`

General helpers for randomness control, JSONL IO, directory creation, and result saving.

#### `metrics.py`

Metric implementations for classification, QA, summarization, and translation.

#### `data.py`

Raw JSONL loading and tokenization into Hugging Face `Dataset` objects.

#### `models.py`

Model construction for classification, QA, and seq2seq generation, with support for `full_ft`, `manual_prompt`, `ptuning_v2`, and `lora`.

#### `trainer.py`

Training / evaluation loops and experiment logging.

---

## Installation

### Recommended environment

Python **3.10** or newer

PyTorch version compatible with your CUDA installation if using GPU

Linux/macOS recommended for large batch experiment runs, although the code can also run on Windows with a properly configured environment

### Install dependencies

```bash
pip install -r requirements.txt
```

If you are using a GPU-enabled machine, install the matching PyTorch build before installing the remaining dependencies if needed.

---

## Supported datasets and expected data format

This benchmark uses five task families. Some can be downloaded automatically from public sources or mirrors, while others may need manual preparation depending on licensing or mirror availability.

### 1. Sentiment — ChnSentiCorp

Expected files:

```text
data/sentiment/train.jsonl
data/sentiment/dev.jsonl
data/sentiment/test.jsonl
```

Expected schema:

```json
{"text": "这家酒店很干净，服务也不错。", "label": 1}
{"text": "体验很差，再也不会来了。", "label": 0}
```

### 2. Topic classification — THUCNews

Expected files:

```text
data/topic/train.jsonl
data/topic/dev.jsonl
data/topic/test.jsonl
```

Expected schema:

```json
{"text": "苹果发布新一代芯片产品", "label": 3}
```

Label mapping used in this repository:

0 财经

1 房产

2 教育

3 科技

4 军事

5 汽车

6 体育

7 游戏

8 娱乐

9 时尚

### 3. QA — CMRC2018

Expected files:

```text
data/qa/train.jsonl
data/qa/dev.jsonl
data/qa/test.jsonl
```

Expected schema:

```json
{"context": "...", "question": "...", "answer": "..."}
```

### 4. Summarization — LCSTS

Expected files:

```text
data/summarization/train.jsonl
data/summarization/dev.jsonl
data/summarization/test.jsonl
```

Expected schema:

```json
{"source": "原文内容……", "summary": "摘要内容……"}
```

### 5. Translation — Chinese to English

Expected files:

```text
data/translation/train.jsonl
data/translation/dev.jsonl
data/translation/test.jsonl
```

Expected schema:

```json
{"source": "今天天气很好。", "target": "The weather is nice today."}
```

### Automatic dataset preparation

Run:

```bash
python scripts/download_datasets.py --data_dir data
```

The script will:

try automatic download where implemented,

normalize raw data to a common JSONL format,

create task folders automatically.

### Manual preparation when needed

If automatic download is unavailable in your environment, prepare the JSONL files manually using the schemas above.

For public repository distribution, it is usually safer to upload:

the data preparation script,

the format specification,

the few-shot construction code,

rather than the raw datasets themselves, unless redistribution is explicitly permitted by the dataset license.

---

## Benchmark workflow

### Step 1 — Prepare datasets

```bash
python scripts/download_datasets.py --data_dir data
```

### Step 2 — Construct few-shot subsets

```bash
python scripts/prepare_fewshot.py --data_dir data --out_dir data/fewshot --seeds 13 21 42 87 100
```

### Step 3 — Run a benchmark experiment

Example: LoRA on 16-shot sentiment classification using `hfl/chinese-roberta-wwm-ext`

```bash
python scripts/run_benchmark.py \
  --task sentiment \
  --method lora \
  --model hfl/chinese-roberta-wwm-ext \
  --train_file data/fewshot/sentiment/seed_13/train_16.jsonl \
  --val_file data/sentiment/dev.jsonl \
  --test_file data/sentiment/test.jsonl \
  --output_dir outputs/sentiment/lora/seed_13_16 \
  --seed 13
```

### Step 4 — Aggregate experiment results

```bash
python scripts/aggregate_results.py --root outputs --out_csv outputs/summary.csv
```

### Step 5 — Generate report figures

```bash
python scripts/plot_all_figures.py --summary_csv outputs/summary.csv --out_dir outputs/figures
```

---

## Figures

The plotting script supports the report’s 20 figures.

### Typically generated from `summary.csv`

These figures are usually available directly once benchmark runs have been aggregated:

Figure 4 to Figure 15

### Figures that may require additional detailed inputs

These figures often depend on extra CSV files generated during ablation, calibration, or diagnostic analysis:

Figure 16 — prompt length ablation

Figure 17 — LoRA rank ablation

Figure 18 — error taxonomy

Figure 19 — THUCNews confusion matrix

Figure 20 — calibration curve and calibration summary

See:

```text
examples/figures_input/README.md
```

for expected CSV templates and field names.

---

## Benchmark configuration notes

### Backbones used in the report-oriented setup

The repository is organized around the backbone choices described in the report:

`hfl/chinese-roberta-wwm-ext`

`hfl/chinese-macbert-base`

`hfl/chinese-bert-wwm`

For seq2seq generation tasks, you may use a generation-capable backbone such as:

`google/mt5-small`

`google/mt5-base`

If you publish benchmark results derived from this repository, document the exact model names used for each task, especially for summarization and translation.

### Default few-shot seeds

The repository examples use:

- `13, 21, 42, 87, 100`

These seeds are chosen for repeated-run comparison and are used consistently throughout the sample workflow.

### Output artifacts

Typical outputs include:

```text
outputs/
├── sentiment/
├── topic/
├── qa/
├── summarization/
├── translation/
├── summary.csv
└── figures/
```

Each experiment directory stores a `metrics.json` file with the task, method, model, seed, and reported scores.

---

## Example quick start

If you want the shortest end-to-end route:

```bash
pip install -r requirements.txt
python scripts/download_datasets.py --data_dir data
python scripts/prepare_fewshot.py --data_dir data --out_dir data/fewshot --seeds 13 21 42 87 100
python scripts/run_benchmark.py --task sentiment --method lora --model hfl/chinese-roberta-wwm-ext --train_file data/fewshot/sentiment/seed_13/train_16.jsonl --val_file data/sentiment/dev.jsonl --test_file data/sentiment/test.jsonl --output_dir outputs/sentiment/lora/seed_13_16 --seed 13
python scripts/aggregate_results.py --root outputs --out_csv outputs/summary.csv
python scripts/plot_all_figures.py --summary_csv outputs/summary.csv --out_dir outputs/figures
```
