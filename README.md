# TimeSense: Making Large Language Models Proficient in Time‑Series Analysis

TimeSense is a multimodal framework that equips LLMs with temporal understanding for time-series analysis. It balances textual reasoning with a preserved temporal sense via a Temporal Sense module and coordinate-based positional embeddings.

![Pipeline](figures/pipeline.png)

## Key Features

- Temporal Sense module: grounds textual reasoning in time-series dynamics
- Coordinate-based positional embeddings for spatial awareness
- Evaluation toolkit and metrics for diverse tasks
- Training and data generation utilities integrated with ChatTS

## Getting Started

### Prerequisites

- Python 3.9+
- PyTorch with CUDA (optional, recommended)
- DeepSpeed, Transformers, Datasets, Accelerate, TRL

### Environment setup

```bash
# Create and activate a virtual environment (example)
python -m venv .venv
source .venv/bin/activate

# Install commonly used dependencies
pip install torch deepspeed transformers datasets accelerate trl
```

### Link ChatTS-Training (required for training script)

The training script expects a `ChatTS-Training` folder under this directory.

```bash
# If ChatTS-Training is elsewhere, create a symlink here
ln -s /path/to/ChatTS-Training ChatTS-Training
```

## Training

Edit experiment settings in [train/train.sh](train/train.sh) (`MODE`, `exp_list`) and then run:

```bash
bash train/train.sh
```

Notes:
- The DeepSpeed config is referenced in `train/train.sh` via `DS_CONFIG`. Ensure it points to `train/ds_config_2_bf16_offload.json` (rename either the file or the referenced path if needed).
- `train/train.sh` copies files from an append directory into the model folder before training; adjust `exp_list` accordingly.

## Evaluation

Run evaluation and compute metrics:

```bash
# Generate predictions or evaluation outputs
python evaluation/evaluate.py --model_path /path/to/checkpoint --results_dir evaluation/results

# Compute metrics based on generated results
python evaluation/calculate_metric.py --results_dir evaluation/results
```

## Data Generation

Scripts in [data_generator](data_generator) synthesize time-series and metadata. Adjust parameters in the scripts and run:

```bash
# New-branch dataset variants
python data_generator/gen_new_branch.py
```

## Figures

The pipeline figure for documentation and papers is in [figures/pipeline.png](figures/pipeline.png).

## Notes & Troubleshooting

- If evaluation expects scripts not present (e.g., `generate.sh`), use [evaluation/evaluate.py](evaluation/evaluate.py) and [evaluation/calculate_metric.py](evaluation/calculate_metric.py) directly.
- Ensure base models and checkpoints are placed under [models](models) following your setup.

## Evaluation Data: 
The evaluation datasets and benchmark annotations will be made publicly available on **Hugging Face** upon paper acceptance.

## Training Framework
- **ChatTS-Training**: Model training is conducted using **ChatTS-Training**, a customized framework **modified from LLaMA-Factory**, with extensions for multimodal time-series supervision and staged training.
