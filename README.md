# 📖 Meta-ResGAD: Bi-Level Meta-Learning for Topology-Agnostic Graph Anomaly Detection (Meta-ResGAD)

This repository contains the official implementation of the paper: **"Meta-ResGAD: Bi-Level Meta-Learning for Topology-Agnostic Graph Anomaly Detection"** (Inferred from project context). This project implements a bi-level optimization framework for Graph Anomaly Detection, utilizing a Scorer network to guide the generation of high-quality pseudo-anomalies.

## 📂 Project Structure

The project file structure is organized as follows:

```text
../
├── dataset/                # Dataset files (.mat format)
│   ├── reddit.mat
│   ├── elliptic.mat
│   └── ...
├── bi_level_runner.py      # Main entry script for Bi-level training and evaluation
├── model.py                # Core model architecture (Generator, Discriminator, GCN)
├── scorer.py               # Scorer module (Meta-learner for sample weighting)
├── utils.py                # Utility functions (Data loading, preprocessing)
├── run.py                  # (Optional) Baseline or alternative runner
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

> **Note**: The `src/` directory contains legacy or alternative implementations (e.g., GraphSAGE baselines) and is not required for the main `bi_level_runner.py` execution.

## ⚙️ Environment Setup

To set up the environment, please ensure you have **Python** (tested on 3.8+) and **Anaconda** installed.

> **Important**: This project depends on `dgl` (Deep Graph Library) and `torch`. Please install the version of DGL and PyTorch that matches your CUDA version.

```bash
# Create a virtual environment
conda create -n AJAX python=3.8
conda activate AJAX

# Install DGL 
pip install -r requirements.txt
```

## 💾 Data Preparation

The datasets should be placed in the `dataset/` directory. The project currently supports `.mat` format datasets (e.g., Reddit, Elliptic, Amazon).

**Directory Structure Example:**

```text
../
└── dataset/
    ├── reddit.mat
    ├── elliptic.mat
    └── ...
```

The data loading logic is handled in `utils.py` via the `load_mat` function.

## 🚀 Training and Inference

The main training script is `bi_level_runner.py`. It performs bi-level optimization where the **Scorer** (Meta-learner) is trained to assign weights to generated anomalies, and the **Generator/Discriminator** (Base-learner) are updated based on these weighted samples.

To run the model on the **Reddit** dataset:

```bash
python bi_level_runner.py --dataset reddit
```

### Common Arguments

You can configure the training using command-line arguments. Here are some key options:

- `--dataset`: Name of the dataset (default: `reddit`). support: `reddit`, `elliptic`, `photo`, `t_finance`.
- `--lr`: Learning rate for the main model (default: auto-configured based on dataset).
- `--lr_phi`: Learning rate for the Scorer (meta-learner).
- `--num_epoch`: Number of training epochs.
- `--lambda_align`: Weight for alignment loss (Prototype-based).
- `--lambda_gen`: Weight for generator adversarial loss.
- `--proto_score_alpha`: Alpha value for blending discriminator logits with prototype scores during inference (default is auto-tuned for Reddit).

**Example using specific hyperparameters:**

```bash
python bi_level_runner.py \
  --dataset reddit \
  --lambda_align 1.0 \
  --lambda_gen 0.5 \
  --num_epoch 300
```

## 📝 Citation

If you find this code useful, please cite our paper:

```bibtex
@inproceedings{Meta-ResGAD,
  title={...},
  author={...},
  booktitle={CIKM},
  year={2026}
}
```

## 📧 Contact

**Coming soon!**
