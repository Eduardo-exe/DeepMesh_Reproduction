# DeepMesh Reproduction

This repository contains an educational and methodological reproduction inspired by the DeepMesh paper, focusing on 3D Reconstruction from Point Clouds. It has been designed to run entirely on native PyTorch, providing a structured, modular, and accessible approach to 3D mesh generation.

## 🚀 Reproducibility Guide

Follow these steps to reproduce our environment and results.

### 1. Installation

We recommend using Anaconda/Miniconda to manage the environment:

```bash
# Create and activate the environment
conda create -n deepmesh python=3.10
conda activate deepmesh

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

Convert all models inside the `examples/` directory into the dataset format expected by the pipeline:

```bash
python convert_dataset.py --input examples/ --output dataset/
```

This command will process every supported 3D model contained in the `examples/` folder and generate the corresponding `.npz` files.

### 3. Running Inference

Run inference for every model contained in the `examples/` directory:

```bash
python infer.py --input examples/ --output outputs/ --num_points 100000
```

> **Output:** For each input model, the pipeline automatically exports:
>
> - Sampled Point Cloud (`.ply`)
> - Reconstructed watertight Mesh (`.obj`)
> - Additional intermediate outputs (when enabled)

All generated files are stored inside the `outputs/` directory.

### 4. Evaluation and Metrics

Evaluate every reconstructed mesh against its corresponding Ground Truth:

```bash
python evaluate_metrics.py --gt examples/ --pred outputs/
```

The script automatically matches the generated meshes with their corresponding ground-truth models and reports metrics such as:

- Chamfer Distance
- Hausdorff Distance
- Other supported benchmark metrics
