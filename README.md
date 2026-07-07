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

To prepare a 3D model for our pipeline, you need to convert it into the expected `.npz` format:

```bash
python convert_dataset.py --input examples/JEEPCOMPASS2021.obj --output dataset/shape.npz
```

### 3. Running Inference

To generate a 3D Mesh from a Point Cloud representation, run the inference script. You can specify the number of points to sample:

```bash
python infer.py --input examples/JEEPCOMPASS2021.obj --output outputs/ --num_points 100000
```
> **Output:** This will automatically export the generated Point Cloud (`.ply`) and the reconstructed watertight Mesh (`.obj`) with gradients into the `outputs/` directory.

### 4. Evaluation and Metrics

To evaluate the generated mesh against the Ground Truth (or other models like MeshAnythingV2 / BPT) using Chamfer Distance and Hausdorff Distance, use the evaluation script:

```bash
python evaluate_metrics.py --gt examples/JEEPCOMPASS2021.obj --pred outputs/JEEPCOMPASS2021_mesh.obj
```
This command will output the computed metrics necessary for benchmark comparisons.
