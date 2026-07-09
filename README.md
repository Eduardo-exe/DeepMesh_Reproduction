# DeepMesh Reproduction

This repository contains an educational and methodological reproduction inspired by the DeepMesh paper, focusing on 3D Reconstruction from Point Clouds. It has been designed to run entirely on native PyTorch, providing a structured, modular, and accessible approach to 3D mesh generation.

---

## 🚀 Reproducibility Guide

Follow the steps below to reproduce the complete pipeline.

### 1. Installation

We recommend using Anaconda/Miniconda to manage the environment.

```bash
# Create and activate the environment
conda create -n deepmesh python=3.10
conda activate deepmesh

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Dataset Preparation

Convert every supported 3D model contained in the `examples/` directory into the dataset format required by the pipeline.

```bash
python convert_dataset.py --input examples/ --output dataset/
```

The converter automatically processes every compatible model found in the `examples/` folder.

---

### 3. Running Inference

Run inference for all models contained in the `examples/` directory.

```bash
python infer.py --input examples/ --output outputs/ --num_points 100000
```

For every processed model, the pipeline automatically generates:
- Sampled Point Cloud (`.ply`)
- Reconstructed Mesh (`.obj`)
- Intermediate files (when enabled)

All generated results are saved inside the `outputs/` directory.

---

### 4. Evaluation and Metrics

Evaluate all reconstructed meshes against their corresponding ground-truth models.

```bash
python evaluate_metrics.py --gt examples/ --pred outputs/
```

The evaluation automatically matches each prediction with its corresponding input model and computes metrics such as:
- Chamfer Distance
- Hausdorff Distance

---

## 📂 Project Structure

```
DeepMesh_Reproduction/
├── examples/                 # Input 3D models
├── dataset/                  # Generated NPZ datasets
├── outputs/                  # Generated meshes and point clouds
├── convert_dataset.py
├── infer.py
├── evaluate_metrics.py
├── requirements.txt
└── README.md
```

---

## 📦 3D Models Source

The 3D models used in the `examples/` directory as ground-truth references were obtained from free assets available on **CGTrader**:

🔗 [https://www.cgtrader.com/3d-models?free=1](https://www.cgtrader.com/3d-models?free=1)

---

## 📌 Notes

- The `examples/` directory may contain one or multiple supported 3D models.
- Every compatible model found in the directory is processed automatically.
- Results preserve the original filenames to simplify evaluation and comparison.
