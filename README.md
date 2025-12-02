# 3D Brain MRI Generation with Diffusion Models (BraTS 2021)
This repository contains experiments on generating 3D brain MRI volumes using diffusion models, trained on the BraTS 2021 dataset.  

---

## Project Structure

```text
.
├── dataset/
├── project/
└── try_around/
    ├── models/
    ├── explore_dataset.ipynb
    └── base_model.ipynb
```

- dataset/

    Expected location for the BraTS 2021 data (not tracked in version control)

- project/

    Placeholder for future experiments, refactored modules, training scripts, etc.

- try_around/
    - models/ 

        Suggested place for model definitions and/or saved weights and checkpoints
    - explore_dataset.ipynb 

        Notebook for inspecting the dataset, loading MRI volumes, and visualizing slices
    - base_model.ipynb

        Notebook for prototyping a baseline diffusion model for 2D image generation

## Setup
1. Clone the repository
```bash
git clone <your-repo-url>.git
cd <your-repo-name>
```

2. (Optional) Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Dataset: BraTS 2021

Download the BraTS 2021 dataset from the official source ([MICCAI BraTS challenge](https://www.med.upenn.edu/cbica/brats2021/)).
Or alternativly from [Kaggle](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1/data).

Extract the dataset into the dataset/ directory at the root of this repository.

Example layout (schematic):

```text
dataset/
  BraTS2021_00000/
    BraTS2021_00000_t1.nii.gz
    BraTS2021_00000_t1ce.nii.gz
    BraTS2021_00000_t2.nii.gz
    BraTS2021_00000_flair.nii.gz
    BraTS2021_00000_seg.nii.gz
  BraTS2021_00001/
    ...
```

Make sure the paths used in the notebooks (e.g. a DATASET_ROOT variable) point to this dataset/ folder.