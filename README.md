# 3D Brain MRI Generation with Diffusion Models (BraTS 2021)

This repository contains experiments on generating 3D brain MRI volumes using diffusion models, trained on the BraTS 2021 dataset.

---

## Project Structure

```text
.
├── dataset/
├── evaluation_scripts/
├── explorations/
├── model_scripts/
│   ├── model_folder/
│   └── [...]
├── venv/
├── .env
├── .gitignore
├── .perun.ini
├── job_submission.sh
├── README.md
└── requirements.txt

```

- dataset/

    Expected location for the BraTS 2021 data (not tracked in version control)

- evaluation_scripts/

    Scripts for evaluating trained models (metrics, quantitative comparisons, etc.)

- explorations/

    Notebooks and scripts for data exploration, visualizations and absic experiments

- model_scripts/
    
    Main training / inference code for diffusion models

- .env_template

    Environment variable configuration

- .perun.ini

    Global Perun configuration for experiment tracking

- job_submission.sh
    
    Helper script to submit jobs to the compute cluster

- requirements.txt

    Python dependencies


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

3. Create .env
- Copy .env_template to .env and edit values as needed
- Necessary variables:
    - MLFLOW_TRACKING_USERNAME=YOUR_USERNAME_HERE
    - MLFLOW_TRACKING_PASSWORD=YOUR_PASSWORD_HERE
    - MLFLOW_TRACKING_URI=YOUR_MLFLOW_TRACKING_URI_HERE

4. Install dependencies
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

## Training

### Submitting a training job (SLURM)

Training is launched via the `job_submission.sh` script, which wraps `sbatch` and runs a Python module inside the project’s virtual environment.

From the project root:

```bash
# Generic pattern
sbatch job_submission.sh <module_or_path> [args...]

# Train the slice-conditioned 2D DDPM (using the module path)
sbatch job_submission.sh model_scripts.slice_cond_2d_ddpm.model

# Same script, referenced via file path
sbatch job_submission.sh model_scripts/slice_cond_2d_ddpm/model.py
```

The script will:

1. Change to the submission directory ($SLURM_SUBMIT_DIR).

2. Load your shell config ($HOME/.bashrc), project .env, and venv

3. Run python -m <module> with any extra arguments you pass

4. Save logs to 
    model_scripts/slice_cond_2d_ddpm/logs/model/<JOB_ID>.out (for the example above)

You can monitor job status with:
```bash
squeue -u $USER
```