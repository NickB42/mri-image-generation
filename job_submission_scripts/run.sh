#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=5:00:00
##SBATCH --cpus-per-task=16
##SBATCH --gres=gpu:1

source venv/bin/activate

# python experiments/ddpm_25d/prep_all.py --root_dir data/brats-2021 --output_file data/preprocessed_all_debug.npy --debug
#python experiments/ddpm_25d/prep_all.py --root_dir data/brats-2021 --output_file data/preprocessed_all.npy
#python experiments/ddpm_25d/prep.py --root_dir data/brats-2021 --output_dir data/preprocessed_masks
#python experiments/ddpm_25d/test.py
# python experiments/ddpm_25d/test_dataloader.py --workers 0
#python experiments/ddpm_25d/sample.py
#python experiments/ddpm_25d/sample3d.py