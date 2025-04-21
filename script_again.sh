#!/bin/bash
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=%N-%j.out
#SBATCH --qos=normal
#SBATCH --account=winter2025-comp579 
#SBATCH --propagate=NONE
pip install -r requirements.txt
export PATH="$HOME/.local/bin:$PATH"

module load miniconda/miniconda-winter2025

python ~/comp579_project/max_script.py
