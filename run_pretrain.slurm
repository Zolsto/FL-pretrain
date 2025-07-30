#!/bin/bash
#SBATCH --job-name pre-sint
#SBATCH --ntasks 1
#SBATCH --time 35:00:00
#SBATCH --partition allgroups
#SBATCH --output out_%x.txt
#SBATCH --error err_%x.txt
#SBATCH --mem=8G
#SBATCH --gres=gpu:rtx

# Comandi da eseguire

# Attiva enviroment
source ~/.bashrc
conda activate foundation-fed

# Naviga nella directory
cd /nfsd/lttm4/tesisti/Bortolatto/codice/locale

# Esegui lo script
nvidia-smi
python3 pretrain.py
