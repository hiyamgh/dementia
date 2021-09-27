#!/usr/bin/env bash
#SBATCH --job-name=crt_fps
#SBATCH --partition=large
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=200000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu

module load python/3
python create_fps_dementia.py