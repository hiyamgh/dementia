#!/usr/bin/env bash
#!/usr/bin/env bash
#SBATCH --job-name=gen10
#SBATCH --account=hkg02
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu

module load python/3

python compile_results_for_paper.py --top 20 --num 1000 --type reptile_trans