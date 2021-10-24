#!/usr/bin/env bash
#SBATCH --job-name=rt10
#SBATCH --account=hkg02
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu

module load python/3
module load python/tensorflow-1.14.0

python main_reptile_trans.py --shots 10 --inner_batch 5 --inner_iters 10 --learning_rate 0.001 --meta_step 1 --meta_step_final 1 --meta_batch 4 --meta_iters 1000 --eval_batch 5 --eval_iters 10 --model_num 1 --dim_hidden "128, 64, 64" --activation_fn swish --categorical_encoding woe --top_features 10 --weights_vector "1, 10" --logdir "reptile_trans_bss_corr/10/"

# 863_test
# 863	10	5	10	0.001	1	1	4	1000	5	10	128, 64, 64	swish	10