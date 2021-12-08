#!/usr/bin/env bash
#SBATCH --job-name=fotrcont
#SBATCH --account=hkg02
#SBATCH --partition=large
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu

module load python/3
module load python/tensorflow-1.14.0

shots=(10 20 25) # number of examples per class: 3
inner_batch=(5 10 15) # mini-batch size in meta-training: 3
inner_iters=(10 20 30) # number of mini-batches in meta-training: 3
learning_rate=(0.001 0.0005) # adam step size: 2
meta_step=(1.0 0.7 0.5 0.25) # (Outer step size) epsilon in training/testing iteration: 4
meta_step_final=(1.0 0.7 0.5 0.25) # (Outer step size) epsilon in final training/testing iteration: 4
meta_batch=(4 8 16 32) # number of tasks: 4
meta_iters=(1000 5000 10000) # number of training iterations: 3
eval_batch=(5 10 15) # size of mini-batches in meta testing: 3
eval_iters=(10 20 30) # number of mini-batches in meta testing: 3
dim_hidden=("128, 64" "128" "128, 64, 64") # dim hidden: 3
IFS=""
activation_fns=("relu" "sigmoid" "tanh" "softmax" "swish") # activation function: 5
IFS=""
weights=("1, 1" "1, 10" "1, 100" "10, 1" "100, 1") # wights cost sensitive: 5
IFS=""
encoding=("catboost" "glmm" "target" "mestimator" "james" "woe") # 6
IFS=""

python main_foml_trans.py --shots 10 --inner_batch 5 --inner_iters 10 --learning_rate 0.001 --meta_step 1 --meta_step_final 1 --meta_batch 8 --meta_iters 5000 --eval_batch 15 --eval_iters 30 --model_num 1 --dim_hidden "128, 64" --activation_fn "softmax" --weights_vector  "1, 10" --categorical_encoding "james"  --logdir "FOML_trans_trained_models/"
