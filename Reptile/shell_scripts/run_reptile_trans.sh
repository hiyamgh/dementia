#!/usr/bin/env bash
#SBATCH --job-name=rptrb1
#SBATCH --account=hkg02
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu
#SBATCH --array=1-900%5

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
top_features=(10 20)
USCOUNTER=1
found_script=false


for shts in ${shots[@]}; do
    for ib in ${inner_batch[@]}; do
        for ii in ${inner_iters[@]}; do
            for lr in ${learning_rate[@]}; do
                for ms in ${meta_step[@]}; do
                    for msf in ${meta_step_final[@]}; do
                        for mb in ${meta_batch[@]}; do
                            for mi in ${meta_iters[@]}; do
                                for eb in ${eval_batch[@]}; do
                                    for ei in ${eval_iters[@]}; do
                                        for dh in ${dim_hidden[*]}; do
                                            for af in ${activation_fns[*]}; do
                                                for w in ${weights[*]}; do
                                                    for e in ${encoding[*]}; do
                                                        for tf in ${top_features[@]}; do
                                                            if [ $USCOUNTER -eq $SLURM_ARRAY_TASK_ID ]; then
                                                                    found_script=true
                                                                    echo "USCOUNTER: " $USCOUNTER
                                                                    echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
                                                                    echo "main_reptile_trans.py --shots ${shts} --inner_batch ${ib} --inner_iters ${ii} --learning_rate ${lr} --meta_step ${ms} --meta_step_final ${msf} --meta_batch ${mb} --meta_iters ${mi} --eval_batch ${eb} --eval_iters ${ei} --model_num $USCOUNTER --dim_hidden ${dh} --activation_fn ${af} --weights_vector ${w} --categorical_encoding ${e} --top_features ${tf} --logdir "reptile_trans_trained_models/${tf}/""
                                                                    python main_reptile_trans.py --shots ${shts} --inner_batch ${ib} --inner_iters ${ii} --learning_rate ${lr} --meta_step ${ms} --meta_step_final ${msf} --meta_batch ${mb} --meta_iters ${mi} --eval_batch ${eb} --eval_iters ${ei} --model_num $USCOUNTER --dim_hidden ${dh} --activation_fn ${af} --weights_vector ${w} --categorical_encoding ${e} --top_features ${tf} --logdir "reptile_trans_trained_models/${tf}/"
                                                            fi
                                                            USCOUNTER=$(expr $USCOUNTER + 1)
                                                            echo "incremented USCOUNTER, now it is: " $USCOUNTER
                                                        done
                                                        if [ "$found_script" = true ] ; then
                                                                    break 15
                                                        fi
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done