#!/usr/bin/env bash
#SBATCH --job-name=fnrp
#SBATCH --account=hkg02
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu
#SBATCH --array=1-810%2

module load python/3
module load python/tensorflow-1.14.0

dim_hidden=("128, 64" "128" "128, 64, 64")
IFS=""
activation_fns=("relu" "sigmoid" "tanh" "softmax" "swish")
IFS=""
shots=(32)
train_shots=(0 16 32)
inner_batch=(5 10 15)
learning_rate=(0.1 0.001)
meta_batch=(1 5 10)

USCOUNTER=1
for dh in ${dim_hidden[*]}; do
    for af in ${activation_fns[*]}; do
        for s in ${shots[@]}; do
            for ts in ${train_shots[@]}; do
                for ib in ${inner_batch[@]}; do
                    for lr in ${learning_rate[@]}; do
                        for mb in ${meta_batch[@]}; do
                            if [ $USCOUNTER -eq $SLURM_ARRAY_TASK_ID ]; then
                                echo "USCOUNTER: " $USCOUNTER
                                echo "$SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
                                echo "main_fn.py --dim_hidden ${dh} --activation_fn ${af} --shots ${s} --train-shots ${ts} --inner-batch ${ib} --learning-rate ${lr} --save_dir "fn/" --model_num $USCOUNTER --meta-batch ${mb}"
                                python main_fn.py --dim_hidden ${dh} --activation_fn ${af} --shots ${s} --train-shots ${ts} --inner-batch ${ib} --learning-rate ${lr} --save_dir "fn/" --model_num $USCOUNTER --meta-batch ${mb}
                            fi
                            USCOUNTER=$(expr $USCOUNTER + 1)
                        done
                    done
                done
            done
        done
    done
done