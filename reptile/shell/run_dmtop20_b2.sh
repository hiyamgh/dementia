#!/usr/bin/env bash
#SBATCH --job-name=drp20b2
#SBATCH --account=hkg02
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu
#SBATCH --array=1-720%2

module load python/3
module load python/tensorflow-1.14.0

dim_hidden=("128, 64" "128" "128, 64, 64")
IFS=""
activation_fns=("relu")
IFS=""
shots=(32)
train_shots=(32)
inner_batch=(5)
learning_rate=(0.1)
meta_batch=(1 5 10)
weights=("1, 1" "1, 10" "1, 100" "10, 1" "100, 1")
IFS=""
sampling_strategy=("minority" "not minority" "all" "0.5" "1" "0.75")
IFS=""
encoding=("catboost" "glmm" "target" "mestimator" "james" "woe")
IFS=""
USCOUNTER=1

for dh in ${dim_hidden[*]}; do
    for af in ${activation_fns[*]}; do
        for s in ${shots[@]}; do
            for ts in ${train_shots[@]}; do
                for ib in ${inner_batch[@]}; do
                    for lr in ${learning_rate[@]}; do
                        for mb in ${meta_batch[@]}; do
                            for w in ${weights[@]}; do
                                for ss in ${sampling_strategy[@]}; do
                                    for e in ${encoding[@]}; do
                                        if [ $USCOUNTER -eq $((SLURM_ARRAY_TASK_ID+900)) ]; then
                                            echo "USCOUNTER: " $USCOUNTER
                                            echo "$SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
                                            echo "main_dm.py --dim_hidden ${dh} --activation_fn ${af} --shots ${s} --train-shots ${ts} --inner-batch ${ib} --learning-rate ${lr} --save_dir "dm_top20/" --model_num $USCOUNTER --meta-batch ${mb} --weights_vector ${w} --sampling_strategy ${ss} --categorical_encoding ${e} --top_features 20"
                                            python main_dm.py --dim_hidden ${dh} --activation_fn ${af} --shots ${s} --train-shots ${ts} --inner-batch ${ib} --learning-rate ${lr} --save_dir "dm_top20/" --model_num $USCOUNTER --meta-batch ${mb} --weights_vector ${w} --sampling_strategy ${ss} --categorical_encoding ${e} --top_features 20
                                        fi
                                        USCOUNTER=$(expr $USCOUNTER + 1)
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