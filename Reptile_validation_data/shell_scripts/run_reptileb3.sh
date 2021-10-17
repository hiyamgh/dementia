#!/usr/bin/env bash
#SBATCH --job-name=rpb3
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

shots=(5 10 15 20 25)
inner_batch=(10 20 25 100)
inner_iters=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
learning_rate=(0.001 0.0005 0.001)
meta_step=(1.0 0.25) # I think this is the outer step size
meta_step_final=(1.0 0.25)
meta_batch=(5 8 16 32) # 8, 16, and 32 are additional from our own
meta_iters=(1000 5000 10000 20000 40000)
eval_batch=(5 15 25)
eval_iters=(5 10 30 50) # 10, 30 are additional from our own
#eval_samples=()
#eval_interval=(10)
dim_hidden=("128, 64" "128" "128, 64, 64")
IFS=""
activation_fns=("relu" "sigmoid" "tanh" "softmax" "swish")
IFS=""
weights=("1, 1" "1, 10" "1, 100" "10, 1" "100, 1")
IFS=""
sampling_strategy=("minority" "not minority" "all" "0.5" "1" "0.75")
IFS=""
encoding=("catboost" "glmm" "target" "mestimator" "james" "woe")
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
                                                    for ss in ${sampling_strategy[*]}; do
                                                        for e in ${encoding[*]}; do
                                                            for tf in ${top_features[@]}; do
                                                                if [ $USCOUNTER -eq $((SLURM_ARRAY_TASK_ID+1801)) ]; then
                                                                    found_script=true
                                                                    echo "USCOUNTER: " $USCOUNTER
                                                                    echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
                                                                    echo "main_reptile.py --shots ${shts} --inner_batch ${ib} --inner_iters ${ii} --learning_rate ${lr} --meta_step ${ms} --meta_step_final ${msf} --meta_batch ${mb} --meta_iters ${mi} --eval_batch ${eb} --eval_iters ${ei} --model_num $USCOUNTER --dim_hidden ${dh} --activation_fn ${af} --weights_vector ${w} --sampling_strategy ${ss} --categorical_encoding ${e} --top_features ${tf} --logdir "trained_models/${tf}/""
                                                                    python main_reptile.py --shots ${shts} --inner_batch ${ib} --inner_iters ${ii} --learning_rate ${lr} --meta_step ${ms} --meta_step_final ${msf} --meta_batch ${mb} --meta_iters ${mi} --eval_batch ${eb} --eval_iters ${ei} --model_num $USCOUNTER --dim_hidden ${dh} --activation_fn ${af} --weights_vector ${w} --sampling_strategy ${ss} --categorical_encoding ${e} --top_features ${tf} --logdir "reptile_trained_models/${tf}/"
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
done