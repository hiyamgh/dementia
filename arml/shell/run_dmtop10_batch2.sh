#!/usr/bin/env bash
#SBATCH --job-name=dmarmlb2
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

metatrain_iterations=(1000)
meta_batch_sizes=(16)
meta_lrs=(0.1)
update_lrs=(0.1)
dim_hidden=("128, 64" "128" "128, 64, 64")
IFS=""
activation_fns=("relu" "sigmoid" "tanh" "softmax" "swish")
IFS=""
num_updates=(4)
weights=("1, 1" "1, 10" "1, 100" "10, 1" "100, 1")
IFS=""
sampling_strategy=("minority" "not minority" "all" "0.5" "1" "0.75")
IFS=""
encoding=("catboost" "glmm" "target" "mestimator" "james" "woe")
IFS=""
num_vertex=(4 6 8)
num_filters=(16 32)
emb_loss_weight=(0.01 0.1)
USCOUNTER=1

for miter in ${metatrain_iterations[@]}; do
  for mbs in ${meta_batch_sizes[@]}; do
    for mlr in ${meta_lrs[@]}; do
      for ulr in ${update_lrs[@]}; do
        for dh in ${dim_hidden[*]}; do
          for afn in ${activation_fns[@]}; do
            for nu in ${num_updates[@]}; do
              for nv in ${num_vertex[@]}; do
                for wt in ${weights[@]}; do
                  for ss in ${sampling_strategy[@]}; do
                    for e in ${encoding[@]}; do
                      for nf in ${num_filters[@]}; do
                        for elw in ${emb_loss_weight[@]}; do
                          if ((USCOUNTER=SLURM_ARRAY_TASK_ID+900)); then
                              echo "USCOUNTER: " $USCOUNTER
                              echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
                              echo "main_fn.py --metatrain_iterations ${miter} --meta_batch_size ${mbs} --meta_lr ${mlr} --update_batch_size ${mbs} --update_lr ${ulr} --num_updates ${nu} --dim_hidden ${dh} --model_num $USCOUNTER --activation_fn ${afn} --categorical_encoding ${e} --num_vertex ${nv} --weights_vector ${wt} --sampling_strategy ${ss} --top_features 10 --logdir "dm_top10/" --num_filters ${nf} --emb_loss_weight ${elw}"
                              python main_dm.py --metatrain_iterations ${miter} --meta_batch_size ${mbs} --meta_lr ${mlr} --update_batch_size ${mbs} --update_lr ${ulr} --num_updates ${nu} --dim_hidden ${dh} --model_num $USCOUNTER --activation_fn ${afn} --categorical_encoding ${e} --num_vertex ${nv} --weights_vector ${wt} --sampling_strategy ${ss} --top_features 10 --logdir "dm_top10/" --num_filters ${nf} --emb_loss_weight ${elw}
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
    done
  done
done