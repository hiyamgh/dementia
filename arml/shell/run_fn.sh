#!/usr/bin/env bash
#SBATCH --job-name=fnarml
#SBATCH --account=hkg02
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu
#SBATCH --array=1-720%5

module load python/3
module load python/tensorflow-1.14.0

metatrain_iterations=(1000)
meta_batch_sizes=(32)
meta_lrs=(0.1)
update_lrs=(0.1)
dim_hidden=("128, 64" "128" "128, 64, 64")
IFS=""
activation_fns=("relu" "sigmoid" "tanh" "softmax" "swish")
num_updates=(2 4 10)
num_vertex=(2 4 8 10)
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
                 for nf in ${num_filters[@]}; do
                   for elw in ${emb_loss_weight[@]}; do
                      if((USCOUNTER=SLURM_ARRAY_TASK_ID)); then
                           echo "USCOUNTER: " $USCOUNTER
                           echo "$SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
                           echo "main_fn.py --metatrain_iterations ${miter} --meta_batch_size ${mbs} --meta_lr ${mlr} --update_batch_size ${mbs} --update_lr ${ulr} --num_updates ${nu} --dim_hidden ${dh} --model_num $USCOUNTER --activation_fn ${afn} --logdir "fn_batch2/" --num_vertex ${nv} --num_filters ${nf} --emb_loss_weight ${elw}"
                           python main_fn.py --metatrain_iterations ${miter} --meta_batch_size ${mbs} --meta_lr ${mlr} --update_batch_size ${mbs} --update_lr ${ulr} --num_updates ${nu} --dim_hidden ${dh} --model_num $USCOUNTER --activation_fn ${afn} --logdir "fn_batch2/" --num_vertex ${nv} --num_filters ${nf} --emb_loss_weight ${elw}
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