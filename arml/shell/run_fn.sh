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
#SBATCH --array=1-360%5

module load python/3
module load python/tensorflow-1.14.0

metatrain_iterations=(1000)
meta_batch_sizes=(32 16)
meta_lrs=(0.001 0.1)
update_lrs=(0.001 0.1)
dim_hidden=("128, 64" "128" "128, 64, 64")
IFS=""
activation_fns=("relu" "sigmoid" "tanh" "softmax" "swish")
num_updates=(4)
num_vertex=(4 6 8)
USCOUNTER=1

for miter in ${metatrain_iterations[@]}; do
  for mbs in ${meta_batch_sizes[@]}; do
    for mlr in ${meta_lrs[@]}; do
      for ulr in ${update_lrs[@]}; do
        for dh in ${dim_hidden[*]}; do
          for afn in ${activation_fns[@]}; do
            for nu in ${num_updates[@]}; do
               for nv in ${num_vertex[@]}; do
                   if [ $USCOUNTER -eq $SLURM_ARRAY_TASK_ID ]; then
                   echo "USCOUNTER: " $USCOUNTER
                   echo "pretrain_iterations: 0 metatrain_iterations: ${miter} meta_batch_size: ${mbs} meta_lr: ${mlr} update_batch_size: ${mbs} update_lr: ${ulr} num_updates: ${nu} dim_hidden: ${dh} model_num: $SLURM_ARRAY_TASK_ID activation_fn: ${afn} logdir: fn/ num_vertex ${nv}"
                   python main_fn.py --metatrain_iterations ${miter} --meta_batch_size ${mbs} --meta_lr ${mlr} --update_batch_size ${mbs} --update_lr ${ulr} --num_updates ${nu} --dim_hidden ${dh} --model_num $SLURM_ARRAY_TASK_ID --activation_fn ${afn} --logdir "fn/" --num_vertex ${nv}
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