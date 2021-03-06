#!/usr/bin/env bash
#SBATCH --job-name=dmfp10
#SBATCH --account=hkg02
#SBATCH --partition=arza
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
meta_lrs=(0.001 0.1)
update_lrs=(0.001 0.1)
dim_hidden=("128, 64, 64" "128, 64" "128")
IFS=""
activation_fns=("relu")
num_updates=(4)
include_fp=("1")
fp_supps=(0.8 0.9 0.7)
weights=("1, 1" "1, 10" "1, 100" "10, 1" "100, 1")
IFS=""
sampling_strategy=("minority" "not minority" "all" "0.5" "1" "0.75")
IFS=""
USCOUNTER=1

for miter in ${metatrain_iterations[@]}; do
  for mbs in ${meta_batch_sizes[@]}; do
    for mlr in ${meta_lrs[@]}; do
      for ulr in ${update_lrs[@]}; do
        for dh in ${dim_hidden[*]}; do
          for afn in ${activation_fns[@]}; do
            for nu in ${num_updates[@]}; do
              for ifp in ${include_fp[@]}; do
                for fp_supp in ${fp_supps[@]}; do
                  for wt in ${weights[@]}; do
                    for ss in ${sampling_strategy[@]}; do
                     if [ $USCOUNTER -eq $SLURM_ARRAY_TASK_ID ]; then
                     echo "USCOUNTER: " $USCOUNTER
                     echo "pretrain_iterations: 0 metatrain_iterations: ${miter} meta_batch_size: ${mbs} meta_lr: ${mlr} update_batch_size: ${mbs} update_lr: ${ulr} num_updates: ${nu} dim_hidden: ${dh} model_num: $SLURM_ARRAY_TASK_ID activation_fn: ${afn}  includefp: ${ifp} fp_file: dementia_colsmeta_top10/fps_dementia_${fp_supp}.pickle colsmeta_file dementia_colsmeta_top10/colsmeta_dementia_${fp_supp}.pickle weights_vector ${wt} --sampling_strategy ${ss} --top_features 10 --logdir dementia_with_fp_top10"
                     python main.py  --pretrain_iterations 0 --metatrain_iterations ${miter} --meta_batch_size ${mbs} --meta_lr ${mlr} --update_batch_size ${mbs} --update_lr ${ulr} --num_updates ${nu} --dim_hidden ${dh} --model_num $SLURM_ARRAY_TASK_ID --activation_fn ${afn} --include_fp ${ifp}  --fp_file "dementia_colsmeta_top10/fps_dementia_${fp_supp}.pickle" --colsmeta_file "dementia_colsmeta_top10/colsmeta_dementia_${fp_supp}.pickle" --weights_vector ${wt} --sampling_strategy ${ss} --top_features 10 --logdir "dementia_with_fp_top10"
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