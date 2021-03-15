#!/usr/bin/env bash
#SBATCH --job-name=fn_fp
#SBATCH --account=hkg02
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu
#SBATCH --array=1-722%10

module load python/3
module load python/tensorflow-1.14.0

metatrain_iterations=(1000)
meta_batch_sizes=(32 16 8 4)
meta_lrs=(0.1 0.001)
update_lrs=(0.1 0.001)
dim_hidden=("128, 64, 64" "128, 64" "128")
IFS=""
activation_fns=("relu" "sigmoid" "tanh" "softmax" "swish")
num_updates=(4)
include_fp=("1")
fp_supps=(0.8 0.9 0.7)
USCOUNTER=1
#721

for miter in ${metatrain_iterations[@]}; do
  for mbs in ${meta_batch_sizes[@]}; do
    for mlr in ${meta_lrs[@]}; do
      for ulr in ${update_lrs[@]}; do
        for dh in ${dim_hidden[*]}; do
          for afn in ${activation_fns[@]}; do
            for nu in ${num_updates[@]}; do
              for ifp in ${include_fp[@]}; do
                for fp_supp in ${fp_supps[@]}; do
                 if [ $USCOUNTER -eq $SLURM_ARRAY_TASK_ID ]; then
                 echo "USCOUNTER: " $USCOUNTER
                 echo "pretrain_iterations: 0 metatrain_iterations: ${miter} meta_batch_size: ${mbs} meta_lr: ${mlr} update_batch_size: ${mbs} update_lr: ${ulr} num_updates: ${nu} dim_hidden: ${dh} model_num: $SLURM_ARRAY_TASK_ID activation_fn: ${afn}  includefp: ${ifp}  fp_file: fake_news_fps_colsmeta/fps_fakenews_$fp_supp.pickle colsmeta_file: fake_news_fps_colsmeta/colsmeta_fakenews_$fp_supp.pickle logdir: fake_news_with_fp"
                 python main2.py  --pretrain_iterations 0 --metatrain_iterations ${miter} --meta_batch_size ${mbs} --meta_lr ${mlr} --update_batch_size ${mbs} --update_lr ${ulr} --num_updates ${nu} --dim_hidden ${dh} --model_num $SLURM_ARRAY_TASK_ID --activation_fn ${afn} --include_fp ${ifp}  --fp_file "fake_news_fps_colsmeta/fps_fakenews_${fp_supp}.pickle" --colsmeta_file "fake_news_fps_colsmeta/colsmeta_fakenews_${fp_supp}.pickle" --logdir "fake_news_with_fp2"
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
# echo "total combinations: " $USCOUNTER