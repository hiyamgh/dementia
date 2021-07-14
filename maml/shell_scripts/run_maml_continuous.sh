#!/usr/bin/env bash
#SBATCH --job-name=dm10
#SBATCH --account=hkg02
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hkg02@mail.aub.edu
#SBATCH --nodelist=onode07
#SBATCH --array=1-361%5

module load python/3
module load python/tensorflow-1.14.0

metatrain_iterations=(1000 5000 10000) # 3
meta_batch_sizes=(4 8 16 32) # 4
update_batch_size=(4 8 16 32) # 4
meta_lrs=(0.001 0.1) # 2
update_lrs=(0.001 0.1) # 2
dim_hidden=("128, 64, 64" "128, 64" "128") # 3
IFS=""
activation_fns=("relu" "sigmoid" "tanh" "softmax" "swish") # 5
IFS=""
num_updates=(1 4 10) # 3
include_fp=("0")
weights=("1, 1" "1, 10" "1, 100" "10, 1" "100, 1") # 5
IFS=""
encoding=("catboost" "glmm" "target" "mestimator" "james" "woe") # 6
IFS=""
top_features=(10 20) # 2
USCOUNTER=1
ADDWHAT=$1 # initially 0
NUMCALLS=$2 # initially 1
found_script=false

# 518,400 / 900 = 576

for miter in ${metatrain_iterations[@]}; do
  for mbs in ${meta_batch_sizes[@]}; do
    for ubs in ${update_batch_size[@]}; do
        for mlr in ${meta_lrs[@]}; do
          for ulr in ${update_lrs[@]}; do
            for dh in ${dim_hidden[@]}; do
              for afn in ${activation_fns[@]}; do
                for nu in ${num_updates[@]}; do
                  for ifp in ${include_fp[@]}; do
                    for wt in ${weights[@]}; do
                      for tf in ${top_features[@]}; do
                        if [ $USCOUNTER -eq $((SLURM_ARRAY_TASK_ID+ADDWHAT)) ]; then
                          found_script=true
                          echo "USCOUNTER: " $USCOUNTER
                          echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
                          echo "ADDWHAT" $ADDWHAT
                          echo "main.py --metatrain_iterations ${miter} --meta_batch_size ${mbs} --update_batch_size ${ubs} --meta_lr ${mlr} --update_lr ${ulr} --num_updates ${nu} --dim_hidden ${dh} --model_num $USCOUNTER --activation_fn ${afn} --include_fp ${ifp} --weights_vector ${wt} --top_features ${tf} --logdir MAML_trained_models/${tf}/"
                          python main.py --metatrain_iterations ${miter} --meta_batch_size ${mbs} --update_batch_size ${ubs} --meta_lr ${mlr} --update_lr ${ulr} --num_updates ${nu} --dim_hidden ${dh} --model_num $USCOUNTER --activation_fn ${afn} --include_fp ${ifp} --weights_vector ${wt} --top_features ${tf} --logdir "MAML_trained_models/${tf}/"
                        fi
                        USCOUNTER=$(expr $USCOUNTER + 1)
                      done
                      if [ "$found_script" = true ] ; then
                         echo "found script..."
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

echo "checking if the array task ID is equal to 900, if yes, then execute ..."
if [ $SLURM_ARRAY_TASK_ID -eq 900 ] && [ $NUMCALLS -lt 576 ]; then
    echo "I am the job 900"
    sleep 2m
    ADDWHAT=$((ADDWHAT + 900))
    NUMCALLS=$((NUMCALLS + 1))
    sbatch run_foml_trans_continuous.sh $ADDWHAT $NUMCALLS
fi