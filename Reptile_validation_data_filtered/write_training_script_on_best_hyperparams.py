import os
import pickle

with open('best_hyps_on_dem1066.pkl', 'rb') as handle:
    properties = pickle.load(handle)

with open('run_training_on_hyperparams.sh', 'w') as f:
    f.writelines('#!/usr/bin/env bash\n')
    f.writelines('#SBATCH --job-name=fotrcont\n')
    f.writelines('#SBATCH --account=hkg02\n')
    f.writelines('#SBATCH --partition=normal\n')
    f.writelines('#SBATCH --nodes=1\n')
    f.writelines('#SBATCH --ntasks-per-node=1\n')
    f.writelines('#SBATCH --cpus-per-task=2\n')
    f.writelines('#SBATCH --mem=16000\n')
    f.writelines('#SBATCH --mail-type=ALL\n')
    f.writelines('#SBATCH --mail-user=hkg02@mail.aub.edu\n')

    f.writelines('module load python/3\n')
    f.writelines('module load python/tensorflow-1.14.0\n')

    for k in properties:
        f.writelines('python main_foml_trans.py --shots {} --inner_batch {} --inner_iters {} --learning_rate {} --meta_step {} --meta_step_final {} --meta_batch {} --meta_iters {} --eval_batch {} --eval_iters {} --model_num "{}" --dim_hidden "{}" --activation_fn "{}" --weights_vector  "{}" --categorical_encoding "{}"  --logdir \"FOML_trans_trained_models/\"\n'.format(
            properties[k]['shots'],
            properties[k]['inner_batch'],
            properties[k]['inner_iters'],
            properties[k]['lr'],
            properties[k]['meta_step'],
            properties[k]['meta_step_final'],
            properties[k]['meta_batch'],
            properties[k]['meta_iters'],
            properties[k]['eval_batch'],
            properties[k]['eval_iters'],
            properties[k]['model'], # model num
            properties[k]['dim_hidden'],
            properties[k]['activation_fns'],
            properties[k]['weights'],
            properties[k]['encoding'],
        ))

    f.close()
