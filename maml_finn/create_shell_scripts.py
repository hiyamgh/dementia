import os
import itertools

# #SBATCH --job-name=graphnei
# #SBATCH --account=hkg02
#
# ## specify the required resources
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=8
# #SBATCH --time=0-08:00:00
# #SBATCH --mail-type=ALL
# #SBATCH --mail-user=hkg02@mail.aub.edu
#
# #
# # add your command here, e.g
# #
# module load python/3


def create_scripts(metatrain_iterations,
                   meta_batch_sizes,
                   meta_lrs,
                   update_batch_sizes,
                   update_lrs,
                   num_updates,
                   fp_supports,
                   dims,
                   activation_fns):

    all_hyper_params = [metatrain_iterations, meta_batch_sizes, meta_lrs, update_batch_sizes,
                        update_lrs, num_updates, fp_supports, dims, activation_fns]

    all_combinations = list(itertools.product(*all_hyper_params))
    print('length of all combinations: {}'.format(len(all_combinations)))
    for i, combination in enumerate(all_combinations):
        with open('job{}.sh'.format(i), 'w') as f:
            f.writelines("#!/usr/bin/env bash\n")
            f.writelines("#SBATCH --job-name={}\n".format('job{}'.format(i)))
            f.writelines("#SBATCH --account=hkg02\n")
            f.writelines("#SBATCH --nodes=1\n")
            f.writelines("#SBATCH --time=0-08:00:00\n")
            f.writelines("#SBATCH --mail-type=ALL\n")
            f.writelines("#SBATCH --mail-user=hkg02@mail.aub.edu\n\n")
            f.writelines("module load python/3\n")
            f.writelines("python main.py \\\n")
            f.writelines("--pretrain_iterations 0 \\\n")
            f.writelines("--metatrain_iterations {} \\\n".format(combination[0]))
            f.writelines("--meta_batch_size {} \\\n".format(combination[1]))
            f.writelines("--meta_lr {} \\\n".format(combination[2]))
            f.writelines("--update_batch_size {} \\\n".format(combination[3]))
            f.writelines("--update_lr {} \\\n".format(combination[4]))
            f.writelines("--num_updates {} \\\n".format(combination[5]))
            f.writelines("--supp_fp {} \\\n".format(combination[6]))
            f.writelines("--dim_hidden {} \\\n".format(' '.join([str(d) for d in combination[7][1]])))
            f.writelines("--dim_name '{}' \\\n".format(combination[7][0]))
            f.writelines("--activation_fn '{}' \\\n".format(combination[8]))
            f.writelines("--logdir 'jobs' \\\n")
            f.writelines("> out_job{}.txt".format(i))
            f.close()


if __name__ == '__main__':
    metatrain_iterations = [1000, 2000, 10000]
    meta_batch_sizes = [4, 8, 16, 32]
    meta_lrs = [1e-3, 1e-1, 0.3]
    update_batch_sizes = [4, 8, 16, 32]
    update_lrs = [1e-3, 1e-1, 0.3]
    num_updates = [4]
    fp_supports = [0.7, 0.8]
    dim_hidden = [[256, 128, 64, 64], [128, 64, 64], [256, 128, 64], [128, 64]]
    dim_names = ['dim{}'.format(i) for i in range(len(dim_hidden))]
    dims = list(zip(dim_names, dim_hidden))
    activation_fns = ['relu', 'sigmoid', 'tanh', 'softmax', 'swish']

    # metatrain_iterations = [1000]
    # meta_batch_sizes = [16]
    # meta_lrs = [1e-1]
    # update_batch_sizes = [16]
    # update_lrs = [1e-1]
    # num_updates = [4]
    # fp_supports = [0.8]
    # dim_hidden = [[256, 128, 64, 64]]
    # dim_names = ['dim{}'.format(i) for i in range(len(dim_hidden))]
    # dims = list(zip(dim_names, dim_hidden))
    # activation_fns = ['relu']

    create_scripts(metatrain_iterations, meta_batch_sizes, meta_lrs,
                   update_batch_sizes, update_lrs, num_updates,
                   fp_supports, dims, activation_fns)
