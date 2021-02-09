import os
import itertools


def create_scripts(metatrain_iterations, meta_batch_sizes, meta_lrs,
                   update_batch_sizes, update_lrs, num_updates,
                   fp_supports, dims, activation_fns, cost_sensitive,
                   scaling,
                   out_dir_name):

    all_hyper_params = [metatrain_iterations, meta_batch_sizes, meta_lrs, update_batch_sizes,
                        update_lrs, num_updates, fp_supports, dims, activation_fns, cost_sensitive, scaling]

    all_combinations = list(itertools.product(*all_hyper_params))
    print('length of all combinations: {}'.format(len(all_combinations)))
    for i, combination in enumerate(all_combinations):
        with open('job{}.sh'.format(i), 'w') as f:
            f.writelines("#!/usr/bin/env bash\n")
            f.writelines("#SBATCH --job-name={}\n".format('job{}'.format(i)))
            f.writelines("#SBATCH --account=hkg02\n")
            f.writelines("#SBATCH --nodes=1\n")
            # f.writelines("#SBATCH --partition=large\n")
            # f.writelines("#SBATCH --mem=32000\n")
            f.writelines("#SBATCH --time=0-08:00:00\n")
            f.writelines("#SBATCH --mail-type=ALL\n")
            f.writelines("#SBATCH --mail-user=hkg02@mail.aub.edu\n\n")
            f.writelines("module load python/3\n")
            f.writelines("module load python/tensorflow\n")
            f.writelines("pipenv run python main.py \\\n")
            f.writelines("--pretrain_iterations 0 \\\n")
            f.writelines("--metatrain_iterations {} \\\n".format(combination[0]))
            f.writelines("--meta_batch_size {} \\\n".format(combination[1]))
            f.writelines("--meta_lr {} \\\n".format(combination[2]))
            f.writelines("--update_batch_size {} \\\n".format(combination[3]))
            f.writelines("--update_lr {} \\\n".format(combination[4]))
            f.writelines("--num_updates {} \\\n".format(combination[5]))
            # f.writelines("--supp_fp {} \\\n".format(combination[6]))
            f.writelines("--fp_file '{}' \\\n".format('fake_news_fps/fps_fakenews_{}.pickle'.format(combination[6])))
            f.writelines("--dim_hidden '{}' \\\n".format(', '.join([str(d) for d in combination[7][1]])))
            f.writelines("--dim_name '{}' \\\n".format(combination[7][0]))
            f.writelines("--activation_fn '{}' \\\n".format(combination[8]))
            f.writelines("--cost_sensitive {} \\\n".format(combination[9]))
            f.writelines("--scaling '{}' \\\n".format(combination[10]))
            f.writelines("--logdir '{}' \\\n".format(out_dir_name))
            f.writelines("> out_job{}.txt".format(i))
            f.close()


if __name__ == '__main__':
    metatrain_iterations = [1000]
    meta_batch_sizes = [32]
    meta_lrs = [1e-1]
    update_batch_sizes = [32]
    update_lrs = [1e-1]
    num_updates = [4]
    fp_supports = [0.5, 0.6, 0.7]
    dim_hidden = [[256, 128, 64], [128, 64], [128]]
    dim_names = ['dim{}'.format(i) for i in range(len(dim_hidden))]
    dims = list(zip(dim_names, dim_hidden))
    activation_fns = ['relu', 'sigmoid', 'tanh', 'softmax', 'swish']
    cost_sensitive = [False]
    scaling = ['z-score']
    jobs_dir_name = 'fake_news/'

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
    # cost_sensitive = [False]
    # scaling = ['z-score']
    # jobs_dir_name = 'fake_news/'

    create_scripts(metatrain_iterations, meta_batch_sizes, meta_lrs,
                   update_batch_sizes, update_lrs, num_updates,
                   fp_supports, dims, activation_fns, cost_sensitive,
                   scaling, out_dir_name=jobs_dir_name)
