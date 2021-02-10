import os
import itertools


def create_scripts(metatrain_iterations, meta_batch_sizes, meta_lrs,
                   update_batch_sizes, update_lrs, num_updates,
                   fp_supports, dims, activation_fns, cost_sensitive,
                   weights, sss, top_features,
                   scaling,
                   out_dir_name):
    # weights = [[1, 10], [1, 100], [1, 1]]
    #     sampling_strategies = ['minority', 'not minority', 'all', 0.5, 1, 0.75]
    #     top_features = [10, 20]

    all_hyper_params = [metatrain_iterations, meta_batch_sizes, meta_lrs, update_batch_sizes,
                        update_lrs, num_updates, fp_supports, dims, activation_fns, cost_sensitive, scaling]

    all_combinations = list(itertools.product(*all_hyper_params))
    print('length of all combinations: {}'.format(len(all_combinations)))
    for i, combination in enumerate(all_combinations):
        with open('job{}.sh'.format(i), 'w') as f:
            f.writelines("#!/usr/bin/env bash\n")
            f.writelines("python main.py \\\n")
            f.writelines("--pretrain_iterations 0 \\\n")
            f.writelines("--metatrain_iterations {} \\\n".format(combination[0]))
            f.writelines("--meta_batch_size {} \\\n".format(combination[1]))
            f.writelines("--meta_lr {} \\\n".format(combination[2]))
            f.writelines("--update_batch_size {} \\\n".format(combination[3]))
            f.writelines("--update_lr {} \\\n".format(combination[4]))
            f.writelines("--num_updates {} \\\n".format(combination[5]))
            f.writelines("--fp_file '{}' \\\n".format('fake_news_fps/fps_fakenews_{}.pickle'.format(combination[6])))
            f.writelines("--dim_hidden '{}' \\\n".format(', '.join([str(d) for d in combination[7][1]])))
            f.writelines("--dim_name '{}' \\\n".format(combination[7][0]))
            f.writelines("--activation_fn '{}' \\\n".format(combination[8]))
            f.writelines("--cost_sensitive {} \\\n".format(combination[9]))
            f.writelines("--weights_vector {} \\\n".format(combination[10]))
            f.writelines("--sampling_strategy '{}' \\\n".format(combination[11]))
            f.writelines("--scaling '{}' \\\n".format(combination[12]))
            f.writelines("--logdir '{}' \\\n".format(out_dir_name))
            f.writelines("> out_job{}.txt".format(i))
            f.close()


if __name__ == '__main__':
    metatrain_iterations = [1000]
    meta_batch_sizes = [16]
    meta_lrs = [1e-1]
    update_batch_sizes = [16]
    update_lrs = [1e-1]
    num_updates = [4]
    fp_supports = [0.7]
    dim_hidden = [[256, 128, 64]]
    dim_names = ['dim{}'.format(i) for i in range(len(dim_hidden))]
    dims = list(zip(dim_names, dim_hidden))
    activation_fns = ['relu']
    cost_sensitive = [True]
    weights = [[1, 10], [1, 100], [1, 1]]
    sampling_strategies = ['minority', 'not minority', 'all', 0.5, 1, 0.75]
    top_features = [10, 20]
    scaling = [None]
    jobs_dir_name = 'dementia/'


    create_scripts(metatrain_iterations, meta_batch_sizes, meta_lrs,
                   update_batch_sizes, update_lrs, num_updates,
                   fp_supports, dims, activation_fns, cost_sensitive,
                   weights, sampling_strategies, top_features,
                   scaling, out_dir_name=jobs_dir_name)
