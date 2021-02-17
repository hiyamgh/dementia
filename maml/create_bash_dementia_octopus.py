import os, itertools

def create_scripts(metatrain_iterations, meta_batch_sizes, meta_lrs,
                   update_batch_sizes, update_lrs, num_updates,
                   dims, activation_fns,
                   weights, sss, top_features,
                   out_dir_name):

    all_hyper_params = [metatrain_iterations, meta_batch_sizes, meta_lrs, update_batch_sizes,
                        update_lrs, num_updates, dims, activation_fns, weights, sss, top_features]

    all_combinations = list(itertools.product(*all_hyper_params))
    print('length of all combinations: {}'.format(len(all_combinations)))
    with open('jobs.txt', 'w') as f:
        for i, combination in enumerate(all_combinations):
            f.writelines("python main.py  --pretrain_iterations 0 --metatrain_iterations {} --meta_batch_size {} --meta_lr {} --update_batch_size {} --update_lr {} --num_updates {} --dim_hidden \"{}\" --dim_name \"{}\" --activation_fn \"{}\" --weights_vector \"{}\" --sampling_strategy \"{}\" --top_features {} --logdir \"{}\" > out_job{}.txt\n".format(
                combination[0], combination[1], combination[2], combination[3], combination[4], combination[5], ', '.join([str(d) for d in combination[6][1]]), combination[6][0],  combination[7], ', '.join(str(d) for d in combination[8]), combination[9], combination[10], out_dir_name, i))
        f.close()


if __name__ == '__main__':
    metatrain_iterations = [1000]
    meta_batch_sizes = [16]
    meta_lrs = [1e-1]
    update_batch_sizes = [16]
    update_lrs = [1e-1]
    num_updates = [4]
    # fp_supports = [0.7]
    dim_hidden = [[128, 64, 64]]
    dim_names = ['dim{}'.format(i) for i in range(len(dim_hidden))]
    dims = list(zip(dim_names, dim_hidden))
    activation_fns = ['relu']
    cost_sensitive = [True]
    weights = [[1, 1], [1, 10], [1, 100], [100, 1], [10, 1]]
    sampling_strategies = ['minority', 'not minority', 'all', 0.5, 1, 0.75]
    top_features = [10, 20]
    scaling = [None]
    jobs_dir_name = 'dementia_relu_3_fp/'

    create_scripts(metatrain_iterations, meta_batch_sizes, meta_lrs,
                   update_batch_sizes, update_lrs, num_updates,
                   dims, activation_fns,
                   weights, sampling_strategies, top_features,
                   out_dir_name=jobs_dir_name)
