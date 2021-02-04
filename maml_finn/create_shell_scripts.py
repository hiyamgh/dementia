import os
import itertools

def create_scripts(metatrain_iterations,
                   meta_batch_sizes,
                   meta_lrs,
                   update_batch_sizes,
                   update_lrs,
                   num_updates,
                   fp_supports):

    all_hyper_params = [metatrain_iterations, meta_batch_sizes, meta_lrs, update_batch_sizes,
                        update_lrs, num_updates]
    all_combinations = list(itertools.product(*all_hyper_params))
    print('length of all combinations: {}'.format(len(all_combinations)))
    for i, combination in enumerate(all_combinations):
        with open('jobfp{}.sh'.format(i), 'w') as f:
            f.writelines("#!/usr/bin/env bash\n")
            f.writelines("python main.py \\\n")
            f.writelines("--pretrain_iterations 0 \\\n")
            f.writelines("--metatrain_iterations {} \\\n".format(combination[0]))
            f.writelines("--meta_batch_size {} \\\n".format(combination[1]))
            f.writelines("--meta_lr {} \\\n".format(combination[2]))
            f.writelines("--update_batch_size {} \\\n".format(combination[3]))
            f.writelines("--update_lr {} \\\n".format(combination[4]))
            f.writelines("--num_updates {} \\\n".format(combination[5]))
            f.writelines("--supp_fp {} \\\n")
            f.writelines("--logdir 'jobs' \\\n")
            f.writelines("> jobfp{}.txt".format(i))
            f.close()


if __name__ == '__main__':
    metatrain_iterations = [1000]
    meta_batch_sizes = [32]
    meta_lrs = [1e-3]
    update_batch_sizes = [32]
    update_lrs = [1e-3]
    num_updates = [4]
    fp_supports = [0.7, 0.8, 0.9]

    # metatrain_iterations = [1000]
    # meta_batch_sizes = [4]
    # meta_lrs = [1e-3]
    # update_batch_sizes = [8]
    # update_lrs = [1e-3]
    # num_updates = [4]

    create_scripts(metatrain_iterations, meta_batch_sizes, meta_lrs,
                   update_batch_sizes, update_lrs, num_updates,
                   fp_supports)

    # metatrain_iterations = [1000]
    # meta_batch_sizes = [4, 8]
    # # meta_lrs = [1e-3, 1e-1]
    # meta_lrs = [1e-1]
    # update_batch_sizes = [8, 16]
    # # update_lrs = [1e-3, 1e-1]
    # update_lrs = [1e-1]
    # num_updates = [4]
    # # metatrain_iterations = [1000]
    # # meta_batch_sizes = [4]
    # # meta_lrs = [1e-3]
    # # update_batch_sizes = [8]
    # # update_lrs = [1e-3]
    # # num_updates = [4]
    #
    # create_scripts(metatrain_iterations, meta_batch_sizes, meta_lrs,
    #                update_batch_sizes, update_lrs, num_updates)