import pandas as pd
import pickle
import os


def get_hyper_parameters(outfile_name, fp=True):
    hyperparameters = {}

    metatrain_iterations = [1000]
    meta_batch_sizes = [32, 16, 8, 4]
    meta_lrs = [0.1, 0.001]
    update_lrs = [0.1, 0.001]
    dim_hidden = ["128, 64, 64", "128, 64", "128"]
    activation_fns = ["relu", "sigmoid", "tanh", "softmax", "swish"]
    num_updates = [4]
    include_fp = ["1"]
    fp_supps = [0.8, 0.9, 0.7]

    if fp:
        count = 1
        for miter in metatrain_iterations:
            for mbs in meta_batch_sizes:
                for mlr in meta_lrs:
                    for ulr in update_lrs:
                        for dh in dim_hidden:
                            for afn in activation_fns:
                                for nu in num_updates:
                                    for ifp in include_fp:
                                        for fp_supp in fp_supps:
                                            hyperparameters["model_{}".format(count)] = {
                                                'miter': miter,
                                                'mbs': mbs,
                                                'mlr': mlr,
                                                'ulr': ulr,
                                                'dh': dh,
                                                'afn': afn,
                                                'nu': nu,
                                                'ifp': ifp,
                                                'fp_supp': fp_supp
                                            }
                                            count += 1
    else:
        include_fp = ["1"]
        count = 1
        for miter in metatrain_iterations:
            for mbs in meta_batch_sizes:
                for mlr in meta_lrs:
                    for ulr in update_lrs:
                        for dh in dim_hidden:
                            for afn in activation_fns:
                                for nu in num_updates:
                                    for ifp in include_fp:
                                        hyperparameters["model_{}".format(count)] = {
                                            'miter': miter,
                                            'mbs': mbs,
                                            'mlr': mlr,
                                            'ulr': ulr,
                                            'dh': dh,
                                            'afn': afn,
                                            'nu': nu,
                                            'ifp': ifp
                                        }
                                        count += 1

    with open('{}.p'.format(outfile_name), 'wb') as f:
        pickle.dump(hyperparameters, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    get_hyper_parameters(outfile_name='hyperparameters_with_fp', fp=True)
    get_hyper_parameters(outfile_name='hyperparameters_without_fp', fp=False)