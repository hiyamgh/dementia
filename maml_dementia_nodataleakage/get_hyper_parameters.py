import pickle


def get_hyper_parameters(outfile_name, fp=True):
    hyperparameters = {}

    metatrain_iterations = [1000]
    meta_batch_sizes = [16]
    meta_lrs = [0.1]
    update_lrs = [0.1]
    dim_hidden = ["128, 64", "128", "128, 64, 64"]
    activation_fns = ["relu"]
    num_updates = [4]
    include_fp = ["1"]
    fp_supps = [0.8, 0.9, 0.7]
    weights= ["1, 1", "1, 10", "1, 100", "10, 1", "100, 1"]
    sampling_strategy = ["minority", "not minority", "all", "0.5", "1", "0.75"]
    encoding = ["catboost", "glmm", "target", "mestimator", "james", "woe"]

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
                                            for wt in weights:
                                                for ss in sampling_strategy:
                                                    for e in encoding:
                                                        hyperparameters["model_{}".format(count)] = {
                                                            'miter': miter,
                                                            'mbs': mbs,
                                                            'mlr': mlr,
                                                            'ulr': ulr,
                                                            'dh': dh,
                                                            'afn': afn,
                                                            'nu': nu,
                                                            'ifp': ifp,
                                                            'fp_supp': fp_supp,
                                                            'weights': wt,
                                                            'sampling_strategy': ss,
                                                            'encoding': e
                                                        }
                                                        count += 1
    else:
        include_fp = ["0"]
        count = 1
        for miter in metatrain_iterations:
            for mbs in meta_batch_sizes:
                for mlr in meta_lrs:
                    for ulr in update_lrs:
                        for dh in dim_hidden:
                            for afn in activation_fns:
                                for nu in num_updates:
                                    for ifp in include_fp:
                                        for wt in weights:
                                            for ss in sampling_strategy:
                                                for e in encoding:
                                                    hyperparameters["model_{}".format(count)] = {
                                                        'miter': miter,
                                                        'mbs': mbs,
                                                        'mlr': mlr,
                                                        'ulr': ulr,
                                                        'dh': dh,
                                                        'afn': afn,
                                                        'nu': nu,
                                                        'ifp': ifp,
                                                        'weights': wt,
                                                        'sampling_strategy': ss,
                                                        'encoding': e
                                                    }
                                                    count += 1
    print('number of models: {}'.format(count))
    with open('{}.p'.format(outfile_name), 'wb') as f:
        pickle.dump(hyperparameters, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    get_hyper_parameters(outfile_name='hyperparameters_with_fp', fp=True)
    get_hyper_parameters(outfile_name='hyperparameters_without_fp', fp=False)