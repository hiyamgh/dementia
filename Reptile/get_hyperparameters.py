import pickle

if __name__ == '__main__':

    shots = [10, 20, 25]
    inner_batch = [5, 10, 15]
    inner_iters = [10, 20, 30]
    learning_rate = [0.001, 0.0005]
    meta_step = [1.0, 0.7, 0.5, 0.25]
    meta_step_final = [1.0, 0.7, 0.5, 0.25]
    meta_batch = [4, 8, 16, 32]
    meta_iters = [1000, 5000, 10000]
    eval_batch = [5, 10, 15]
    eval_iters = [10, 20, 30]
    dim_hidden = ["128, 64", "128", "128, 64, 64"]
    activation_fns = ["relu", "sigmoid", "tanh", "softmax", "swish"]
    weights = ["1, 1", "1, 10", "1, 100", "10, 1", "100, 1"]
    encoding = ["catboost", "glmm", "target", "mestimator", "james", "woe"]
    top_features = [10, 20]

    count = 1
    idx2hyps = {}
    for shts in shots:
        for ib in inner_batch:
            for ii in inner_iters:
                for lr in learning_rate:
                    for ms in meta_step:
                        for msf in meta_step_final:
                            for mb in meta_batch:
                                for mi in meta_iters:
                                    for eb in eval_batch:
                                        for ei in eval_iters:
                                            for dh in dim_hidden:
                                                for af in activation_fns:
                                                    for w in weights:
                                                        for e in encoding:
                                                            for tf in top_features:
                                                                if count <= 40000:
                                                                    idx2hyps[count] = {
                                                                        'shots': shts,
                                                                        'inner_batch': ib,
                                                                        'inner_iters': ii,
                                                                        'lr': lr,
                                                                        'meta_step': ms,
                                                                        'meta_step_final': msf,
                                                                        'meta_batch': mb,
                                                                        'meta_iters': mi,
                                                                        'eval_batch': eb,
                                                                        'eval_iters': ei,
                                                                        'dim_hidden': dh,
                                                                        'activation_fns': af,
                                                                        'weights': w,
                                                                        'encoding': e,
                                                                        'top_features': tf
                                                                    }

                                                                    count += 1
                                                                else:
                                                                    with open('idx2hyps_{}.pkl'.format(count), 'wb') as f:
                                                                        pickle.dump(idx2hyps, f, pickle.HIGHEST_PROTOCOL)