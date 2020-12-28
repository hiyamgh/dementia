import pandas as pd
import pickle
import os
from AdvancedEvaluation import ShallowModel

if __name__ == '__main__':
    df = pd.read_csv('input/toy_data/simulated_data.csv')
    df = df.drop(['id'], axis=1)
    train = pd.read_csv('input/toy_data/df_train.csv')
    test = pd.read_csv('input/toy_data/df_test.csv')

    from models_container import shallow_models

    sm = ShallowModel(df=df, df_train=train, df_test=test,
                      target_variable='nograd',
                      plots_output_folder='plots/toy/',
                      trained_models_dir='trained_models',
                      models_dict=shallow_models,
                      scaling='robust',
                      cols_drop=None)

    # identify frequent patterns in data
    sm.identify_frequent_patterns()

    for shallow_model in shallow_models:
        print('\n~~~~~~~~~~~~~~~~~~~~~~~~ Model: {} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'.format(shallow_model))
        # load the model from disk
        file_name = os.path.join(sm.trained_models_dir, '{}.sav'.format(shallow_model))
        trained_model = pickle.load(open(file_name, 'rb'))
        sm.classify(trained_model=trained_model, trained_model_name=shallow_model)

    # add mistake
    sm.add_mistake()
    #
    # generate probabilities of mistakes per model per frequent pattern
    probabilities_per_fp = sm.pattern_probability_of_mistake()

    # produce roc curves when applying classification using all models
    sm.produce_roc_curves()

    # produce mean empirical risk curves
    sm.produce_empirical_risk_curves()

    # produce precision/recall at topK curves
    sm.produce_curves_topK(topKs=[60, 50, 40, 30, 20, 10], metric='precision')
    sm.produce_curves_topK(topKs=[60, 50, 40, 30, 20, 10], metric='recall')

    # produce jaccard similarity at topK
    sm.compute_jaccard_similarity(topKs=list(range(20, 200, 20)))


