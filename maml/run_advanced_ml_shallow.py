import pandas as pd
import pickle
import os
from AdvancedEvaluation import AdvancedEvaluator

if __name__ == '__main__':
    df_train = pd.read_csv('feature_extraction_train_updated.csv')
    df_test = pd.read_csv('feature_extraction_test_updated.csv')
    df = pd.concat([df_train, df_test]).sample(frac=1).reset_index(drop=True)
    cols_drop = ['article_title', 'article_content', 'source', 'source_category', 'unit_id']
    trained_models_dir = 'trained_models_shallow/'
    models = ['ada_boost', 'decision_tree', 'extra_trees',
              'logistic_regression', 'random_forest']
    model_objs = []

    # create dictionary for trained models
    for model in models:
        # file_name = os.path.join(trained_models_dir, '{}.p'.format(model))
        file_name = os.path.join(trained_models_dir, '{}.pickle'.format(model))
        trained_model = pickle.load(open(file_name, 'rb'))
        model_objs.append(trained_model)

    models_dict = dict(zip(models, model_objs))

    sm = AdvancedEvaluator(df=df, df_train=df_train, df_test=df_test,
                           target_variable='label',
                           plots_output_folder='plots_shallow/',
                           fp_growth_output_folder='plots_shallow/',
                           fp_file=None,
                           models_dict=models_dict,
                           scaling='z-score',
                           cols_drop=cols_drop,
                           pos_class_label=1)

    # advanced ML classification
    for model in models:
        print('\n~~~~~~~~~~~~~~~~~~~~~~~~ Model: {} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'.format(model))
        sm.classify(trained_model=models_dict[model], trained_model_name=model, nb_bins=10)

    # save error metrics
    sm.save_results()

    # add mistake
    sm.add_mistake()

    # generate probabilities of mistakes per model per frequent pattern
    probabilities_per_fp = sm.pattern_probability_of_mistake()

    # produce roc curves when applying classification using all models
    sm.produce_roc_curves()

    # produce mean empirical risk curves
    sm.produce_empirical_risk_curves()

    # produce precision/recall at topK curves
    sm.produce_curves_topK(topKs=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150], metric='precision')
    sm.produce_curves_topK(topKs=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150], metric='recall')

    # produce jaccard similarity at topK
    sm.compute_jaccard_similarity(topKs=list(range(20, 200, 20)))