import pandas as pd
import pickle
import os
from AdvancedEvaluation import ShallowModel

if __name__ == '__main__':
    training_data_path = 'input/fake_news_datasets/feature_extraction_train_updated.csv'
    testing_data_path = 'input/fake_news_datasets/feature_extraction_test_updated.csv'

    train_df = pd.read_csv(training_data_path, encoding='latin-1')
    test_df = pd.read_csv(testing_data_path, encoding='latin-1')
    df = pd.concat([train_df, test_df]).sample(frac=1).reset_index(drop=True)
    cols_drop = ['article_title', 'article_content', 'source', 'source_category', 'unit_id']
    models = ['ada_boost', 'decision_tree', 'extra_trees',
              'logistic_regression', 'random_forest']

    experiments = ['Experiment1']
    for exp in experiments:
        trained_models_dir = 'input/fake_news_trained_models/{}/'.format(exp)
        model_objs = []
        for model in models:
            path_to_file = os.path.join(trained_models_dir, '{}.sav'.format(model))
            if os.path.exists(path_to_file):
                file_name = path_to_file
            else:
                path_to_file = os.path.join(trained_models_dir, '{}.pickle'.format(model))
                if os.path.exists(path_to_file):
                    file_name = path_to_file
                else:
                    file_name = os.path.join(trained_models_dir, '{}.p'.format(model))
            trained_model = pickle.load(open(file_name, 'rb'))
            model_objs.append(trained_model)

        models_dict = dict(zip(models, model_objs))

        sm = ShallowModel(df=df, df_train=train_df, df_test=test_df,
                          target_variable='label',
                          plots_output_folder='plots/fake_news/{}/'.format(exp),
                          trained_models_dir='input/fake_news_trained_models/{}/'.format(exp),
                          models_dict=models_dict,
                          scaling='z-score',
                          cols_drop=cols_drop,
                          pos_class_label=1)

        # identify frequent patterns in data
        sm.identify_frequent_patterns()

        # advanced ML classification
        for model in models:
            print('\n~~~~~~~~~~~~~~~~~~~~~~~~ Model: {} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'.format(model))
            sm.classify(trained_model=models_dict[model], trained_model_name=model, nb_bins=10)

        # add mistake
        sm.add_mistake()

        # generate probabilities of mistakes per model per frequent pattern
        probabilities_per_fp = sm.pattern_probability_of_mistake()

        # produce roc curves when applying classification using all models
        sm.produce_roc_curves()

        # produce mean empirical risk curves
        sm.produce_empirical_risk_curves()

        # produce precision/recall at topK curves
        sm.produce_curves_topK(topKs=[10, 20, 30, 40, 50, 60], metric='precision')
        sm.produce_curves_topK(topKs=[10, 20, 30, 40, 50, 60], metric='recall')

        # produce jaccard similarity at topK
        sm.compute_jaccard_similarity(topKs=list(range(20, 200, 20)))