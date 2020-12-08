import pandas as pd
import pickle
import os
from AdvancedEvaluation import ShallowModel


def removekey(d, key):
    r = dict(d)
    del r[key]
    return r


def apply_analysis(df, train_df, test_df, models_passed, models_dictionary, trained_models_dir,
                   cols_to_drop, nb_bins_passed, testing, pos_class_label=1):
    sm = ShallowModel(df=df, df_train=train_df, df_test=test_df,
                       target_variable='label',
                       plots_output_folder='plots/fake_news/{}_test_{}/'.format(exp, testing),
                       trained_models_dir=trained_models_dir,
                       models_dict=models_dictionary,
                       scaling='z-score',
                       cols_drop=cols_to_drop,
                      pos_class_label=pos_class_label)

    # identify frequent patterns in data
    # sm.identify_frequent_patterns()

    for model in models_passed:
        print('\n~~~~~~~~~~~~~~~~~~~~~~~~ Model: {} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'.format(model))
        sm.classify(trained_model=models_dictionary[model], trained_model_name=model, nb_bins=nb_bins_passed)

    # add mistake
    # sm.add_mistake()

    # generate probabilities of mistakes per model per frequent pattern
    # probabilities_per_fp = sm.pattern_probability_of_mistake()

    # produce roc curves when applying classification using all models
    sm.produce_roc_curves()

    # produce mean empirical risk curves
    sm.produce_empirical_risk_curves()

    # produce precision/recall at topK curves
    sm.produce_curves_topK(topKs=[60, 50, 40, 30, 20, 10], metric='precision')
    sm.produce_curves_topK(topKs=[60, 50, 40, 30, 20, 10], metric='recall')

    # produce jaccard similarity at topK
    sm.compute_jaccard_similarity(topKs=list(range(20, 200, 20)))


if __name__ == '__main__':

    # training and testing data paths for BuzzFeed datasets
    training_data_path = 'input/fake_news_datasets/buzzfeed_feature_extraction_train_80_updated.csv'
    testing_data_path = 'input/fake_news_datasets/buzzfeed_feature_extraction_test_20_updated.csv'

    # BuzzFeed train and test datasets
    buzz_train = pd.read_csv(training_data_path, encoding='latin-1')
    buzz_test = pd.read_csv(testing_data_path, encoding='latin-1')

    df_buzzbuzz = pd.concat([buzz_train, buzz_test]).sample(frac=1).reset_index(drop=True)

    # just for later use
    buzz_train_orig = buzz_train

    # drop column
    cols_drop = ['article_content']

    # FA-KES test dataset (with dropping un-needed columns)
    fakes_test = pd.read_csv('input/fake_news_datasets/feature_extraction_test_updated.csv')
    fakes_test = fakes_test.drop(['article_title', 'article_content', 'source', 'source_category', 'unit_id'], axis=1)

    df_buzzfakes = pd.concat([buzz_train, fakes_test]).sample(frac=1).reset_index(drop=True)

    # models trained on buzzfeed
    exp = 'Experiment3'
    models = ['ada_boost', 'decision_tree', 'extra_trees',
              'logistic_regression', 'random_forest']
    trained_models_dir = 'input/fake_news_trained_models/{}/'.format(exp)
    model_objs, model_objs2 = [], []
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

    models2 = [model for model in models if model != 'random_forest']

    for model in models2:
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
        model_objs2.append(trained_model)

    models_dict2 = dict(zip(models2, model_objs2))

    # test on buzz feed
    apply_analysis(df=df_buzzbuzz, train_df=buzz_train, test_df=buzz_test,
                   models_passed=models, models_dictionary=models_dict, trained_models_dir=trained_models_dir,
                   cols_to_drop=cols_drop, nb_bins_passed=3, testing='buzzfeed', pos_class_label=0)

    # test on fakes
    apply_analysis(df=df_buzzfakes, train_df=buzz_train.drop(['article_content'], axis=1), test_df=fakes_test,
                   models_passed=models2, models_dictionary=models_dict2, trained_models_dir=trained_models_dir,
                   cols_to_drop=None, nb_bins_passed=2, testing='fakes', pos_class_label=0)

















