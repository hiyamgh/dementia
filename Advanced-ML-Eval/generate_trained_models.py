import pandas as pd
from TrainedModelsGenerator import ModelGenerator
from models_container import shallow_models


def generate_trained_models_toydata(train_df, test_df, target_var, scaling_mech, results_output_folder, trained_models_output_folder, nb_splits=5, nb_repeats=5, cols_to_drop=None):
    mg = ModelGenerator(df_train=train_df, df_test=test_df,
                        target_variable=target_var, scaling=scaling_mech,
                        results_output_folder=results_output_folder,
                        trained_models_output_folder=trained_models_output_folder,
                        nb_splits=nb_splits, nb_repeats=nb_repeats,
                        cols_drop=cols_to_drop)

    for shallow_model in shallow_models:
        print('~~~~~~~~~~~~~~~~~~~~~  {} ~~~~~~~~~~~~~~~~~~~~~'.format(shallow_model))
        # get the model and its name
        model_name = shallow_model
        model = shallow_models[shallow_model]

        # apply cross validation
        mg.cross_validation(model, model_name)

        # apply training and testing + saves trained model
        mg.train_test_model(model, model_name, apply_smote=False)


if __name__ == '__main__':
    train = pd.read_csv('input/df_train.csv')
    test = pd.read_csv('input/df_test.csv')
    generate_trained_models_toydata(train_df=train, test_df=test, target_var='nograd', scaling_mech='robust', results_output_folder='results_toy',
                                    trained_models_output_folder='trained_models_toy',
                                    nb_splits=5, nb_repeats=5, cols_to_drop=None)

    # # Experiment 1 - Fake news -- train on FA-KES, test on FA-KES
    # training_data_path = 'input/fake_news_datasets/feature_extraction_train_updated.csv'
    # testing_data_path = 'input/fake_news_datasets/feature_extraction_test_updated.csv'
    #
    # train_df = pd.read_csv(training_data_path, encoding='latin-1')
    # test_df = pd.read_csv(testing_data_path, encoding='latin-1')
    # cols_drop = ['article_title', 'article_content', 'source', 'source_category', 'unit_id']
    # generate_trained_models_toydata(train_df=train, test_df=test, target_var='nograd', scaling_mech='robust',
    #                                 results_output_folder='results_toy',
    #                                 trained_models_output_folder='trained_models_toy',
    #                                 nb_splits=5, nb_repeats=5, cols_to_drop=None)
    #
    #
    # lm = LearningModelCrossVal(train_df, test_df, output_folder='output/',
    #                            cols_drop=cols_drop, over_sample=False, standard_scaling=True, minmax_scaling=False)

