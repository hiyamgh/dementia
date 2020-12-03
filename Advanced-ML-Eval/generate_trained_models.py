import pandas as pd
from TrainedModelsGenerator import ModelGenerator
from models_container import shallow_models


if __name__ == '__main__':
    train = pd.read_csv('input/df_train.csv')
    test = pd.read_csv('input/df_test.csv')
    mg = ModelGenerator(df_train=train, df_test=test,
                        target_variable='nograd', scaling='robust',
                        results_output_folder='results',
                        trained_models_output_folder='trained_models',
                        nb_splits=5, nb_repeats=5,
                        cols_drop=None)

    for shallow_model in shallow_models:
        print('~~~~~~~~~~~~~~~~~~~~~  {} ~~~~~~~~~~~~~~~~~~~~~'.format(shallow_model))
        # get the model and its name
        model_name = shallow_model
        model = shallow_models[shallow_model]

        # apply cross validation
        mg.cross_validation(model, model_name)

        # apply training and testing + saves trained model
        mg.train_test_model(model, model_name, apply_smote=False)