import pandas as pd
import os
from AdvancedEvaluateClassification import ShallowModel

if __name__ == '__main__':
    training_data_path = 'input/fake_news_datasets/feature_extraction_train_updated.csv'
    testing_data_path = 'input/fake_news_datasets/feature_extraction_test_updated.csv'

    train_df = pd.read_csv(training_data_path, encoding='latin-1')
    test_df = pd.read_csv(testing_data_path, encoding='latin-1')
    df = pd.concat([train_df, test_df], axis=`)
    cols_drop = ['article_title', 'article_content', 'source', 'source_category', 'unit_id']

    sm = ShallowModel(df=df, df_train=train, df_test=test,
                      target_variable='nograd',
                      plots_output_folder='plots', trained_models_dir='trained_models',
                      scaling='robust')

    # identify frequent patterns in data
    sm.identify_frequent_patterns()

    # for model_name in clfs:
    #     print('\n~~~~~~~~~~~~~~~~~~~~~~~~ Model: {} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'.format(model_name))
    #     # print('Without SMOTE')
    #     # sm.classify(model=clfs[model_name], model_name=model_name, applySmote=False)
    #     print('\nWith SMOTE')
    #     sm.classify(model=clfs[model_name], model_name=model_name, applySmote=True, nb_bins=8)

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