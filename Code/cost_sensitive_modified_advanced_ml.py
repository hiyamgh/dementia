import pandas as pd
import numpy as np
import pickle
from AdvancedEvaluation import AdvancedEvaluator
import os

if __name__ == '__main__':
    optimizations = ['f2', 'gmean', 'bss', 'pr_auc', 'sensitivity', 'specificity']
    tops = [10, 20]
    all_models = {
        'probabilistic': ['Weighted XGBoost', 'Weighted Logistic Regression', 'Weighted Decision Tree Classifier',
                          'Balanced Bagging Classifier'],
        # 'regular': ['Weighted SVM', 'KNeighbors', 'Easy Ensemble Classifier']
    }

    train_df_orig = pd.read_csv('../input/train_imputed_scaled.csv')
    test_df_orig = pd.read_csv('../input/test_imputed_scaled.csv')
    feature_importance = list(pd.read_csv('../input/feature_importance_modified.csv')['Feature'])

    results_folder = '../output/results/'
    for t in tops:
        train_df = train_df_orig[feature_importance[:t] + ['dem1066']]
        test_df = test_df_orig[feature_importance[:t] + ['dem1066']]
        df_data = pd.concat([train_df, test_df])
        with open('../input/columns/categorical.p', 'rb') as f:
            categorical_cols = pickle.load(f)
        all_cols = list(df_data.columns)
        cat_cols = [c for c in categorical_cols if c in all_cols]

        for opt in optimizations:
            for m in all_models:
                models_obj = []
                models_names = []
                winning_encodings = []
                if m == 'probabilistic' or (m == 'regular' and opt == 'f2'):
                    print('\n~~~~~~~~~~~~~~~~~~~~~~~~ Top{} - {} - {} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'.format(t, m, opt))
                    # get the trained models of the winning models
                    df = pd.read_csv(os.path.join(results_folder, 'top_{}_{}_{}.csv'.format(t, m, opt)))
                    for i, row in df.iterrows():
                        model_name = row['model_name']
                        encoding = row['encoding_strategy']
                        # now get the trained model
                        trained_model_file = '../output/output_{}/top_{}/{}/{}/{}.sav'.format(m, t, opt, encoding, model_name)
                        trained_model = pickle.load(open(trained_model_file, 'rb'))

                        models_names.append(model_name)
                        models_obj.append(trained_model)
                        winning_encodings.append(encoding)

                    models_dict = dict(zip(models_names, models_obj))
                    encodings_dict = dict(zip(models_names, winning_encodings))

                    out_folder = os.path.join(results_folder, 'advanced_ml_shallow/top_{}/{}/{}/'.format(t, m, opt))
                    sm = AdvancedEvaluator(df=df_data, df_train=train_df, df_test=test_df,
                                           target_variable='dem1066',
                                           plots_output_folder=out_folder,
                                           fp_growth_output_folder=out_folder,
                                           models_dict=models_dict,
                                           scaling=None,
                                           encodings_dict=encodings_dict,
                                           cat_cols=cat_cols,
                                           cols_drop=None,
                                           pos_class_label=1)

                    # identify frequent patterns in data
                    sm.identify_frequent_patterns()

                    # advanced ML classification
                    for model in models_names:
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
                    sm.compute_jaccard_similarity(topKs=list(range(20, 160, 20)))