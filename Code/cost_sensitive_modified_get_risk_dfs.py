import pandas as pd
import numpy as np
import pickle

model1_file = '../output/output_probabilistic/top_20/f2/james/Balanced Bagging Classifier.sav' # top 20 best model
model2_file = '../output/output_probabilistic/top_10/f2/catboost/Weighted Logistic Regression.sav' # top 10 best model

models_dict = {
    'Balanced Bagging Classifier': model1_file,
    'Weighted Logistic Regression': model2_file
}

df_test = pd.read_csv('../input/test_imputed_scaled.csv')
features_imp = pd.read_csv('../input/feature_importance_modified.csv')

for model_name, model_file in models_dict.items():
    trained_model = pickle.load(open(model_file, 'rb'))

    if 'Bagging' in model_name:
        features = features_imp['Feature'][:20]
    else:
        features = features_imp['Feature'][:10]

    X_test = np.array(df_test[features])
    y_test = list(df_test['dem1066'])

    y_pred = trained_model.predict(X_test)
    y_prob = trained_model.predict_proba(X_test)[:, 1]

    risk_df = pd.DataFrame(np.column_stack((y_test, y_pred, y_prob)), columns=['y_test', 'y_pred', 'risk_scores'])
    risk_df = risk_df.sort_values(by='risk_scores', ascending=False)
    risk_df.to_csv('risk_df_{}.csv'.format(model_name))
