# weighted logistic regression model on an imbalanced classification dataset
from collections import Counter
from numpy import mean
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

def baseline_logistic(X,y):
    model = LogisticRegression()
    scores = cross_val_score(model, X, y, scoring='roc_auc')
    print('Baseline ROC AUC: %.3f' % mean(scores))
    return model

def weighted_logistic(X,y):
    model = LogisticRegression(max_iter=1000,C=0.01)
    counter = Counter(y)
    reverse_counter={0:counter[1],1:counter[0]}
    balance = [{0:1,1:10}, {0:1,1:100},reverse_counter]
    param_grid = dict(class_weight=balance,
                    #   solver=['newton-cg', 'lbfgs', 'liblinear','sag', 'saga'],
                    #   C=[0.01,0.1,1,10,100]
                    #   penalty=['l1', 'l2', 'elasticnet', 'none']
                      )
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid = GridSearchCV(estimator=model,param_grid=param_grid,cv=cv,scoring='roc_auc')
    grid_result = grid.fit(X, y)
    print('Best Weighted logistic regression: %f using %s' % (grid_result.best_score_, grid_result.best_params_))
    return model

def test_model():
    df=pd.read_csv('../input/pooled_imputed_scaled_no_na.csv')
    y=np.array(df['dem1066'])
    X=df[df.columns[:-1]]
    # baseline_logistic(X,y)
    # weighted_logistic(X,y)

if __name__=='__main__':
    test_model()