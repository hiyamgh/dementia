from collections import Counter
from numpy import mean
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC
from data_preprocessing import get_columns
from sklearn import metrics
from imblearn.metrics import geometric_mean_score
# from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import fbeta_score,make_scorer

f2_score=make_scorer(fbeta_score, beta=2)

def brier_skill_score(y, yhat):
    probabilities = [0.01 for _ in range(len(y))]
    brier_ref=metrics.brier_score_loss(y, probabilities)
    bs=metrics.brier_score_loss(y, yhat)
    return 1.0 - (bs / brier_ref)

def train_validate_test():
    split_save('../input/pooled_imputed_scaled.csv','../input/train.csv','../input/test.csv')
    
def split_save(full_path,train_path,test_path):
    df=pd.read_csv(full_path)
    y=np.array(df['dem1066'])
    X=df[df.columns[:-1]]
    split(X,y,train_path,test_path)
    split(X,y,train_path,test_path)

def split(X,y,train_path,test_path):
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2,random_state=36851234)
    for train_index, test_index in rskf.split(X, y):
        X_train, X_test = X.iloc[list(train_index)], X.iloc[list(test_index)]
        y_train, y_test = y[train_index], y[test_index]
    
    X_train['dem1066']=y_train
    X_test['dem1066']=y_test
    X_train.to_csv(train_path)
    X_test.to_csv(test_path)

def get_categorical(columns):
    codebook=pd.read_csv('../input/codebooks/erroneous_codebook_legal_outliers_filtered.csv')
    _,_,categorical=get_columns(codebook)
    categorical_index=[]
    for col in categorical:
        index=np.where(columns == col)
        categorical_index.append(int(index[0][0]))
    return categorical_index

def create_pipeline(model,columns):
    categorical_index=get_categorical(columns)
    steps = [('s', SMOTENC(categorical_index)), ('m', model)]
    pipeline = Pipeline(steps=steps)
    return pipeline

def baseline_logistic(X,y,test_X,test_y):
    model = LogisticRegression()
    model.fit(X,y)
    y_predicted=model.predict(test_X)
    print_results('baseline logistic regression',test_y,y_predicted)

def create_model(y,model):
    counter = Counter(y)
    reverse_counter={0:counter[1],1:counter[0]}
    balance = [{0:1,1:10}, {0:1,1:100},reverse_counter]
    param_grid = {'m__class_weight':[reverse_counter],'s__sampling_strategy':['minority','not minority','all']}
    # param_grid = {}
    
                    #   solver=['newton-cg', 'lbfgs', 'liblinear','sag', 'saga'],
                    #   C=[0.01,0.1,1,10,100]
                    #   penalty=['l1', 'l2', 'elasticnet', 'none']
                    
    return model,param_grid

def print_results(model_name,best_params,y,y_predicted):
    roc=metrics.roc_auc_score(y,y_predicted)
    gmean=geometric_mean_score(y, y_predicted, average='weighted')
    fscore=fbeta_score(y, y_predicted, beta=2)
    bss=brier_skill_score(y,y_predicted)
    pr_auc=metrics.average_precision_score(y, y_predicted)
    tn, fp, fn, tp = metrics.confusion_matrix(y,y_predicted).ravel()
    f=open(f"../output/cost_sensitive_results/{model_name}.txt", "w")
    f.write('%s Results:\n'%model_name)
    f.write('%s ROC AUC: %.3f\n' % ('',mean(roc)))
    f.write('%s GMEAN: %.3f\n' % ('',gmean))
    f.write('%s f-score: %.3f\n' % ('',fscore))
    f.write('%s bss: %.3f\n' % ('',bss))
    f.write('%s PR AUC: %.3f\n' % ('',pr_auc))
    f.write(f'best params:{best_params}')
    f.write('\nCost Matrix (assuming cost = 0 for correct labels and cost = 1 for wrong labels)\n')
    f.write('\t\t  |Actual Negative|Actual Positive\n')
    f.write(f'Predicted Negative|0\t\t  |{fn}\n')
    f.write(f'Predicted Positive|{fp}\t\t  |0\n')
    f.close()
    
def test_model(X,y,model_class,model_name,test_X,test_y):
    model,grid=create_model(y,model_class)
    pipeline=create_pipeline(model,X.columns)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid = GridSearchCV(estimator=pipeline,param_grid=grid,cv=cv,scoring=f2_score)
    grid_result = grid.fit(X, y)
    # print('Best Weighted logistic regression: %f using %s' % (grid_result.best_score_, grid_result.best_params_))
    estimator=grid_result.best_estimator_
    estimator.fit(X,y)
    y_predicted=estimator.predict(test_X)
    print_results(model_name,grid_result.best_params_,test_y,y_predicted)
    
if __name__=='__main__':
    df=pd.read_csv('../input/train.csv')
    X=df[df.columns[:-1]]
    y=df['dem1066']
    
    test_df=pd.read_csv('../input/test.csv')
    test_X=test_df[df.columns[:-1]]
    test_y=test_df['dem1066']
    
    # baseline_logistic(X,y,test_X,test_y)
    for model,model_name in [
        # (XGBClassifier(),'XGBoost'),
                            # (KNeighborsClassifier(),'KNeighbors'),
                            # (BalancedRandomForestClassifier(),'Balanced Random Forest')
                             (LogisticRegression(),'Weighted Logistic Regression'),
                            #  (DecisionTreeClassifier(),'Weighted Decision Tree Classifier'),
                            #  (SVC(),'Weighted SVM')
                             ]:
        test_model(X,y,model,model_name,test_X,test_y)