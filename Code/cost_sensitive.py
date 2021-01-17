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
from imblearn.ensemble import BalancedRandomForestClassifier,EasyEnsembleClassifier
from sklearn.metrics import fbeta_score,make_scorer
from AdvancedEvaluation import AdvancedEvaluator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

f2_score=make_scorer(fbeta_score, beta=2)

# make a prediction with a lof model
def lof_predict(model, trainX, testX):
	# create one large dataset
	composite = np.vstack((trainX, testX))
	# make prediction on composite dataset
	yhat = model.fit_predict(composite)
	# return just the predictions on the test set
	return yhat[len(trainX):]

def to_labels(pos_probs, threshold):
	return (pos_probs >= threshold).astype('int')

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
        if len(index)>0 and len(index[0])>0:
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
    print_results('baseline logistic regression','',test_y,y_predicted)

def create_model(y,model):
    counter = Counter(y)
    reverse_counter={0:counter[1],1:counter[0]}
    balance = [{0:1,1:10}, {0:1,1:100},reverse_counter]
    param_grid = {
        # 'm__class_weight':balance,
                  's__sampling_strategy':['minority','not minority','all',0.5,1,0.75],
                    #   solver=['newton-cg', 'lbfgs', 'liblinear','sag', 'saga'],
                    #   C=[0.01,0.1,1,10,100]
                    #   penalty=['l1', 'l2', 'elasticnet', 'none']              
                  }
    # param_grid={'m__class_weight':[reverse_counter],'s__sampling_strategy':['minority']}
    return model,param_grid

def print_results(model_name,best_params,y,y_predicted,proba=False,one_class=False):

    f=open(f"../output/cost_sensitive_results/{model_name}.txt", "w")

    if one_class:
        fscore=fbeta_score(y, y_predicted, beta=2, pos_label=-1)
    else:
        fscore=fbeta_score(y, y_predicted, beta=2)
        
    f.write('%s Results:\n'%model_name)
    f.write('%s f-score: %.3f\n' % ('',fscore))

    if one_class:
        f.close()
        return
    
    if proba:
        gmean=geometric_mean_score(y, y_predicted, average='weighted')
        f.write('%s GMEAN: %.3f\n' % ('',gmean))
        bss=brier_skill_score(y,y_predicted)
        f.write('%s bss: %.3f\n' % ('',bss))
        pr_auc=metrics.average_precision_score(y, y_predicted)
        f.write('%s PR AUC: %.3f\n' % ('',pr_auc))
    
    roc=metrics.roc_auc_score(y,y_predicted)
    f.write('%s ROC AUC: %.3f\n' % ('',mean(roc)))

    tn, fp, fn, tp = metrics.confusion_matrix(y,y_predicted).ravel()
    f.write(f'best params:{best_params}')
    f.write('\nCost Matrix:\n')
    f.write('\t\t  |Actual Negative|Actual Positive\n')
    f.write(f'Predicted Negative|{tn}\t\t  |{fn}\n')
    f.write(f'Predicted Positive|{fp}\t\t  |{tp}\n')
    f.close()

def advanced_metrics(train_df,test_df,trained_model,trained_model_name):
    
    df = pd.concat([train_df, test_df]).sample(frac=1).reset_index(drop=True)
    sm = AdvancedEvaluator(df=df, 
                           df_train=train_df, 
                           df_test=test_df,
                           target_variable='dem1066',
                           plots_output_folder='../output/cost_sensitive_results/advanced_plots/',
                           fp_growth_output_folder='../output/cost_sensitive_results/fp_growth/',
                           models_dict=models_dict,
                           scaling='z-score',
                           cols_drop=None,
                           pos_class_label=1)

    print('frequent patterns')
    sm.identify_frequent_patterns()
    print('classfiy')
    sm.classify(trained_model=trained_model, trained_model_name=trained_model_name, nb_bins=10)
    print('add mistake')
    sm.add_mistake()
    print('proba per fp')
    probabilities_per_fp = sm.pattern_probability_of_mistake()
    print('ROC')
    sm.produce_roc_curves()
    print('produce_empirical_risk_curves')
    sm.produce_empirical_risk_curves()
    print('produce_curves_topK 1')
    sm.produce_curves_topK(topKs=[10, 20, 30, 40, 50, 60], metric='precision')
    print('produce_curves_topK 2')
    sm.produce_curves_topK(topKs=[10, 20, 30, 40, 50, 60], metric='recall')
    print('compute_jaccard_similarity')
    sm.compute_jaccard_similarity(topKs=list(range(20, 200, 20)))
            
def test_model(train_df,test_df,feature_importance,model_class,model_name,proba=False,one_class=False):
    if one_class:
        train_df=train_df[train_df['dem1066']==0]

    X=train_df[df.columns[:-1]]
    # X=train_df[feature_importance[:11]]
    y=train_df['dem1066']
    test_X=test_df[df.columns[:-1]]
    test_y=test_df['dem1066']
    if one_class:    
        test_y[test_y == 1] = -1
        test_y[test_y == 0] = 1
    
    model,param_grid=create_model(y,model_class)
    pipeline=create_pipeline(model,X.columns)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    if not one_class:
        grid = GridSearchCV(estimator=pipeline,param_grid=param_grid,cv=cv,scoring=f2_score)
        grid_result = grid.fit(X,y)
        print('Best model: %f using %s' % (grid_result.best_score_, grid_result.best_params_))
        estimator=grid_result.best_estimator_
        best_params=grid_result.best_params_
    else:
        estimator=model
        best_params={}
        
    estimator.fit(X,y)
    if proba:
        yhat = estimator.predict_proba(test_X)
        probs = yhat[:, 1]
        # define thresholds
        thresholds = np.arange(0, 1, 0.001)
        # evaluate each threshold
        scores = [fbeta_score(test_y, to_labels(probs, t), beta=2) for t in thresholds]
        # get best threshold
        ix = np.argmax(scores)
        print('Threshold=%.3f, F-measure=%.5f' % (thresholds[ix], scores[ix]))
        y_predicted=to_labels(probs,thresholds[ix])
        best_params['f2_score']=scores[ix]
        best_params['threshold']=thresholds[ix]
    else:
        y_predicted=estimator.predict(test_X)
        # y_predicted=lof_predict(estimator, X, test_X)
    print('printing results')
    print_results(model_name,best_params,test_y,y_predicted,proba=proba,one_class=one_class)
    return estimator
    
if __name__=='__main__':
    
    df=pd.read_csv('../input/train.csv')
    test_df=pd.read_csv('../input/test.csv')
    feature_importance=np.array(pd.read_csv('../input/feature_importance.csv')['Feature'])
    # print(feature_importance)
    # baseline_logistic(X,y,test_X,test_y)
    models_dict={}
    for model,model_name,proba,one_class in [
        # (XGBClassifier(),'XGBoost'),
                            (CalibratedClassifierCV(LogisticRegression(class_weight={0: 162, 1: 640})),'Weighted Logistic Regression',True,False),
                            (CalibratedClassifierCV(DecisionTreeClassifier(class_weight={0: 162, 1: 640})),'Weighted Decision Tree Classifier',True,False),
                            (CalibratedClassifierCV(SVC(class_weight={0: 162, 1: 640})),'Weighted SVM',False,False),
                            (CalibratedClassifierCV(KNeighborsClassifier()),'KNeighbors',False,False),
                            (BalancedRandomForestClassifier(),'Balanced Random Forest',True,False),
                            (EasyEnsembleClassifier(),'Easy Ensemble Classifier',False,False),
                            #  (OneClassSVM(gamma='scale', nu=0.1),'One Class SVM',False,True),
                            #  (EllipticEnvelope(contamination=0.1),'EllipticEnvelope',False,True),
                            #  (IsolationForest(contamination=0.1),'IsolationForest',False,True),
                            #  (LocalOutlierFactor(contamination=0.1),'LocalOutlierFactor',False,True)
                            ]:
        print(model_name)
        estimator=test_model(df,test_df,feature_importance,model,model_name,proba=proba,one_class=one_class)
        # models_dict[model_name]=estimator
    
    # for model_name,estimator in models_dict.items():
    #     advanced_metrics(df,test_df,estimator,model_name)