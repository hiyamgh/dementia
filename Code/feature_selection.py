from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE,RFECV
import numpy as np
import pandas as pd


def feature_importance(pooled):
    # Y_arr=np.array(pooled['dem1066'])
    Y_arr = list(pooled['dem1066'])
    # why tf do we have missing values in the target variable ??????????????????????
    Y_arr = [x if x != ' ' else '0' for x in Y_arr]
    Y_arr = [int(i) for i in Y_arr]
    # X_arr=pooled[pooled.columns[:-1]]
    X_arr = pooled.loc[:, pooled.columns != 'dem1066']
    tree=DecisionTreeRegressor().fit(X_arr,Y_arr)
    all_cols = list(pooled.columns)
    all_cols.remove('dem1066')
    feature_impotances = list(tree.feature_importances_)
    # importance=pd.DataFrame()
    importance = pd.DataFrame({'Feature': all_cols, 'Rank_FI': feature_impotances})
    # for i in range(len(pooled.columns)-1):
    #     importance=importance.append(pd.DataFrame({'column_name':pooled.columns[i],'feature_importance':[tree.feature_importances_[i]]}))

    # sort the feature importances
    importance = importance.sort_values(by=['Rank_FI'], ascending=False)
    return importance


def select_K(pooled):
    # Y_arr=np.array(pooled['dem1066'])
    Y_arr = list(pooled['dem1066'])
    # why tf do we have missing values in the target variable ??????????????????????
    Y_arr = [x if x != ' ' else '0' for x in Y_arr]
    Y_arr = [int(i) for i in Y_arr]
    X_arr=pooled.loc[:, pooled.columns != 'dem1066']
    model=DecisionTreeRegressor()
    # rfecv = RFECV(estimator=model,step=1)
    # rfecv.fit(X_arr,Y_arr)
    # print("Optimal number of features : %d" % rfecv.n_features_)
    
    rfe=RFE(model,10,step=1) 
    res=rfe.fit(X_arr,Y_arr)
    names=pd.DataFrame(pooled.columns[:-1])

    selected=ranked.loc[ranked['Rank']==1]
    ranked.to_csv('../input/k_best_features.csv',index=False)
    return ranked

if __name__=='__main__':
    pooled=pd.read_csv('../input/pooled_processed.csv')
    feature_importance(pooled)
    # select_K(pooled)
