from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE,RFECV
import numpy as np
import pandas as pd

def feature_importance(pooled):
    Y_arr=np.array(pooled['dem1066'])
    X_arr=pooled[pooled.columns[:-1]]
    tree=DecisionTreeRegressor().fit(X_arr,Y_arr)
    importance=pd.DataFrame()
    for i in range(len(pooled.columns)-1):
        importance=importance.append(pd.DataFrame({'column_name':pooled.columns[i],'feature_importance':[tree.feature_importances_[i]]}))
    importance.to_csv('../input/feature_importance.csv',index=False)

def select_K(pooled):
    Y_arr=np.array(pooled['dem1066'])
    X_arr=pooled[pooled.columns[:-1]]
    model=DecisionTreeRegressor()
    # rfecv = RFECV(estimator=model,step=1)
    # rfecv.fit(X_arr,Y_arr)
    # print("Optimal number of features : %d" % rfecv.n_features_)
    
    rfe=RFE(model,20,step=1) 
    res=rfe.fit(X_arr,Y_arr)
    names=pd.DataFrame(pooled.columns[:-1])

    rankings=pd.DataFrame(res.ranking_)
    ranked=pd.concat([names,rankings], axis=1)
    ranked.columns=["Feature","Rank"]

    selected=ranked.loc[ranked['Rank']==1]
    selected.to_csv('../input/k_best_features.csv',index=False)

if __name__=='__main__':
    pooled=pd.read_csv('../input/pooled_processed.csv')
    # feature_importance(pooled)
    select_K(pooled)