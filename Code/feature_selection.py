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
    
    rfe = RFE(model, 20,step=1)
    res = rfe.fit(X_arr, Y_arr)
    names = pd.DataFrame(pooled.columns[:-1])

    rankings = pd.DataFrame(res.ranking_)
    ranked = pd.concat([names,rankings], axis=1)
    ranked.columns = ["Feature", "Rank_univariate"]

    # selected=ranked.loc[ranked['Rank_univariate'] == 1]
    # selected.to_csv('../input/k_best_features.csv',index=False)
    ranked = ranked.sort_values(by=['Rank_univariate'], ascending=True)

    return ranked


if __name__=='__main__':
    pooled = pd.read_csv('../input/pooled_proc_scimp.csv')

    # features ranked by their importance
    ranked_FI = feature_importance(pooled)
    ranked_univariate = select_K(pooled)

    all_ranks = {}
    ranks_FI_dict = ranked_FI.set_index('Feature').T.to_dict('list')
    ranks_univariate_dict = ranked_univariate.set_index('Feature').T.to_dict('list')

    all_cols = list(pooled.columns)
    all_cols.remove('dem1066')

    # data frame containing both ranks for each feature
    uni = [ranks_univariate_dict[feature][0] for feature in all_cols]
    fi = [ranks_FI_dict[feature][0] for feature in all_cols]
    all_fs = pd.DataFrame({'Features': all_cols, 'Rank_uni': uni, 'Rank_FI': fi})
    all_fs = all_fs.sort_values(by=['Rank_uni', 'Rank_FI']).reset_index(drop=True)
    all_fs.to_csv('../input/feature_selections.csv', index=False)

    top50_FI = list(ranked_FI[:50]['Feature'])
    top50_univariate = list(ranked_univariate[:50]['Feature'])
    intersection1_top50 = set(top50_FI).intersection(top50_univariate)
    top_features1 = pd.DataFrame({'univariate_FI': list(intersection1_top50)})
    top_features1.to_csv('../input/top_intersection.csv', index=False)
