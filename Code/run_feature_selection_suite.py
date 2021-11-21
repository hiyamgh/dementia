import pandas as pd
from feature_selection_suite import FeatureSelection
import os

if __name__ == '__main__':

    # read the training + testing data
    df_train = pd.read_csv('../input/train_imputed_scaled.csv')
    df_test = pd.read_csv('../input/test_imputed_scaled.csv')

    # concatenate
    df = pd.concat([df_train, df_test])

    output_folder = '../output/feature_selection/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # don't change the name of the target variable (as it will be taken automatically from the loop)
    fs = FeatureSelection(df, target_variable='dem1066',
                          output_folder=output_folder,
                          cols_drop=None,
                          scale=False,
                          scale_input=False,
                          scale_output=False,
                          output_zscore=False,
                          output_minmax=False,
                          output_box=False,
                          output_log=False,
                          input_zscore=None,
                          input_minmax=None,
                          input_box=None,
                          input_log=None,
                          regression=False)

    fs.feature_importance(xg_boost=True, extra_trees=True, random_forest=True)
    fs.univariate()
    fs.rfe()

    df_rfe = pd.read_csv('../output/feature_selection/rfe_rankings.csv')
    rfe_top20 = list(df_rfe['feature'])[:20]
    print(rfe_top20)

    df_xgb = pd.read_csv('../output/feature_selection/xgb_fs_importances.csv')
    xgb_top25 = list(df_xgb['feature'])[:30]
    print(xgb_top25)

    df_rf = pd.read_csv('../output/feature_selection/randomforest_fs_importances.csv')
    rf_top25 = list(df_rf['feature'])[:30]
    print(rf_top25)

    df_et = pd.read_csv('../output/feature_selection/extratrees_fs_importances.csv')
    et_top25 = list(df_et['feature'])[:30]
    print(et_top25)

    df_uni = pd.read_csv('../output/feature_selection/univariate_fs_scores.csv')
    uni_top25 = list(df_uni['feature'])[-20:]
    print(uni_top25)

    df = pd.read_csv('../input/feature_importance_modified.csv')
    features = list(df['Feature'])[:20]
    for f in features:
        found = []
        if f in rfe_top20:
            found.append('rfe')
        if f in xgb_top25:
            found.append('xgb')
        if f in rf_top25:
            found.append('rf')
        if f in et_top25:
            found.append('et')
        if f in uni_top25:
            found.append('uni')

        print('{} found in {}'.format(f, found))

    # ['Q451_2', 'PUT_2', 'Q1551_2', 'NRECALL_2', 'Q281_2', 'MONTH_2', 'Q401_2', 'LEARN3_2', 'LEARN2_2', 'LEARN1_2', 'LASTSEE_2', 'LASTDAY_2', 'Q722_2', 'Q72_2', 'FEED_2', 'Q1041_2', 'TRIALNO_2', 'ANIMALS_2', 'Q76_2', 'RECALL_2']
    # ['LASTDAY_2', 'TOILET_2', 'Q401_2', 'Q76_2', 'RECALL_2', 'Q72_2', 'MEMORY_2', 'PUT_2', 'ANIMALS_2', 'Q75_2', 'KEPT_2', 'Q1754_2', 'MONTH_2', 'Q722_2', 'MUDDLED_2', 'Q1061_2', 'Q191_2', 'Q1808_2', 'YEAR_2', 'Q32_2', 'Q41_2', 'Q71_2', 'ICINTCOM_2', 'SEASON_2', 'Q31_2', 'Q451_2', 'Q651_2', 'Q1141_2', 'CHANGE_2', 'Q1709_2']
    # ['LASTDAY_2', 'RECALL_2', 'LEARN1_2', 'Q75_2', 'LEARN2_2', 'YEAR_2', 'MEMORY_2', 'Q76_2', 'Q72_2', 'LEARN3_2', 'Q401_2', 'CONVERS_2', 'PUT_2', 'REASON_2', 'LASTSEE_2', 'MONEY_2', 'Q1781_2', 'ANIMALS_2', 'SEASON_2', 'MUDDLED_2', 'PAST_2', 'FRDNAME_2', 'Q74_2', 'STORY_2', 'WORDDEL_2', 'Q391_2', 'MONTH_2', 'Q1551_2', 'Q31_2', 'Q331_2']
    # ['MONEY_2', 'Q1781_2', 'LASTDAY_2', 'PUT_2', 'MEMORY_2', 'YEAR_2', 'Q75_2', 'Q76_2', 'SEASON_2', 'LASTSEE_2', 'Q401_2', 'PAST_2', 'Q72_2', 'REASON_2', 'MONTH_2', 'FRDNAME_2', 'RECALL_2', 'STORY_2', 'LEARN2_2', 'MENTAL_2', 'LEARN1_2', 'Q391_2', 'STREET_2', 'WORDFIND_2', 'Q73_2', 'LONGMEM_2', 'KEPT_2', 'Q31_2', 'LEARN3_2', 'Q74_2']
    # ['CONVERS_2', 'Q72_2', 'FRDNAME_2', 'Q73_2', 'Q1781_2', 'CARENEED_2', 'LEARN1_2', 'MEMORY_2', 'LEARN2_2', 'REASON_2', 'MONEY_2', 'LASTSEE_2', 'LEARN3_2', 'YEAR_2', 'Q401_2', 'PUT_2', 'Q76_2', 'Q75_2', 'RECALL_2', 'LASTDAY_2']

    # LASTDAY_2 found in ['rfe', 'xgb', 'rf', 'et', 'uni']
    # Q75_2 found in ['xgb', 'rf', 'et', 'uni']
    # YEAR_2 found in ['xgb', 'rf', 'et', 'uni']
    # RECALL_2 found in ['rfe', 'xgb', 'rf', 'et', 'uni']
    # MEMORY_2 found in ['xgb', 'rf', 'et', 'uni']
    # LEARN3_2 found in ['rfe', 'rf', 'et', 'uni']
    # Q1781_2 found in ['rf', 'et', 'uni']
    # LEARN2_2 found in ['rfe', 'rf', 'et', 'uni']
    # LEARN1_2 found in ['rfe', 'rf', 'et', 'uni']
    # PUT_2 found in ['rfe', 'xgb', 'rf', 'et', 'uni']
    # Q72_2 found in ['rfe', 'xgb', 'rf', 'et', 'uni']
    # Q401_2 found in ['rfe', 'xgb', 'rf', 'et', 'uni']
    # SEASON_2 found in ['xgb', 'rf', 'et']
    # MONEY_2 found in ['rf', 'et', 'uni']
    # STORY_2 found in ['rf', 'et']
    # CONVERS_2 found in ['rf', 'uni']
    # STREET_2 found in ['et']
    # REASON_2 found in ['rf', 'et', 'uni']
    # Q76_2 found in ['rfe', 'xgb', 'rf', 'et', 'uni']
    # ANIMALS_2 found in ['rfe', 'xgb', 'rf']

    # allls = [rfe_top20, xgb_top25, rf_top25, et_top25, uni_top25]
    # inter = set.intersection(*map(set, allls))
    # print(inter)
    # print(len(inter))
