import pandas as pd

df_281 = pd.read_csv('full validation data (281).csv')
df_imp = pd.read_csv('../input/feature_importance_modified.csv')
imp_feats = list(df_imp['Feature'])[:20]
imp_feats = [f[:-2] if '_2' in f else f for f in imp_feats]
df_281[imp_feats + ['dem1066']].to_csv('full validation data (281) filtered.csv', index=False)