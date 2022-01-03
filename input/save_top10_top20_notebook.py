import pandas as pd

df = pd.read_csv('feature_importance_modified.csv')
features = list(df['Feature'])
f10 = features[:10]
f20 = features[:20]

top10 = df[:10].drop(['Importance'], axis=1)
top20 = df[:20].drop(['Importance'], axis=1)

# top10 = df.loc[0:10, ['Feature', 'description']] # in loc, start and stop are both included
# top20 = df.loc[0:20, ['Feature', 'description']]# in loc, start and stop are both included
#
top10.to_csv('codebooks/top10.csv', index=False)
top20.to_csv('codebooks/top20.csv', index=False)