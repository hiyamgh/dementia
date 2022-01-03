import pandas as pd

df = pd.read_csv('feature_importance_modified.csv')
top10 = df.loc[0:9, ['Feature', 'description']] # in loc, start and stop are both included
top20 = df.loc[0:19, ['Feature', 'description']]# in loc, start and stop are both included

top10.to_csv('codebooks/top10.csv', index=False)
top20.to_csv('codebooks/top20.csv', index=False)