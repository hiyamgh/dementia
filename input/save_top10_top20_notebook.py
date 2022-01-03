import pandas as pd

df = pd.read_csv('feature_importance_modified.csv')
top10 = df.loc[:10, ['Feature', 'description']]
top20 = df.loc[:20, ['Feature', 'description']]

top10.to_csv('codebooks/top10.csv', index=False)
top20.to_csv('codebooks/top20.csv', index=False)