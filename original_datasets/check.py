import pandas as pd
import pickle

# df1 = pd.read_csv('dementia data 502 baseline updated.csv')
# df2 = pd.read_csv('full validation data (281).csv')
#
# cols1 = list(df1.columns)
# cols2 = list(df2.columns)
#
# print('df1.shape: {}'.format(df1.shape))
# print('df2.shape: {}'.format(df2.shape))

pooled = pd.read_csv('../input/pooled_data.csv')
val = pd.read_csv('../original_datasets/full validation data (281).csv')
baseline = pd.read_csv('dementia data 502 baseline updated.csv')
# val.rename(columns={
#     'CAREAGE': 'CARERAGE_2',
#     'CARESEX': 'CARESEX_2',
#     'CAREREL': 'CAREREL_2',
#     'CLUSID': 'CLUSTID',
#     'Q366A': 'Q366a_2',
#     'ABUSE1': 'ABUSE_2',
#     'CJOBCAT1': 'CJOBCAT_2',
#     'HOURHELP': 'HELPHOUR_2',
#     'IRDELHAL': 'IRDISCR_2',
#     'KNUCKLE': 'Chin_2',
#     'CAREPAID': 'CARPAID_2',
#     'NTRPAID': 'NTPAID_2',
#     'HELPCUT': 'HELPCUT1_2',
#     'HELPCUT1': 'HELPCUT2_2',
#     'HELPCUT2': 'HELPCUT3_2',
#     'CINCOME9': 'CKIN_2_SPE_2',
#     'CINCOME': 'CINCOM1_2',
#     'CINCOME10': 'CKINCOM_2',
#     "CINCOME1": "CKINCOM_2/1",
#     "CINCOME2": "CKINCOM_2/2",
#     "CINCOME3": "CKINCOM_2/3",
#     "CINCOME4": "CKINCOM_2/4",
#     "CINCOME5": "CKINCOM_2/5",
#     "CINCOME6": "CKINCOM_2/6",
#     "CINCOME7": "CKINCOM_2/7",
#     "CINCOME8": "CKINCOM_2/8",
# }, inplace=True)

cols_pooled = list(pooled.columns)
cols_val = list(val.columns)

colstochange={}
for colp in cols_pooled:
    for colv in cols_val:
        if colp == colv + '_2':
            # print(colv, colp)
            colstochange[colv] = colp

val.rename(columns=colstochange, inplace=True)
# cols_val_filt = list(val.columns)
common_columns = list(colstochange.values())
print(common_columns)
print(len(common_columns))

# # loop over the rows of the pooled data and check if you find exact similar rows in full validation data
pooled_filtered = pooled[common_columns]
val_filtered = val[common_columns]

c1 = list(pooled_filtered.columns)
c2 = list(val_filtered.columns)

print([c for c in c1 if c not in c2])
print([c for c in c2 if c not in c1])
print(list(set(c2) - set(c1)))
print(list(set(c1) - set(c2)))
print(pooled_filtered.shape)
print(val_filtered.shape)
extra = len(pooled_filtered) - len(val_filtered)

# for _ in range(extra):
#     val_filtered = val_filtered.append(pd.Series(), ignore_index=True)
val_filtered = val_filtered[pooled_filtered.columns]
#
# val_filtered['index'] = list(range(len(pooled_filtered)))
# val_filtered.set_index('index')
# pooled_filtered['index'] = list(range(len(pooled_filtered)))
# pooled_filtered.set_index('index')
#
# val_filtered = val_filtered.fillna('-')
# pooled_filtered = pooled_filtered.fillna('-')

# with open("../input/codebooks/legal_cols_filtered.txt", "rb") as fp:  # Unpickling
#     legal_cols = pickle.load(fp)

# legal = pd.read_csv('../Code/legal.csv')
# legal_cols = list(legal['COLUMN'])

# legal = pd.read_csv('../input/feature_importance_modified.csv')
# legal_cols = list(legal['Feature'])[:20]

# legal_common = [c for c in legal_cols if c in common_columns]
# pooled_filtered = pooled_filtered[legal_common]
# val_filtered = val_filtered[legal_common]
# print('after getting legal cols:')
# print(pooled_filtered.shape)
# print(val_filtered.shape)
val_filtered.to_csv('val_filtered.csv', index=False)
pooled_filtered.to_csv('pooled_filtered.csv', index=False)

pooled_filtered.isna().sum().to_csv('summation_pooled.csv')
print('-====================================================')
val_filtered.isna().sum().to_csv('summation_val.csv')
print('-====================================================')
baseline.isna().sum().to_csv('summation_baseline.csv')


#
# # #
# # # pooled_filtered.sort_index(inplace=True)
# #
# # print(pooled_filtered.shape)
# # print(val_filtered.shape)
# #
# # equality = pooled_filtered.where(pooled_filtered.values==val_filtered.values)
# # equality.to_csv('equality.csv')
# equality = pooled_filtered == val_filtered
# # equality.to_csv('equality.csv')
#
# equalities = []
# for i, row in equality.iterrows():
#     # print('i: {}, sum: {}'.format(i, row.sum()))
#     equalities.append((i, row.sum()))
# sequalities = list(sorted(equalities, key=lambda x: x[1], reverse=True))
# print('==================================================')
# count=1
# for t in sequalities:
#     print('{}: row: {}, sum: {}'.format(count, t[0], t[1]))
#     count += 1
# # pooled_filtered.where(pooled_filtered.values==val_filtered.values).to_csv('equality.csv', index=False)