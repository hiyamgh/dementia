import pandas as pd
import numpy as np


df = pd.read_csv('../input/pooled_data.csv')
count = 0
is_validation = []
for i, row in df.iterrows():
    if np.isnan(row['CARERAGE_2']) and np.isnan(row['CARERSEX_2']) and np.isnan(row['CARERREL_2']):
        is_validation.append(1)
    else:
        is_validation.append(0)
df['is_validation'] = is_validation
df.to_csv('../input/pooled_data_validation_marked.csv', index=0)
print(len(df[df['is_validation'] == 1]))

# get statistics aout validation data (% of each class)
df_train = df[df['is_validation'] == 0]
df_test = df[df['is_validation'] == 1]

print('df_train')
print(df_train['dem1066'].value_counts())
print('0: {}'.format(list(df_train['dem1066']).count('0')/len(df_train)))
print('1: {}'.format(list(df_train['dem1066']).count('1')/len(df_train)))
print('df_test')
print(df_test['dem1066'].value_counts())
print('0: {}'.format(list(df_test['dem1066']).count('0')/len(df_test)))
print('1: {}'.format(list(df_test['dem1066']).count('1')/len(df_test)))

# size of the training/testing:
print('\ntraining size: {}'.format(len(df_train)/len(df)))
print('testing size: {}'.format(len(df_test)/len(df)))
# df_train
# 0    630
# 1     79
#       22
# Name: dem1066, dtype: int64
# df_test
# 0    169
# 1    124
# Name: dem1066, dtype: int64

# Class Distribution Statistics:
# Name: dem1066, dtype: int64
# 0: 0.8618331053351573
# 1: 0.10807113543091655
# df_test
# 0    169
# 1    124
# Name: dem1066, dtype: int64
# 0: 0.5767918088737202
# 1: 0.4232081911262799