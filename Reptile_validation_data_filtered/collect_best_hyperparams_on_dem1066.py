import pandas as pd
import pickle

"""
This script gets the best hyper parameters on dem1066 - the non-overfitting ones
"""

# get the dataframe that contains hyperparameter combinations per model number
df_hyperparams_dem1066_20 = pd.read_csv('../Reptile_validation_data/results_errors/final_hyps_top20.csv')
df_hyperparams_dem1066_10 = pd.read_csv('../Reptile_validation_data/results_errors/final_hyps_top10.csv')

# load all the hyperparameters
with open('../Reptile_validation_data/idx2hyps.pkl', 'rb') as handle:
    idx2hyps = pickle.load(handle)

# get the top 20 non overfitting ones
non_overfitting_rows_20 = []
with open('../Reptile_validation_data/results_errors/non_overfitting_top20.txt', 'r') as f:
    model_nums = f.readlines()

    for mn in model_nums:
        mn_cleaned = mn.split(',')[0][2:-1].split('_')[0]
        num = mn.split(',')[1][:-2]
        print(mn, mn_cleaned, num)
        non_overfitting_rows_20.append((mn_cleaned, num))

print('=============================================================================================')

# get the top 10 non overfitting ones
non_overfitting_rows_10 = []
with open('../Reptile_validation_data/results_errors/non_overfitting_top10.txt', 'r') as f:
    model_nums = f.readlines()

    for mn in model_nums:
        mn_cleaned = mn.split(',')[0][2:-1].split('_')[0]
        num = mn.split(',')[1][:-2]
        print(mn, mn_cleaned, num)
        non_overfitting_rows_10.append((mn_cleaned, num))

print('=============================================================================================')


properties_20, properties_10 = {}, {}
count = 0
for nov in non_overfitting_rows_20:
    row = df_hyperparams_dem1066_20.loc[df_hyperparams_dem1066_20['model'] == int(nov[0])]
    cols = list(row.columns)
    vals = list(row.values[0])
    col2val = dict(zip(cols, vals))
    properties_20[count] = col2val
    properties_20[count]['weights'] = idx2hyps[int(nov[0])]['weights']
    properties_20[count]['encoding'] = idx2hyps[int(nov[0])]['encoding']
    count += 1
for k, v in properties_20.items():
    print(k)
    print(v)
    break

with open('best_hyps_on_dem1066_top10.pkl', 'wb') as handle:
    pickle.dump(properties_20, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('=============================================================================================')

count = 0
for nov in non_overfitting_rows_10:
    row = df_hyperparams_dem1066_10.loc[df_hyperparams_dem1066_10['model'] == int(nov[0])]
    cols = list(row.columns)
    vals = list(row.values[0])
    col2val = dict(zip(cols, vals))
    properties_10[count] = col2val
    properties_10[count]['weights'] = idx2hyps[int(nov[0])]['weights']
    properties_10[count]['encoding'] = idx2hyps[int(nov[0])]['encoding']
    count += 1
for k, v in properties_10.items():
    print(k)
    print(v)
    break

with open('best_hyps_on_dem1066_top20.pkl', 'wb') as handle:
    pickle.dump(properties_10, handle, protocol=pickle.HIGHEST_PROTOCOL)

