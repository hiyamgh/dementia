import pandas as pd
import pickle

"""
This script gets the best hyper parameters on dem1066 - the non-overfitting ones
"""

# get the dataframe that contains hyperparameter combinations per model number
df_hyperparams_dem1066 = pd.read_csv('../Reptile_validation_data/results_errors/final_hyps_top20.csv')

# load all the hyperparameters
with open('../Reptile_validation_data/idx2hyps.pkl', 'rb') as handle:
    idx2hyps = pickle.load(handle)

# get the non overfitting ones
non_overfitting_rows = []
with open('../Reptile_validation_data/results_errors/non_overfitting_top20.txt', 'r') as f:
    model_nums = f.readlines()

    for mn in model_nums:
        mn_cleaned = mn.split(',')[0][2:-1].split('_')[0]
        num = mn.split(',')[1][:-2]
        print(mn, mn_cleaned, num)
        non_overfitting_rows.append((mn_cleaned, num))


properties = {}
count = 0
for nov in non_overfitting_rows:
    row = df_hyperparams_dem1066.loc[df_hyperparams_dem1066['model'] == int(nov[0])]
    cols = list(row.columns)
    vals = list(row.values[0])
    col2val = dict(zip(cols, vals))
    properties[count] = col2val
    properties[count]['weights'] = idx2hyps[int(nov[0])]['weights']
    properties[count]['encoding'] = idx2hyps[int(nov[0])]['encoding']
    count += 1


for k, v in properties.items():
    print(k)
    print(v)
    break

with open('best_hyps_on_dem1066.pkl', 'wb') as handle:
    pickle.dump(properties, handle, protocol=pickle.HIGHEST_PROTOCOL)

