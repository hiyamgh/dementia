import pandas as pd
import os

path_out = 'results_errors_filtered/'
if not os.path.exists(path_out):
    os.makedirs(path_out)

path_orig = 'results_errors/'
for file in os.listdir(path_orig):
    dataset = pd.read_csv(os.path.join(path_orig, file))
    dataset_filtered = dataset.head(20).drop(['ppv'], axis=1).to_csv(os.path.join(path_out, file), index=False)
