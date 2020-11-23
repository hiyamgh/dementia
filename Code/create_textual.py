import pandas as pd
import numpy as np
import os


def check_create_dir(dest):
    if not os.path.exists(dest):
        os.makedirs(dest)


if __name__ == '__main__':
    df = pd.read_csv('../input/pooled_data.csv')
    textual = pd.read_excel('textual.xlsx')
    vals = []
    for t in textual['COLUMN']:
        temp = df[df[t].notna()]
        vals.append(','.join(np.unique(temp[t])))
    
    textual['values'] = vals

    dest = '../input/codebooks/'
    check_create_dir(dest)
    textual.to_excel(os.path.join(dest, 'textual.xlsx'))