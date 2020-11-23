import pandas as pd
import numpy as np
import os
from utils import get_col_name_in_pooled


def check_create_dir(dest):
    if not os.path.exists(dest):
        os.makedirs(dest)

'''
Hiyam doing the change here:
The commented block below reads from textual.xlsx and writes back to it.
So, I'm not sure what the original textual.xlsx is before we read from it.
Therefore, I will do my own filtering. 
'''

# if __name__ == '__main__':
#     df = pd.read_csv('../input/pooled_data.csv')
#     textual = pd.read_excel('textual.xlsx')
#     vals = []
#     for t in textual['COLUMN']:
#         temp = df[df[t].notna()]
#         vals.append(','.join(np.unique(temp[t])))
#
#     textual['values'] = vals
#
#     dest = '../input/codebooks/'
#     check_create_dir(dest)
#     textual.to_excel(os.path.join(dest, 'textual.xlsx'))


def get_description(copyofdem, col):
    return copyofdem.loc[col, 'label::English']


def create_textual_data(features_with_cats, pooled):
    textual = features_with_cats[features_with_cats['data_type'] == 'text']
    textual = textual.drop(['min', 'max', 'nb_categories'], axis=1)

    descriptions = []
    copyofdemcols = list(copyofdem.index)
    for col in list(textual['COLUMN']):
        if col in copyofdemcols:
            descriptions.append(get_description(copyofdem, col))
        else:
            col_renamed = get_col_name_in_pooled(col, pooled)
            if col_renamed in copyofdemcols:
                descriptions.append(get_description(copyofdem, col))
            else:
                descriptions.append('not found')
    textual['description'] = descriptions

    # get the description and values
    dest = '../input/codebooks/'
    check_create_dir(dest)

    textual = textual[['COLUMN', 'description', 'data_type', 'name_type', 'val_range']]
    textual.to_excel(os.path.join(dest, 'textual.xlsx'))


if __name__ == '__main__':
    copyofdem = pd.read_excel('../input/Copy of Dementia_baseline_questionnaire_V1.xlsx').set_index('name')
    pooled = pd.read_csv('../input/pooled_data.csv')
    features_with_categories = pd.read_csv('features_with_categories.csv')
    create_textual_data(features_with_cats=features_with_categories, pooled=pooled)
