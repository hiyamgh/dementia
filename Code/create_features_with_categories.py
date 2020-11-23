import pandas as pd
import numpy as np
import os

choices = pd.read_excel('../input/Copy of Dementia_baseline_questionnaire_V1.xlsx',sheet_name='choices')


def check_create_dir(dest):
    if not os.path.exists(dest):
        os.makedirs(dest)


def match(row):
    if 'select' not in row['name_type']: # type is not a category
        row['data_type']=row['name_type']
        # print(row['name_type'])
    else:
        c=row['name_type'].split(' ')[1]
        if c=='yes_no':
            row['data_type']='boolean'
        else:
            row['data_type']='categorical'
        sub=choices[choices['list_name']==c]
        row['nb_categories']=len(sub)
        row['val_range']=','.join(np.array(sub['label::English']))

    return row


def match_df():
    df=pd.read_csv('features_meta.csv', encoding='utf-8-sig')
    df=df.apply(lambda row: match(row), axis=1)
    df.to_csv('features_with_categories.csv', encoding='utf-8-sig', index=False)


if __name__=='__main__':
    match_df()
