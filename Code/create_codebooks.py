import pandas as pd
import numpy as np
from create_numeric import get_col_name_in_pooled
import os
import collections


def check_create_dir(dest):
    if not os.path.exists(dest):
        os.makedirs(dest)


def get_missing_codebook(merged, pooled, perc):
    ''' 
    creates a code-book for features having more than `perc` missing entries 
    :param merged: merged data 
    :param pooled: pooled data (Beqaa questionnaire)
    :param perc: percentage cut-off. Example: 40 denoting 40 %
    :return returns two dataframes:
    1. the merged data frame with missing values added as columns in the data frame
    2. a code-book with only features & their missing values
    '''

    missing = []
    missing_codebook = pd.DataFrame(columns=['perc_missing'])

    for _, row in merged.iterrows():
        # column name in merged
        col_name = row['COLUMN']

        # column name in pooled data 
        repres = get_col_name_in_pooled(col_name, pooled)

        if isinstance(repres, str):
            # get the percentage of missing data
            num_missing = pooled[repres].isna().sum()
            perc_missing = num_missing / len(pooled) * 100

            if perc_missing >= perc:
                missing.append(perc_missing)
                missing_codebook.loc[col_name] = pd.Series({
                    'perc_missing': perc_missing
                })
            else:
                missing.append(0)
        else:
            missing.append(0)
            print('Will not process {} because it is not found in Beqaa Questionnaire'.format(col_name))

    merged['perc_missing'] = missing
    return merged, missing_codebook


def get_erroneous_codebook(features, data):
    '''

    :param features: merged data
    :param data: pooled data
    :return: returns two dataframes:
    1. the merged data frame with erroneous data added as columns in the data frame
    2. a code-book with only features & their erroneous data
    '''

    nb_erroneous, erroneous_vals, cut_off = [], [], []
    erroneous_codebook = pd.DataFrame(columns=['perc_erroneous', 'erroneous', 'cut_off'])

    for _, row in features.iterrows():
        col = row['COLUMN']
        if 'age' in col.lower():

            erroneous = data[(data[col] > 100) & (2020 - data[col] > 100)]
            perc_erroneous = (len(erroneous)/len(features))*100
            err_vals = ','.join(str(v).replace('.0', '') for v in np.unique(erroneous[col]))
            co = '100'

            nb_erroneous.append(perc_erroneous)
            erroneous_vals.append(err_vals)
            cut_off.append(co)

            erroneous_codebook.loc[col] = pd.Series({
                'perc_erroneous': perc_erroneous,
                'erroneous': err_vals,
                'cut_off': co
            })

        elif 'animals' in col.lower():
            erroneous = data[(data[col] > 10)]
            perc_erroneous = (len(erroneous)/len(features))*100
            err_vals = ','.join(str(v).replace('.0', '') for v in np.unique(erroneous[col]))
            co = '10'

            nb_erroneous.append(perc_erroneous)
            erroneous_vals.append(err_vals)
            cut_off.append(co)

            erroneous_codebook.loc[col] = pd.Series({
                'perc_erroneous': perc_erroneous,
                'erroneous': err_vals,
                'cut_off': co
            })
        elif row['data_type'] == 'categorical' or row['data_type'] == 'ordinal':

            if isinstance(row['cat_options'], str) and col != 'CKINCOM_2':
                erroneous = data[data[col] > len(row['cat_options'].split(','))]
                # nb_erroneous.append((len(erroneous)/len(features))*100)
                # erroneous_vals.append(','.join(str(v).replace('.0','') for v in np.unique(erroneous[col])))
                # cut_off.append('%d'%(len(row['cat_options'].split(','))-1))

                perc_erroneous = (len(erroneous)/len(features))*100
                err_vals = ','.join(str(v).replace('.0', '') for v in np.unique(erroneous[col]))
                co = '%d' % (len(row['cat_options'].split(',')) - 1)

                nb_erroneous.append(perc_erroneous)
                erroneous_vals.append(err_vals)
                cut_off.append(co)

                erroneous_codebook.loc[col] = pd.Series({
                    'perc_erroneous': perc_erroneous,
                    'erroneous': err_vals,
                    'cut_off': co
                })

            else:
                nb_erroneous.append(0)
                erroneous_vals.append('')
                cut_off.append('')

                erroneous_codebook.loc[col] = pd.Series({
                    'perc_erroneous': 0,
                    'erroneous': '',
                    'cut_off': ''
                })

        else:
            data[col] = data[col].fillna(-1)
            if col == 'HELPHOUR_2':
                erroneous=data[data[col]>168]
                # nb_erroneous.append((len(erroneous)/len(features))*100)
                # erroneous_vals.append(','.join(str(v).replace('.0','') for v in np.unique(erroneous[col])))
                # cut_off.append('168 i.e. number of hours per week')

                perc_erroneous = (len(erroneous)/len(features))*100
                err_vals = ','.join(str(v).replace('.0', '') for v in np.unique(erroneous[col]))
                co = '168 i.e. number of hours per week'

                nb_erroneous.append(perc_erroneous)
                erroneous_vals.append(err_vals)
                cut_off.append(co)

                erroneous_codebook.loc[col] = pd.Series({
                    'perc_erroneous': perc_erroneous,
                    'erroneous': err_vals,
                    'cut_off': co
                })

            elif 'LEARN' in col:
                erroneous=data[data[col]>10]
                # nb_erroneous.append((len(erroneous)/len(features))*100)
                # erroneous_vals.append(','.join(str(v).replace('.0','') for v in np.unique(erroneous[col])))
                # cut_off.append('10')

                perc_erroneous = (len(erroneous) / len(features)) * 100
                err_vals = ','.join(str(v).replace('.0', '') for v in np.unique(erroneous[col]))
                co = 10

                nb_erroneous.append(perc_erroneous)
                erroneous_vals.append(err_vals)
                cut_off.append(co)

                erroneous_codebook.loc[col] = pd.Series({
                    'perc_erroneous': perc_erroneous,
                    'erroneous': err_vals,
                    'cut_off': co
                })

            elif 'DAY1A' in col:
                erroneous = data[data[col] > 24]
                # nb_erroneous.append((len(erroneous)/len(features))*100)
                # erroneous_vals.append(','.join(str(v).replace('.0','') for v in np.unique(erroneous[col])))
                # cut_off.append('24')

                perc_erroneous = (len(erroneous) / len(features)) * 100
                err_vals = ','.join(str(v).replace('.0', '') for v in np.unique(erroneous[col]))
                co = 24

                nb_erroneous.append(perc_erroneous)
                erroneous_vals.append(err_vals)
                cut_off.append(co)

                erroneous_codebook.loc[col] = pd.Series({
                    'perc_erroneous': perc_erroneous,
                    'erroneous': err_vals,
                    'cut_off': co
                })

            else:
                nb_erroneous.append(0)
                erroneous_vals.append('')
                cut_off.append('')

                erroneous_codebook.loc[col] = pd.Series({
                    'perc_erroneous': 0,
                    'erroneous': '',
                    'cut_off': ''
                })

    features['perc_erroneous'] = nb_erroneous
    features['erroneous'] = erroneous_vals
    features['cut_off'] = cut_off

    return features, erroneous_codebook

# def get_erroneous_codebook(features,data):
#     nb_erroneous=[]
#     erroneous_vals=[]
#     cut_off=[]
#     for _,row in features.iterrows():
#         col=row['COLUMN']
#         if 'age' in col.lower():
#             erroneous=data[(data[col] > 100) & (2020 - data[col] > 100)]
#             nb_erroneous.append((len(erroneous)/len(features))*100)
#             erroneous_vals.append(','.join(str(v).replace('.0','') for v in np.unique(erroneous[col])))
#             cut_off.append('100')
#         elif row['data_type']=='categorical' or row['data_type']=='ordinal':
#             if isinstance(row['cat_options'], str) and col != 'CKINCOM_2':
#                 erroneous=data[data[col]>len(row['cat_options'].split(','))]
#                 nb_erroneous.append((len(erroneous)/len(features))*100)
#                 erroneous_vals.append(','.join(str(v).replace('.0','') for v in np.unique(erroneous[col])))
#                 cut_off.append('%d'%(len(row['cat_options'].split(','))-1))
#             else:
#                 nb_erroneous.append(0)
#                 erroneous_vals.append('')
#                 cut_off.append('')
#         else:
#             data[col]=data[col].fillna(-1)
#             if col == 'HELPHOUR_2':
#                 erroneous=data[data[col]>168]
#                 nb_erroneous.append((len(erroneous)/len(features))*100)
#                 erroneous_vals.append(','.join(str(v).replace('.0','') for v in np.unique(erroneous[col])))
#                 cut_off.append('168 i.e. number of hours per week')
#
#             elif 'LEARN' in col:
#                 erroneous=data[data[col]>10]
#                 nb_erroneous.append((len(erroneous)/len(features))*100)
#                 erroneous_vals.append(','.join(str(v).replace('.0','') for v in np.unique(erroneous[col])))
#                 cut_off.append('10')
#
#             elif 'DAY1A' in col:
#                 erroneous=data[data[col]>24]
#                 nb_erroneous.append((len(erroneous)/len(features))*100)
#                 erroneous_vals.append(','.join(str(v).replace('.0','') for v in np.unique(erroneous[col])))
#                 cut_off.append('24')
#             else:
#                 nb_erroneous.append(0)
#                 erroneous_vals.append('')
#                 cut_off.append('')
#
#     features['perc_erroneous']=nb_erroneous
#     features['erroneous']=erroneous_vals
#     features['cut_off']=cut_off
#     return features


def get_missing_rows_codebook(pooled):
    ''' Please write a loop that goes over all the rows, and produces the following:
    for $i= 1$ to $n$ where $n$ is the total number of columns,
    produce x_i, the number of rows with $i$ missing entries.
    :param pooled: pooled data (Beqaa questionnaire)
    '''

    # get the number of missing entries in each row; key: missing values, vale: number of rows
    num_missing = {}
    # loop over the rows
    for i in range(len(pooled)):
        # the number of missing entries in the current row
        curr_missing = pooled.iloc[i].isnull().sum()

        # increment the number of rows with $curr_missing$ number of entries (missing columns)
        if curr_missing in num_missing:
            num_missing[curr_missing] += 1
        else:
            num_missing[curr_missing] = 1

    # sort dictionary by increasing order of missing entries
    num_missing = collections.OrderedDict(sorted(num_missing.items()))

    # nrme: n: number, r: rows, m: missing, e: entries
    nrme = pd.DataFrame(columns=['missing_entries', 'num_rows'])
    nrme['missing_entries'] = list(num_missing.keys())
    nrme['num_rows'] = list(num_missing.values())

    return nrme


if __name__ == '__main__':
    pooled = pd.read_csv('../input/pooled_data.csv')
    merged = pd.read_excel('merged_mod.xlsx')

    # generate code-books for: number of rows with missing entries, missing greater than 40%,
    # and erroneous data
    nrme_codebook = get_missing_rows_codebook(pooled)
    merged, missing_codebook = get_missing_codebook(merged, pooled, 40)
    merged, erroneous_codebook = get_erroneous_codebook(merged, pooled)

    # create destination folder to store code-books
    dest = '../input/codebooks/'
    check_create_dir(dest)

    # save datasets as csv files in input folder
    merged.to_excel(os.path.join(dest, 'numeric.xlsx'), index=False)
    nrme_codebook.to_csv(os.path.join(dest, 'num_rows_missing_entries_codebook.csv'), index=False)
    missing_codebook.to_csv(os.path.join(dest, 'missing_40_codebook.csv'))
    erroneous_codebook.to_csv(os.path.join(dest, 'erroneous_codebook.csv'))
