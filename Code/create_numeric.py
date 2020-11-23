import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

mapping = {
    'CAREAGE': 'CARERAGE_2',
    'CARESEX': 'CARERSEX',
    'CAREREL': 'CARERREL',
    'CLUSID': 'CLUSTID',
    'Q366A': 'Q366a_2',
    'ABUSE1': 'ABUSE_2',
    'CJOBCAT1': 'CJOBCAT_2',
    'HOURHELP': 'HELPHOUR_2',
    'IRDELHAL': 'IRDISCR_2',
    'KNUCKLE': 'Chin_2',
    'CAREPAID': 'CARPAID_2',
    'NTRPAID': 'NTPAID_2',
    'HELPCUT': 'HELPCUT1_2',
    'HELPCUT1': 'HELPCUT2_2',
    'HELPCUT2': 'HELPCUT3_2',
    'CINCOME9': 'CKIN_2_SPE_2',
    'CINCOME': 'CINCOM1_2',
    'CINCOME10': 'CKINCOM_2',
    "CINCOME1": "CKINCOM_2/1",
    "CINCOME2": "CKINCOM_2/2",
    "CINCOME3": "CKINCOM_2/3",
    "CINCOME4": "CKINCOM_2/4",
    "CINCOME5": "CKINCOM_2/5",
    "CINCOME6": "CKINCOM_2/6",
    "CINCOME7": "CKINCOM_2/7",
    "CINCOME8": "CKINCOM_2/8",
}


def check_create_dir(dest):
    if not os.path.exists(dest):
        os.makedirs(dest)


def plot_distributions(pooled, col):
    # dest, resembling destination folder, create it if not there already
    dest = 'distributions'
    if not os.path.exists(dest):
        os.makedirs(dest)

    plt.hist(pooled[col].dropna())
    plt.xlabel('{}'.format(col))
    plt.ylabel('frequency')
    plt.savefig(os.path.join(dest, '{}'.format(col)))
    plt.close()


def make_clickable(val):
    # target _blank to open new window
    return '<a target="_blank" href="{}">{}</a>'.format(val, val)


def get_col_name_in_pooled(col, pooled):
    # flip the dictionary
    mapping_inv = {v: k for k, v in mapping.items()}

    if col == 'Unnamed: 552':
        return -1

    col_pooled = None

    if col in pooled.columns:
        col_pooled = col
    else:
        if col[-2:] == '_2' or col in mapping or col in mapping_inv:
            if col in mapping:
                print('col {} in mapping, transformed to {}'.format(col, mapping[col]))
                col_temp = mapping[col]
                if col_temp in pooled.columns:
                    print('{} in pooled'.format(col_temp))
                    col_pooled = col_temp
                else:
                    raise ValueError('col_temp {} not found in pooled'.format(col_temp))
            elif col in mapping_inv:
                print('col {} in mapping, transformed to {}'.format(col, mapping_inv[col]))
                col_temp = mapping_inv[col]
                if col_temp in pooled.columns:
                    print('{} in pooled'.format(col_temp))
                    col_pooled = col_temp
                else:
                    raise ValueError('col_temp {} not found in pooled'.format(col_temp))
                    # if the column, after transformation, is still not found
                    # return -1

            else:
                if col not in pooled.columns:
                    col_temp = col[:-2]
                    col_pooled = col_temp
                    if col_temp in mapping:
                        col_temp = mapping[col_temp]
                        col_pooled = col_temp
                    if col_temp in mapping_inv:
                        col_temp = mapping_inv[col_temp]
                        col_pooled = col_temp
                    if col_temp not in pooled.columns and 'Q' in col_temp:
                        col_temp = 'q' + col_temp[1:]
                        col_pooled = col_temp
                else:
                    col_temp = col
                    if col_temp not in pooled.columns and 'Q' in col_temp:
                        col_temp = 'q' + col_temp[1:]
                        col_pooled = col_temp

    if col_pooled is not None:
        return col_pooled
    return col


if __name__ == '__main__':
    # split the dataset into numeric and textual
    df = pd.read_csv('features_with_categories.csv')
    df_numeric = df[df['data_type'] != 'text']
    df_textual = df[df['data_type'] == 'text']

    if not len(df_numeric) + len(df_textual) == len(df):
        raise ValueError('df_numeric and df_textual do not sum up correctly')
    else:
        print('correct')
        df_numeric.to_csv('features_with_categories_numeric.csv', index=False)
        df_textual.to_csv('features_with_categories_textual.csv', index=False)

    pool = pd.read_csv('../input/pooled_data.csv')
    merged = pd.read_excel('merged.xlsx')
    copyofdem = pd.read_excel('../input/Copy of Dementia_baseline_questionnaire_V1.xlsx')
    cols_copyofdem = list(copyofdem['name'])
    merged = merged.drop(['colorcode', 'name_type', 'nb_categories', 'missing', 'PATIENT'], axis=1)

    descs = []
    data_types = []
    missing = []
    dists = []

    categorical_options = []
    val_range = []

    min_vals, max_vals = [], []

    for i, row in merged.iterrows():
        col_name = row['COLUMN']
        dt = row['data_type']

        # description of the columns
        if col_name in cols_copyofdem:
            ser = copyofdem.loc[copyofdem['name'] == col_name]
            idx = ser.index[0]
            desc = ser.at[idx, 'label::English']
            descs.append(desc)
        else:
            descs.append('not found')

        # data types of the column
        if row['feature_binning'] == 'ordinal':
            data_types.append('ordinal')
        else:
            # convert all 'integer' to 'numeric'
            if dt == 'integer':
                dt = 'numeric'
            data_types.append(dt)

        co = row['val_range_orig']
        if dt in ['categorical', 'ordinal']:
            categorical_options.append(co)
        else:
            categorical_options.append('')

        repres = get_col_name_in_pooled(col_name, pool)

        # number of missing entries
        num_missing = pool[repres].isna().sum()
        perc_missing = num_missing / len(pool) * 100
        missing.append(perc_missing)

        # distribution
        hyperlink = 'https://bitbucket.org/HiyamGh/dementia/src/master/distributions/{}.png'.format(repres)
        dists.append(hyperlink)

        # range of values
        vr_act = list(set(pool[repres].dropna().values))
        # if col_name == 'CJOBCAT_2':
        #     vr_act = [i for i in vr_act if i.isnumeric()]
        # if 'CKINCOM' not in repres:
        #     vr_act = list(map(int, vr_act))
        vr_act = sorted(vr_act)

        # get min and max
        if vr_act:
            minv = min(vr_act)
            maxv = max(vr_act)
            if dt == 'numeric':
                val_range.append('{}-{}'.format(minv, maxv))
            else:
                val_range.append(','.join(map(str, vr_act)))

            # erroneous data
            errors = []
            num_err = 0
            if dt in ['numeric', 'ordinal']:
                # get 20% difference between min and max
                diff = ((maxv - minv) * 5)/100
                for val in vr_act:
                    if val >= diff:
                        errors.append(val)
                        num_err += 1
        else:
            minv = 'not found'
            maxv = 'not found'
            val_range.append('not found')

        min_vals.append(minv)
        max_vals.append(maxv)

    merged['description'] = descs
    merged['data_type'] = data_types
    merged['pmissing'] = missing
    merged['distribution'] = dists
    merged['val_range_orig'] = categorical_options
    merged['val_range_act'] = val_range
    merged['min'] = min_vals
    merged['max'] = max_vals
    merged['nb_erroneous'] = np.nan
    merged['erroneous'] = np.nan
    merged = merged.drop(['feature_binning', 'notes'], axis=1)

    ordered = ['COLUMN', 'data_type', 'description', 'val_range_orig', 'val_range_act', 'min', 'max', 'pmissing', 'nb_erroneous','erroneous','distribution']
    merged = merged[ordered]
    merged = merged.rename(columns={'val_range_orig': 'cat_options', 'val_range_act': 'val_range'})
    merged = merged.style.format({'distribution': make_clickable})

    dest = '../input/features_data/'
    check_create_dir(dest)

    merged.to_excel(os.path.join(dest, 'numeric.xlsx'), index=False)

