import pandas as pd

features = pd.DataFrame(columns=['COLUMN', 'name_type', 'data_type', 'val_range', 'val_map', 'min', 'max', 'feature_binning', 'distribution'])
df = pd.read_csv('../input/pooled_data.csv')
copyofdem = pd.read_excel('../input/Copy of Dementia_baseline_questionnaire_V1.xlsx')
copyofdemcols = list(copyofdem['name'].values)
features['COLUMN'] = list(df.columns)

# print(set(df['contact_note'].values))
missing = []
val_ranges = []
name_types = []
mins, maxs = [], []
for col in list(df.columns):
    # print('processing {}'.format(col))
    if col not in copyofdemcols:
        print('col {} not found in copy of dem'.format(col))
        name_types.append('not found')
    else:
        target_row = copyofdem[copyofdem['name'] == col]
        nt = target_row['type'].values[0]
        name_types.append(nt)
    if '_note' not in col and col != 'CINCOME3':
        val_range = list(set(sorted(list(df[col].dropna().values))))
        # print('{}: {}'.format(col, val_range))

        yes = True
        temp_list = []
        for v in val_range:
            try:
                temp_list.append(int(v))
            except ValueError:
                yes = False
                break
        if yes:
            # val_range = [int(v) for v in val_range]
            # if len(val_range) == (max(val_range) - min(val_range)) + 1:
            if len(val_range) != 0:
                mn = min(val_range)
                mx = max(val_range)
            vr = '{}-{}'.format(mn, mx)
            val_ranges.append(vr)
            mins.append(mn)
            maxs.append(mx)
            # else:
            #     val_range = ', '.join(map(str, val_range))
        else:
            vr = ', '.join(map(str, val_range))
            val_ranges.append(vr)
            mins.append(' ')
            maxs.append(' ')

        num_missing = df[col].isna().sum()
        perc_missing = (num_missing / len(df)) * 100
        missing.append(perc_missing)
    else:
        print(col)
        val_ranges.append(' ')
        mins.append(' ')
        maxs.append(' ')

        # print('{}: {}'.format(col, perc_missing))

features['val_range'] = val_ranges
features['min'] = mins
features['max'] = maxs
features['name_type'] = name_types



features.to_csv('features_meta.csv',index=False,encoding='utf-8-sig')

nametypes = set(name_types)
for nt in nametypes:
    print(nt)