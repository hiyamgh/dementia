
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
                # print('col {} in mapping, transformed to {}'.format(col, mapping[col]))
                col_temp = mapping[col]
                if col_temp in pooled.columns:
                    # print('{} in pooled'.format(col_temp))
                    col_pooled = col_temp
                else:
                    raise ValueError('col_temp {} not found in pooled'.format(col_temp))
            elif col in mapping_inv:
                # print('col {} in mapping, transformed to {}'.format(col, mapping_inv[col]))
                col_temp = mapping_inv[col]
                if col_temp in pooled.columns:
                    # print('{} in pooled'.format(col_temp))
                    col_pooled = col_temp
                else:
                    # raise ValueError('col_temp {} not found in pooled'.format(col_temp))
                    # if the column, after transformation, is still not found
                    print('col_temp {} not found in pooled'.format(col_temp))
                    return -1

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
