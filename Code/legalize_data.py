"""
This code is for doing the following

1. Remove all informants and their children
2. Rest of the legal columns:
    a. remove all children
    b. keep all the rest (mix of parents and those that are neither parents nor children)
    c. NOTE: Some parents were actually children for other columns. So we made sure we removed
             these as well.

=============================================================================
1. Filter dropped columns by excluding columns related to informants as well as columns that
trigger a skip or jump to another question.

2. Order these columns by increasing percentage of missing values. Send us the sort list with
a description of the questions. Need to have items 1 and 2 in two days time if possible please.


=============================================================================
Regarding the missing:

Drop all the columns that have missing than 40%.
This will automatically drop all the variables with any missing

Regarding the erroneous:

1. For animals_2 replace all the values by 1 (i.e: yes)
2. Q1181_2, Q21_2, Q53_2, Q54_2: Can you please give me the frequency of 8 and 9 seperately,
because we may need to impute those with response=8.
Alternatively we may regretfully have to drop this column.

3. For all the rest, drop all other observations that has erroneous values.
These are mainly not applicable or no answer.
Given the small frequency numbers, these should not compromise our sample size a lot.
"""

import pandas as pd
import numpy as np
from numpy import percentile
import os
from utils import get_col_name_in_pooled
from collections import Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor


def check_create_dir(dest):
    """ creates the directory if it doesn't exist already """
    if not os.path.exists(dest):
        os.makedirs(dest)


def drop_erroneous(pooled,numeric):
    """
    For all the rest, drop all other observations that has erroneous values.
    These are mainly not applicable or no answer.
    Given the small frequency numbers, these should not compromise our sample size a lot.
    """
    drop_indexes=np.array([])
    for index,row in pooled.iterrows():
        for _,r in numeric.iterrows():
            if r['COLUMN'] != 'ANIMALS_2':
                if (str(row[r['COLUMN']]) in str(r['erroneous'])) and (str(row[r['COLUMN']]) not in ['8','9','97','98','99']):
                    drop_indexes=np.append(drop_indexes,index)
    
    drop_indexes=np.unique(drop_indexes)
    pooled=pooled.drop(drop_indexes)
    return pooled

def process_animals2(merged, pooled):
    """
    So from what I understand, ANIMALS_2 should be if the patient was able to remember the animals
    correctly or not, thus values are either 0 or 1, but we had values from 3 till 40 which we
    considered as erroneous, thus, we have to replace all these by 1 as recommended by Dr. Khalil ?
    """

    # first, we must change the values inside the main df: pooled_data.csv
    repres = get_col_name_in_pooled('ANIMALS_2', pooled)
    pooled.loc[pooled[repres] > 1, repres] = 1
    print('set of values for {}: {}'.format(repres, set(pooled[repres].dropna())))

    # TODO then we must change the entry for 'ANIMALS_2' in numeric dataframe (merged)

    return pooled, merged


def drop_informant_columns():
    """
    Filter dropped columns by excluding columns related to informant
    """
 
    def get_perc_missing(row):
        """ get the % of missing values for a certain column """
        # repres = get_col_name_in_pooled(row['COLUMN_NAME'], pooled)
        col=row['COLUMN_NAME']
        num_missing = pooled[col].isna().sum()
        perc_missing = num_missing / len(pooled) * 100
        row['perc_missing']=perc_missing
        row['INFORMANT']=True
        return row

    household=pd.read_excel('../input/codebooks/household_questions.xlsx')
    household=household.apply(lambda row: get_perc_missing(row),axis=1)
   
    return household


def drop_jumping_columns(beqaa_df, merged, pooled, household_df):
    """ Filter dropped columns by excluding columns
    that trigger a skip or jump to another question.
    :param beqaa_df: copy of dementia data frame
    :param merged: the 'numeric' data frame we are building
    :param pooled: the pooled data
    :param household_df: the dataset that contains all informants
    """

    # instantiate the main columns
    beqaa_cols = list(beqaa_df['name'])
    informant_cols = list(household_df['COLUMN_NAME'])
    beqaa_df = beqaa_df.set_index('name')

    def get_perc_missing(col):
        """ get the % of missing values for a certain column """
        repres = get_col_name_in_pooled(col, pooled)
        if repres in pooled.columns:
            if isinstance(repres, str):
                # get the percentage of missing data
                num_missing = pooled[repres].isna().sum()
                perc_missing = num_missing / len(pooled) * 100

                return perc_missing

        return -1

    def is_relevant(col_name):
        '''
        it is relevant if and only if the 'relevant' column in beqaa
        questionnaire is not empty or nan AND it is not an informant column,
        then, this feature is a parent feature that causes jump to child features
        '''
        if col_name in beqaa_cols:
            # if col_name not in informant_columns:
            if col_name not in informant_cols:
                rel = str((beqaa_df.loc[col_name, 'relevant']))
                if isinstance(rel, str):
                    if rel and rel != 'nan':
                        return True
            # else:
            #     print('{} is an informant'.format(col_name))
        else:
            print('{} is not in beqaa questionnaire'.format(col_name))
        return False

    def clean_string(mystr):
        ''' removes un-necessary characters found in a relation ('relevant' column in copy of dementia) '''
        unwanted = [" ", "$", "{", "}", "\\", "\'"]
        for c in unwanted:
            mystr = mystr.replace(c, '')

        if 'and' in mystr:
            mystr = mystr.replace('and', ' and ')

        if 'or' in mystr:
            mystr = mystr.replace('or', ' or ')

        return mystr

    def get_parents(rel_string):
        ''' gets the parent column
            agedisc has: agepart!= 999 and ageinfo!= 999 and agedoc}!= 999
            agepart parent --> child1: agedisc
            ageinfo parent --> child1: agedisc
            agedoc parent --> child1: agedisc
        '''

        parent_names = []
        if 'and' in rel_string:
            parents = rel_string.split(' and ')
            for possible_parent in parents:
                parent_names.append(possible_parent[:possible_parent.index('!') if '!' in possible_parent else
                                    possible_parent.index('=')])

        elif 'or' in rel_string:
            parents = rel_string.split(' or ')
            for possible_parent in parents:
                parent_names.append(possible_parent[:possible_parent.index('!') if '!' in possible_parent else
                                    possible_parent.index('=')])
        else:
            parent_names.append(rel_string[:rel_string.index('!') if '!' in rel_string else
            rel_string.index('=')])

        return parent_names

    def update_parents2child(child, parents, rel_string, parents2childdic):
        ''' creates and updates parents-to-children dictionary of relations '''
        for parent in parents:
            # if dictionary is not empty
            if parent not in parents2childdic:
                parents2childdic[parent] = {
                    'children': [child],
                    'rels': [rel_string]
                }
            else:
                if child not in parents2childdic[parent]['children']:
                    parents2childdic[parent]['children'].append(child)
                    parents2childdic[parent]['rels'].append(rel_string)

        return parents2childdic

    def get_description(col):
        return beqaa_df.loc[col, 'label::English']

    parent2child = {}
    legal_cols = [col for col in merged['COLUMN'] if is_relevant(col_name=col)]
    legal_rest = [col for col in merged['COLUMN'] if col not in informant_cols]

    for col in legal_cols:
        # get the relation and clean it
        rel = str((beqaa_df.loc[col, 'relevant']))
        rel_cleaned = clean_string(rel)

        # get the column's parents
        parents = get_parents(rel_string=rel_cleaned)

        parent2child = update_parents2child(child=col, parents=parents,
                                            rel_string=rel_cleaned, parents2childdic=parent2child)

    # relations tree between parents and children
    relations = pd.DataFrame(columns=['parent', 'parent_perc_missing', 'parent_desc',
                                      'child', 'child_perc_missing', 'child_desc',
                                      'relation2parent'])
    for parent in parent2child:
        count = 0
        for idx, child in enumerate(parent2child[parent]['children']):
            relations = relations.append({
                'parent': parent if count == 0 else '',
                'parent_perc_missing': get_perc_missing(col=parent) if count == 0 else '',
                'parent_desc': get_description(col=parent) if count == 0 else '',
                'child': child,
                'child_perc_missing': get_perc_missing(col=child),
                'child_desc': get_description(col=child),
                'relation2parent': parent2child[parent]['rels'][idx]
            }, ignore_index=True)

            count += 1

    children_final = []
    for parent in parent2child:
        curr_children = parent2child[parent]['children']
        children_final = children_final + curr_children

    relationsdf = pd.read_csv('../input/codebooks/relations.csv')
    children_column = list(relationsdf['child'])

    # both below evaluate to the same thing
    legal_final = set(legal_rest) - set(children_final)
    # legal_final = set(legal_rest) - set(children_column)
    # print(set(legal_rest) - set(children_column) == set(legal_rest) - set(children_final)) # True

    # perform one last sanity check:
    for col in legal_final:
        if col in informant_cols:
            print('WARNING: {} in informants'.format(col))
        if col in children_column:
            print('WARNING: {} in children'.format(col))

    print('len of legal_final: {}'.format(len(legal_final)))
    print('len of informants: {}'.format(len(informant_cols)))
    print('len of children: {}'.format(len(set(children_final))))
    print((len(legal_final) + len(informant_cols) + len(children_final)) == len(numeric))

    # remove DAY_2 because it has 50% missing and its not important
    legal_final.remove('DAY_2')

    return relations, legal_final


def get_freq(pooled):
    """
    Q1181_2, Q21_2, Q53_2, Q54_2: Can you please give me the
    frequency of 8 and 9 seperately,
    because we may need to impute those with response=8.
    """

    cols = ['Q1181_2', 'Q21_2', 'Q53_2', 'Q54_2']

    for col in cols:
        repres = get_col_name_in_pooled(col, pooled)

        values = list(pooled[repres])
        fr8 = values.count(8)
        fr9 = values.count(9)

        print('{}: 8: {}, 9: {}'.format(col, fr8, fr9))


def drop_missing(merged, pooled, missing_codebook):
    """
    drops - from numeric.xlsx (merged) - features with > 40% missing
    returns: the new numeric (merged) after dropping, plus the new missing codebook (all features that have
    missing)
    """
    # list of columns that have missing values > 40%
    cols = list(missing_codebook['Unnamed: 0'])

    # numeric data frame without the columns having > 40% missing
    len_orig = len(merged)
    merged = merged[~ merged.COLUMN.isin(cols)]
    len_new = len(merged)
    print('size of the numeric dataset decreased from {} to {}'.format(len_orig, len_new))

    # check if there are still features with missing values, if yes, how many of them
    mc = pd.DataFrame(columns=['perc_missing'])
    for _, row in merged.iterrows():
        # column name in merged
        col_name = row['COLUMN']

        # column name in pooled data
        repres = get_col_name_in_pooled(col_name, pooled)

        if isinstance(repres, str):
            # get the percentage of missing data
            num_missing = pooled[repres].isna().sum()
            perc_missing = num_missing / len(pooled) * 100

            mc.loc[col_name] = pd.Series({
                'perc_missing': perc_missing
            })

    # sort the missing codebook by ascending order of missing percentages
    mc = mc.sort_values(by=['perc_missing'])

    # some statistics: percentage of features with missing values:
    print('number of features with missing values: {} ~ {}% of the numeric features'.format(len(mc),
                                                                                            (len(mc)/len(merged)) * 100))

    return merged, mc


def keep_legal(numeric, legal_cols, pooled, output_folder):
    ''' saves the legal columsn as a '''
    numeric_copy = numeric
    numeric_copy = numeric_copy[numeric_copy['COLUMN'].isin(legal_cols)]

    # sort by percentage of missing
    # numeric = numeric.sort_values(by='perc_missing', ascending=False)
    # I will re-compute the percentage of missing
    allcols = list(numeric_copy['COLUMN'])
    perc_missing = []
    for col in allcols:
        pm = (pooled[col].isna().sum() / len(pooled)) * 100
        perc_missing.append(pm)
    numeric_copy['perc_missing'] = perc_missing
    numeric_copy = numeric_copy.sort_values(by='perc_missing', ascending=False)
    check_create_dir(dest=output_folder)
    numeric_copy.to_csv(os.path.join(output_folder, 'legal.csv'), index=False)
    print('\nNumber of legal cols: {}'.format(len(numeric_copy)))
    print('Number of legal cols with missing > 40%: {}'.format(len(numeric_copy[numeric_copy['perc_missing'] >= 40])))


def legalize_erroneous(legal_cols, erroneous_df, output_folder, copyofdem):
    ''' function to keep only the legal columns in erroneous code book '''
    erroneous_df = erroneous_df.rename(columns={'Unnamed: 0': 'COLUMN'})
    erroneous_df = erroneous_df[erroneous_df['COLUMN'].isin(legal_cols)]
    erroneous_df = erroneous_df.sort_values(by=['perc_erroneous'], ascending=False)
    erroneous_df = erroneous_df.apply(lambda row: set_description(row, copyofdem), axis=1)
    check_create_dir(dest=output_folder)
    erroneous_df.to_csv(os.path.join(output_folder, 'erroneous_codebook_legal.csv'), index=False)


def set_description(row, copyofdem):
    print(row['COLUMN'])
    vals=copyofdem.loc[(copyofdem['name'] == row['COLUMN'])]['label::English'].values
    if len(vals) > 0:
        row['description']=vals.item()
    return row


def calculate_outliers_zscore(pooled, col):
    data = pooled[col]
    upper = data.mean() + 3*data.std()
    lower = data.mean() - 3*data.std()

    outliers = [x for x in pooled[col] if x < lower or x > upper]
    perc_outliers = (len(outliers) / len(data)) * 100
    return perc_outliers

    # count = 0
    # outliers = []
    # for val in pooled[col]:
    #     if val < lower or val > upper:
    #         count += 1
    #         outliers = np.append(outliers, val)
    #
    # outliers = np.unique(outliers)
    # outlier_str = [str(outlier) for outlier in outliers]
    # outlier_vals=' '.join(outlier_str)
    #
    # return count, outlier_vals


def calculate_outliers_iqr(pooled, col):
    data = pooled[col]
    q25, q75 = percentile(data, 25), percentile(data, 75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off

    outliers = [x for x in data if x < lower or x > upper]
    perc_outliers = (len(outliers)/len(data)) * 100
    return perc_outliers


def identify_outliers(pooled):
    ''' identify outliers using 3 ways
        all legal columns have 0% missing (with the exception of DAY_2
        which upon Dr. Khalil's recommendation, he said its safe
        to discard this column, so we did.)
    '''

    # df_numeric = pd.read_csv('../input/codebooks/numeric_new.csv')
    df_numeric = pd.read_excel('../input/codebooks/numeric.xlsx')
    erroneous_codebook = pd.read_csv('../input/codebooks/erroneous_codebook_legal.csv')
    # pooled = pd.read_csv('../input/pooled_new.csv')
        # try:
        #     count,outlier_vals=calculate_outliers(pooled,col)
        #     erroneous_codebook.loc[erroneous_codebook['COLUMN'] == col,'%_outliers']=count/len(pooled)*100
        #     erroneous_codebook.loc[erroneous_codebook['COLUMN'] == col,'outlier_vals']=outlier_vals
        #
        #     count,outlier_vals=calculate_outliers(pooled_scaled,col)
        #     erroneous_codebook.loc[erroneous_codebook['COLUMN'] == col,'scaled_%_outliers']=count/len(pooled_scaled)*100
        #     erroneous_codebook.loc[erroneous_codebook['COLUMN'] == col,'scaled_outlier_vals']=outlier_vals
        #
        # except Exception as e:
        #     print(e)
    
    # no outliers exist in categorical data types.
    # we can consider outliers as being the 'erroneous values'
    # are there outliers in ordinal data ?

    # for an ordinal variable corresponding to a ranking,
    # no unit can be considered as an outlier, because the
    # observations take on values (ranks) from 1 to n.
    # In an ordered categorical variable with k levels,
    # a unit may have each of k, a priori, defined categories
    # and therefore no outlier could be detected. However, in
    # a few special cases the frequency distribution of a variable
    # may show univariate outliers.

    # all legal columns are either categorical or ordinal (we have no numeric
    # and the book deals with outliers in numeric data) types
    # we will print the frequency of all unique variables
    # the ones with the lowest frequency are either normally
    # like this or outliers

    # dictionary mapping each column to its data type
    columns2type = dict(zip(df_numeric['COLUMN'], df_numeric['data_type']))
    col_types = []
    frequencies = []
    perc_misisng = []
    outliers_zscore, outliers_iqr = [], []
    for col in list(erroneous_codebook['COLUMN']):
        type = columns2type[col]
        col_types.append(type)
        counter = Counter(list(pooled[col].dropna().values))
        mycount = ''
        for k, v in counter.items():
            mycount += ', {}: {}%%'.format(k, v/len(pooled[col] * 100))
        # frequencies.append(str(Counter(list(pooled[col].dropna().values))))
        frequencies.append(mycount)

        # percentage of missing values
        pm = (pooled[col].isna().sum() / len(pooled)) * 100
        # perc_misisng.append(str(pm) + '%')
        perc_misisng.append(pm)

        # outliers (only for numeric & ordinal)
        if type in ['ordinal', 'numeric']:
            outliers_zscore.append(calculate_outliers_zscore(pooled, col))
            outliers_iqr.append(calculate_outliers_iqr(pooled, col))
        else:
            outliers_zscore.append('')
            outliers_iqr.append('')

    # create a new column for data types in the erroneous code book
    erroneous_codebook.insert(1, 'data_type', col_types)
    erroneous_codebook.insert(2, 'frequencies', frequencies)
    erroneous_codebook.insert(3, 'perc_missing', perc_misisng)
    erroneous_codebook.insert(4, 'outliers_zscore', outliers_zscore)
    erroneous_codebook.insert(5, 'outliers_iqr', outliers_zscore)

    # sort by decreasing % missing
    erroneous_codebook = erroneous_codebook.sort_values(by='perc_missing', ascending=False)
    erroneous_codebook = erroneous_codebook[['COLUMN', 'data_type', 'description', 'frequencies',
                                             'perc_missing',
                                             'erroneous', 'perc_erroneous', 'cut_off',
                                             'outliers_zscore', 'outliers_iqr']]
    # COLUMN,data_type,frequencies,perc_missing,outliers_zscore,outliers_iqr,cut_off,
    # description,erroneous,perc_erroneous
    erroneous_codebook.to_csv('../input/codebooks/erroneous_codebook_legal_outliers.csv', index=False)

    # remove columns with > 40% missing
    erroneous_codebook = erroneous_codebook[erroneous_codebook['perc_missing'] <= 40]
    erroneous_codebook.to_csv('../input/codebooks/erroneous_codebook_legal_outliers_filtered.csv', index=False)

    print('\nMax outlier %%: {}%%'.format(min(erroneous_codebook[erroneous_codebook['outliers_zscore'] != '']['outliers_zscore'])))
    print('\nMax outlier %%: {}%%'.format(max(erroneous_codebook[erroneous_codebook['outliers_zscore'] != '']['outliers_zscore'])))


def normalize_data(df, cols_to_scale):
    # cols_not_to_scale = list(set(df.columns) - set(cols_to_scale))
    scaler = MinMaxScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df


def standardize_data(df, cols_to_scale):
    # cols_not_to_scale = list(set(df.columns) - set(cols_to_scale))
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df


def _get_ordinal(df):
    ''' Gets the list of ordinal columns '''
    df = df[df['data_type'] == 'ordinal']
    return list(df['COLUMN'])


def _get_categorical(df):
    ''' Gets the list of categorical columns '''
    df = df[(df.data_type == 'categorical') & (df.data_type == 'boolean')]
    return list(df['COLUMN'])


def _get_numeric(df):
    ''' Gets the list of numeric columns '''
    df = df[df['data_type'] == 'numeric']
    return list(df['COLUMN'])


def impute_missing_values(data):
    ''' simple imputation strategy that works for categorical '''
    print('before imputation: Nans: {}'.format(sum(data.isna().sum())))
    # fillna by the most frequent values
    imputed_data = data.fillna(data.mode().iloc[0])
    print('after imputation: Nans: {}'.format(sum(imputed_data.isna().sum())))
    return imputed_data


def create_pooled_legal_filtered(pooled, filtered_codebook):
    ''' filter pooled by legal columns after removing legal columns having > 40% missing '''
    print('pooled data dim - before filtering: {}'.format((len(pooled), len(pooled.columns))))
    cols_legal_filtered = filtered_codebook['COLUMN']
    pooled = pooled[list(cols_legal_filtered) + ['dem1066']]
    print('pooled data dim - after filtering: {}'.format((len(pooled), len(pooled.columns))))
    pooled.to_csv('../input/pooled_data_filtered.csv', index=False)


if __name__ == '__main__':
    path = '../input/codebooks/'
    mc = pd.read_csv(os.path.join(path, 'missing_40_codebook.csv'))
    numeric = pd.read_excel(os.path.join(path, 'numeric.xlsx'))
    pooled = pd.read_csv('../input/pooled_data.csv')
    household = pd.read_excel('../input/codebooks/household_questions.xlsx')
    copyofdem = pd.read_excel('../input/Copy of Dementia_baseline_questionnaire_V1.xlsx')
    erroneous = pd.read_csv('../input/codebooks/erroneous_codebook.csv')

    of = '../input/codebooks/'
    relations, legal_cols = drop_jumping_columns(beqaa_df=copyofdem, merged=numeric,
                                                 pooled=pooled, household_df=household)
    keep_legal(numeric, legal_cols, pooled, output_folder=of)
    print('===============================================================')
    legalize_erroneous(legal_cols=legal_cols, erroneous_df=erroneous, output_folder=of,copyofdem=copyofdem)

    #  ============================== NEW PART STARTS BELOW =============================================
    identify_outliers(pooled)

    filtered = pd.read_csv('../input/codebooks/erroneous_codebook_legal_outliers_filtered.csv')
    pooled_imputed = impute_missing_values(data=pooled)
    create_pooled_legal_filtered(pooled_imputed, filtered)
