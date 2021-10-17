import pandas as pd
import os

# on the validation data
dirdata2nameval = {
    'Reptile_validation_data/results_errors/final_results_top10.csv': {
        'name': 'FOMAML - trans - top 10',
        'idx': 699
    },
    'Reptile_validation_data/results_errors/final_results_top20.csv': {
        'name': 'FOMAML - trans - top 20',
        'idx': 60
    }
}

# foml trans top 10: ('23111_test', 699)
# foml trans top 20: ('39838_test', 60)

# on the original (non-validation) data
dirdata2name = {
    'maml_dementia_nodataleakage/results_errors/dementia_without_fp_top10_f2.csv': {
        'name': 'MAML - top 10',
    },
    'maml_dementia_nodataleakage/results_errors/dementia_without_fp_top20_f2.csv': {
        'name': 'MAML - top 20',
    },

    'Reptile_orig_data/results_errors/final_results_top10_reptile.csv': {
        'name': 'Reptile - top 10',
        'idx': 12
    },
    'Reptile_orig_data/results_errors/final_results_top20_reptile.csv': {
        'name': 'Reptile - top 20',
        'idx': 0
    },

    'Reptile_orig_data/results_errors/final_results_top10_reptile_trans.csv': {
        'name': 'Reptile - trans - top 10',
        'idx': 0
    },

    'Reptile_orig_data/results_errors/final_results_top20_reptile_trans.csv': {
        'name': 'Reptile - trans - top 20',
        'idx': 3
    },

    'Reptile_orig_data/results_errors/final_results_top10_foml_trans.csv': {
        'name': 'FOMAML - trans - top 10',
        'idx': 0
    },
    'Reptile_orig_data/results_errors/final_results_top20_foml_trans.csv': {
        'name': 'FOMAML - trans - top 20',
        'idx': 0
    }
}

# foml top 10: ('831_test', 0)
# foml top 20: ('196_test', 0)
# reptile trans top 10: ('863_test', 0)
# reptile trans top 20: ('780_test', 3)
# reptile top 10: ('1313_test', 12)
# reptile top 20: ('1576_test', 0)


def compile_results(dir2names):
    df_all_results = pd.DataFrame(
        columns=['experiment', 'model', 'f2', 'gmean', 'bss', 'pr_auc', 'sensitivity', 'specificity'])
    for k, v in dir2names.items():
        df = pd.read_csv(k)
        if 'idx' in v:
            idx = v['idx']
        else:
            idx = 0
        name = v['name']
        row = df.iloc[[idx]]
        model = row['model'].item()
        f2 = row['f2'].item()
        gmean = row['gmean'].item()
        bss = row['bss'].item()
        prauc = row['pr_auc'].item()
        sens = row['sensitivity'].item()
        spec = row['specificity'].item()

        df_all_results = df_all_results.append({
            'experiment': name,
            'model': model,
            'f2': f2,
            'gmean': gmean,
            'bss': bss,
            'pr_auc': prauc,
            'sensitivity': sens,
            'specificity': spec
        }, ignore_index=True)

    return df_all_results


if __name__ == '__main__':

    df_all_results = compile_results(dirdata2name)
    df_all_results.drop(['model'], axis=1).to_csv('all_results_original.csv', index=False) # save without model name
    df_all_results.to_csv('all_results_original_with_model_name.csv', index=False) # save with model name

    df_all_results = compile_results(dirdata2nameval)
    df_all_results.drop(['model'], axis=1).to_csv('all_results_validation.csv', index=False)  # save without model name
    df_all_results.to_csv('all_results_original_with_model_validation.csv', index=False)  # save with model name
