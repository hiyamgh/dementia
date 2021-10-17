import pandas as pd

tops = [10, 20]
for tp in tops:
    df = pd.read_csv('results_errors/final_results_top{}.csv'.format(tp))
    non_overfitting = []
    for index, row in df.iterrows():
        if index + 1 < len(df):
            row_bef = df.iloc[[index]]
            row_aft = df.iloc[[index + 1]]
            if row_bef['model'].item() != '' and row_aft['model'].item() != '':
                f2_bef, f2_aft = float(row_bef['f2'].item()), float(row_aft['f2'].item())
                bss_bef, bss_aft = float(row_bef['bss'].item()), float(row_aft['bss'].item())
                gmean_bef, gmean_aft = float(row_bef['gmean'].item()), float(row_aft['gmean'].item())

                if f2_bef < f2_aft and bss_bef < bss_aft and gmean_bef < gmean_aft:
                    non_overfitting.append((row_bef['model'].item(), index))

    with open('results_errors/non_overfitting_top{}.txt'.format(tp), 'w') as f:
        for nno in non_overfitting:
            f.writelines(str(nno) + '\n')

