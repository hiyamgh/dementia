import pandas as pd
from collections import Counter
import pickle

# Comments by Dr. Khalil
# encode values 0, 0.5 to being not demented (negative class)
# encode values 1, 2 to being demented (positive class)

if __name__ == '__main__':
    df = pd.read_csv('input/full validation data (281).csv')
    print(set(df['CRD']))
    crd = list(df['CRD'])
    crd_encoded = []
    for val in crd:
        if val in [0, 0.5]:
            crd_encoded.append(0)
        else:
            crd_encoded.append(1)

    # sanity checks
    print(set(crd_encoded))
    print(Counter(crd_encoded))
    print(len(crd_encoded))

    # save
    with open('input/y_test_crd.pkl', 'wb') as f:
        pickle.dump(crd_encoded, f)