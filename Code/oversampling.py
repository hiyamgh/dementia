from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTENC
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

def random_sampling(X,y,model):
    over = RandomOverSampler(sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.5)
    pipeline = Pipeline(steps=[('o', over), ('u', under), ('m', model)])
    return pipeline

def smote_sampling(X,y,model):
    # define pipeline
    over = SMOTENC(sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.5)
    steps = [('o', over), ('u', under), ('m', model)]
    pipeline = Pipeline(steps=steps)
    return pipeline

def smote_tomek(X,y,model):
    resample = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
    pipeline = Pipeline(steps=[('r', resample), ('m', model)])
    return pipeline
 
def smote_enn(X,y,model):
    resample = SMOTEENN()
    pipeline = Pipeline(steps=[('r', resample), ('m', model)])
    return pipeline