from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,\
    AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

shallow_models = {
    # Trees
    'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
    # 'DT': DecisionTreeClassifier(max_depth=10),
    # 'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),

    # Boosting
    # 'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=200),
    'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
    'BG': BaggingClassifier(),
    'XGB': XGBClassifier(),

    # Other
    'LR': LogisticRegression(C=1e5),
    # 'GNB': GaussianNB(),
    # 'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
}
