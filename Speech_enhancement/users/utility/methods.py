import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, KFold, train_test_split
from scipy.stats import uniform, randint
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
import xgboost
from xgboost import XGBClassifier
from django.conf import settings

path = settings.MEDIA_ROOT + '//' + 'PDFMalware2022.parquet'
df = pd.read_parquet(path)

df.head()

df['Class'].value_counts()
#df['Class'] = df['Class'].replace(['Benign','Malicious'],[0,1])
df.drop(columns=['FileName'], inplace=True)


dep = 'Class'
cats = df.select_dtypes(include='category').columns
conts = df.columns.difference([dep]+list(cats))
def different_methods():
    def xs_y(df_, targ):    
        if not isinstance(targ, list):
            xs = df_[df_.columns.difference([targ])].copy()
        else:
            xs = df_[df_.columns.difference(targ)].copy()
        y = df_[targ].copy()
        return xs,y
    from sklearn.model_selection import train_test_split
    trn_df, val_df = train_test_split(df, test_size=0.30)
    val_df, test_df = train_test_split(df, test_size=0.75)
    trn_df[cats] = trn_df[cats].apply(lambda x: x.cat.codes)
    val_df[cats] = val_df[cats].apply(lambda x: x.cat.codes)
    test_df[cats] = test_df[cats].apply(lambda x: x.cat.codes)

    X,y = xs_y(trn_df, dep)
    print(y)


    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


    from imblearn.over_sampling import RandomOverSampler

    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X_train,y_train)



    xgb = XGBClassifier(
        n_estimators= 100,
        use_label_encoder= False,
        max_depth= 8,
        booster= 'gbtree',
        tree_method= 'hist',
        subsample= 0.5,
        colsample_bytree= 0.5,
        importance_type= 'gain',
        objective='binary:logistic',
        eval_metric='logloss',
        predictor= 'cpu_predictor',
        n_jobs= -1)

    xgb.fit(X_res,y_res)

    preds = xgb.predict(X_test)
  
    average_precision1 = average_precision_score(y_test,preds)

    roc_auc_score1 =  roc_auc_score(y_test,preds)
    accuracy_score1 = accuracy_score(y_test,preds)
    precision_score1 = precision_score(y_test,preds)
    recall_score1 = recall_score(y_test,preds)
    matthews_corrcoef1 = matthews_corrcoef(y_test,preds)

    from imblearn.over_sampling import SMOTE

    sm = SMOTE(random_state=42)
    X_smote, y_smote = sm.fit_resample(X,y)

    X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.33, random_state=42)

    xgb = XGBClassifier(
        n_estimators= 100,
        use_label_encoder= False,
        max_depth= 8,
        booster= 'gbtree',
        tree_method= 'hist',
        subsample= 0.5,
        colsample_bytree= 0.5,
        importance_type= 'gain',
        objective='binary:logistic',
        eval_metric='logloss',
        predictor= 'cpu_predictor',
        n_jobs= -1)

    xgb.fit(X_smote, y_smote)

    preds = xgb.predict(X_test)
  

    average_precision2 = average_precision_score(y_test,preds)

    roc_auc_score2 =  roc_auc_score(y_test,preds)
    accuracy_score2 = accuracy_score(y_test,preds)
    precision_score2 = precision_score(y_test,preds)
    recall_score2 = recall_score(y_test,preds)
    matthews_corrcoef2 = matthews_corrcoef(y_test,preds)


    from sklearn.ensemble import RandomForestClassifier

    random = RandomForestClassifier()

    from imblearn.over_sampling import SVMSMOTE
    sm = SVMSMOTE(random_state=42)
    X_ssmote, y_ssmote = sm.fit_resample(X,y)

    X_train, X_test, y_train, y_test = train_test_split(X_ssmote, y_ssmote, test_size=0.33, random_state=42)

    random.fit(X_train,y_train)

    preds = random.predict(X_test)

    average_precision3 = average_precision_score(y_test,preds)

    roc_auc_score3 =  roc_auc_score(y_test,preds)
    accuracy_score3 = accuracy_score(y_test,preds)
    precision_score3 = precision_score(y_test,preds)
    recall_score3 = recall_score(y_test,preds)
    matthews_corrcoef3 = matthews_corrcoef(y_test,preds)

    from imblearn.over_sampling import ADASYN

    ada = ADASYN(random_state=42)
    X_ada, y_ada = ada.fit_resample(X,y)

    X_train, X_test, y_train, y_test = train_test_split(X_ada, y_ada, test_size=0.33, random_state=42)

    random.fit(X_train,y_train)

    preds = random.predict(X_test)
 
    average_precision4 = average_precision_score(y_test,preds)

    roc_auc_score4 =  roc_auc_score(y_test,preds)
    accuracy_score4 = accuracy_score(y_test,preds)
    precision_score4 = precision_score(y_test,preds)
    recall_score4 = recall_score(y_test,preds)
    matthews_corrcoef4 = matthews_corrcoef(y_test,preds)


    from imblearn.under_sampling import CondensedNearestNeighbour

    undersample = CondensedNearestNeighbour(n_neighbors=1)
    X_cnn, y_cnn = undersample.fit_resample(X,y)

    X_train, X_test, y_train, y_test = train_test_split(X_cnn, y_cnn, test_size=0.33, random_state=42)

    random.fit(X_train,y_train)

    preds = random.predict(X_test)

    average_precision5 = average_precision_score(y_test,preds)

    roc_auc_score5 =  roc_auc_score(y_test,preds)
    accuracy_score5 = accuracy_score(y_test,preds)
    precision_score5 = precision_score(y_test,preds)
    recall_score5 = recall_score(y_test,preds)
    matthews_corrcoef5 = matthews_corrcoef(y_test,preds)

    from imblearn.pipeline import Pipeline
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import cross_validate

    from imblearn.under_sampling import EditedNearestNeighbours
    undersample = EditedNearestNeighbours(n_neighbors=3)
    X_enn, y_enn = undersample.fit_resample(X, y)


    X_train, X_test, y_train, y_test = train_test_split(X_enn, y_enn, test_size=0.33, random_state=42)

    random.fit(X_train,y_train)

    preds = random.predict(X_test)

    average_precision6 = average_precision_score(y_test,preds)

    roc_auc_score6 =  roc_auc_score(y_test,preds)
    accuracy_score6 = accuracy_score(y_test,preds)
    precision_score6 = precision_score(y_test,preds)
    recall_score6 = recall_score(y_test,preds)
    matthews_corrcoef6 = matthews_corrcoef(y_test,preds)

    from imblearn.under_sampling import NearMiss
    undersample = NearMiss(version=1, n_neighbors=3)
    X_near, y_near = undersample.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_near, y_near, test_size=0.33, random_state=42)

    random.fit(X_train,y_train)

    preds = random.predict(X_test)

    average_precision7 = average_precision_score(y_test,preds)

    roc_auc_score7 =  roc_auc_score(y_test,preds)
    accuracy_score7 = accuracy_score(y_test,preds)
    precision_score7 = precision_score(y_test,preds)
    recall_score7 = recall_score(y_test,preds)
    matthews_corrcoef7 = matthews_corrcoef(y_test,preds)

    from imblearn.under_sampling import TomekLinks
    undersample = TomekLinks()
    # transform the dataset
    X_TomekLinks, y_TomekLinks = undersample.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_TomekLinks, y_TomekLinks, test_size=0.33, random_state=42)

    random.fit(X_train,y_train)

    preds = random.predict(X_test)

    average_precision8 = average_precision_score(y_test,preds)

    roc_auc_score8 =  roc_auc_score(y_test,preds)
    accuracy_score8 = accuracy_score(y_test,preds)
    precision_score8 = precision_score(y_test,preds)
    recall_score8 = recall_score(y_test,preds)
    matthews_corrcoef8 = matthews_corrcoef(y_test,preds)

    from imblearn.under_sampling import OneSidedSelection
    undersample = OneSidedSelection(n_neighbors=1, n_seeds_S=200)
    X_Oss, y_Oss = undersample.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_Oss, y_Oss, test_size=0.33, random_state=42)

    random.fit(X_train,y_train)

    preds = random.predict(X_test)

    average_precision9 = average_precision_score(y_test,preds)

    roc_auc_score9 =  roc_auc_score(y_test,preds)
    accuracy_score9 = accuracy_score(y_test,preds)
    precision_score9 = precision_score(y_test,preds)
    recall_score9 = recall_score(y_test,preds)
    matthews_corrcoef9 = matthews_corrcoef(y_test,preds)

    from imblearn.under_sampling import NeighbourhoodCleaningRule
    ncr = NeighbourhoodCleaningRule()
    X_NCL, y_NCL = ncr.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_Oss, y_Oss, test_size=0.33, random_state=42)
    random.fit(X_train,y_train)

    preds = random.predict(X_test)

    average_precision0 = average_precision_score(y_test,preds)

    roc_auc_score0 =  roc_auc_score(y_test,preds)
    accuracy_score0 = accuracy_score(y_test,preds)
    precision_score0 = precision_score(y_test,preds)
    recall_score0 = recall_score(y_test,preds)
    matthews_corrcoef0 = matthews_corrcoef(y_test,preds)
    nb_report = pd.DataFrame({
        'methods':['Random Oversampling','SMOTE','SVM SMOTE','ADASYN','CNN','ENN','Near_Miss','Tome_Links','OSS','NCL'],
        
        'matthews_corrcoef': [matthews_corrcoef1,matthews_corrcoef2,matthews_corrcoef3,matthews_corrcoef4,matthews_corrcoef5,matthews_corrcoef6,matthews_corrcoef7,matthews_corrcoef8,matthews_corrcoef9,matthews_corrcoef0],
        'PR_precision':[average_precision1,average_precision2,average_precision3,average_precision4,average_precision5,average_precision6,average_precision7,average_precision8,average_precision9,average_precision0],
        'precision_score': [precision_score1,precision_score2,precision_score3,precision_score4,precision_score5,precision_score6,precision_score7,precision_score8,precision_score9,precision_score0],
        'roc_auc_score': [roc_auc_score1,roc_auc_score2,roc_auc_score3,roc_auc_score4,roc_auc_score5,roc_auc_score6,roc_auc_score7,roc_auc_score8,roc_auc_score9,roc_auc_score0]

    })
    return nb_report

def UNSW_NB15():
    import warnings
    warnings.filterwarnings('ignore')

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
  

    sns.set_context('notebook')
    sns.set_style('white')

    import dtreeviz
    from django.conf import settings
    
    training = pd.read_csv("C:/Users/AMMA/Downloads/archive (5)/UNSW_NB15_training-set.csv")
    testing = pd.read_csv("C:/Users/AMMA/Downloads/archive (5)/UNSW_NB15_testing-set.csv")
    df = pd.concat([training,testing]).drop('id',axis=1)
    df = df.reset_index(drop=True)
    for col in ['proto', 'service', 'state']:
        df[col] = df[col].astype('category').cat.codes
    
    df['attack_cat'] = df['attack_cat'].astype('category')

    from sklearn.model_selection import train_test_split

    X = df.drop(columns = ['attack_cat', 'label'])
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


    from imblearn.over_sampling import RandomOverSampler

    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X_train,y_train)



    xgb = XGBClassifier(
        n_estimators= 100,
        use_label_encoder= False,
        max_depth= 8,
        booster= 'gbtree',
        tree_method= 'hist',
        subsample= 0.5,
        colsample_bytree= 0.5,
        importance_type= 'gain',
        objective='binary:logistic',
        eval_metric='logloss',
        predictor= 'cpu_predictor',
        n_jobs= -1)

    xgb.fit(X_res,y_res)

    preds = xgb.predict(X_test)
  
    average_precision1 = average_precision_score(y_test,preds)

    roc_auc_score1 =  roc_auc_score(y_test,preds)
    accuracy_score1 = accuracy_score(y_test,preds)
    precision_score1 = precision_score(y_test,preds)
    recall_score1 = recall_score(y_test,preds)
    matthews_corrcoef1 = matthews_corrcoef(y_test,preds)

    from imblearn.over_sampling import SMOTE

    sm = SMOTE(random_state=42)
    X_smote, y_smote = sm.fit_resample(X,y)

    X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.33, random_state=42)

    xgb = XGBClassifier(
        n_estimators= 100,
        use_label_encoder= False,
        max_depth= 8,
        booster= 'gbtree',
        tree_method= 'hist',
        subsample= 0.5,
        colsample_bytree= 0.5,
        importance_type= 'gain',
        objective='binary:logistic',
        eval_metric='logloss',
        predictor= 'cpu_predictor',
        n_jobs= -1)

    xgb.fit(X_smote, y_smote)

    preds = xgb.predict(X_test)
  

    average_precision2 = average_precision_score(y_test,preds)

    roc_auc_score2 =  roc_auc_score(y_test,preds)
    accuracy_score2 = accuracy_score(y_test,preds)
    precision_score2 = precision_score(y_test,preds)
    recall_score2 = recall_score(y_test,preds)
    matthews_corrcoef2 = matthews_corrcoef(y_test,preds)


    from sklearn.ensemble import RandomForestClassifier

    random = RandomForestClassifier()

    from imblearn.over_sampling import SVMSMOTE
    sm = SVMSMOTE(random_state=42)
    X_ssmote, y_ssmote = sm.fit_resample(X,y)

    X_train, X_test, y_train, y_test = train_test_split(X_ssmote, y_ssmote, test_size=0.33, random_state=42)

    random.fit(X_train,y_train)

    preds = random.predict(X_test)

    average_precision3 = average_precision_score(y_test,preds)

    roc_auc_score3 =  roc_auc_score(y_test,preds)
    accuracy_score3 = accuracy_score(y_test,preds)
    precision_score3 = precision_score(y_test,preds)
    recall_score3 = recall_score(y_test,preds)
    matthews_corrcoef3 = matthews_corrcoef(y_test,preds)

    from imblearn.over_sampling import ADASYN

    ada = ADASYN(random_state=42)
    X_ada, y_ada = ada.fit_resample(X,y)

    X_train, X_test, y_train, y_test = train_test_split(X_ada, y_ada, test_size=0.33, random_state=42)

    random.fit(X_train,y_train)

    preds = random.predict(X_test)
 
    average_precision4 = average_precision_score(y_test,preds)

    roc_auc_score4 =  roc_auc_score(y_test,preds)
    accuracy_score4 = accuracy_score(y_test,preds)
    precision_score4 = precision_score(y_test,preds)
    recall_score4 = recall_score(y_test,preds)
    matthews_corrcoef4 = matthews_corrcoef(y_test,preds)


    from imblearn.under_sampling import CondensedNearestNeighbour

    undersample = CondensedNearestNeighbour(n_neighbors=1)
    X_cnn, y_cnn = undersample.fit_resample(X,y)

    X_train, X_test, y_train, y_test = train_test_split(X_cnn, y_cnn, test_size=0.33, random_state=42)

    random.fit(X_train,y_train)

    preds = random.predict(X_test)

    average_precision5 = average_precision_score(y_test,preds)

    roc_auc_score5 =  roc_auc_score(y_test,preds)
    accuracy_score5 = accuracy_score(y_test,preds)
    precision_score5 = precision_score(y_test,preds)
    recall_score5 = recall_score(y_test,preds)
    matthews_corrcoef5 = matthews_corrcoef(y_test,preds)

    from imblearn.pipeline import Pipeline
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import cross_validate

    from imblearn.under_sampling import EditedNearestNeighbours
    undersample = EditedNearestNeighbours(n_neighbors=3)
    X_enn, y_enn = undersample.fit_resample(X, y)


    X_train, X_test, y_train, y_test = train_test_split(X_enn, y_enn, test_size=0.33, random_state=42)

    random.fit(X_train,y_train)

    preds = random.predict(X_test)

    average_precision6 = average_precision_score(y_test,preds)

    roc_auc_score6 =  roc_auc_score(y_test,preds)
    accuracy_score6 = accuracy_score(y_test,preds)
    precision_score6 = precision_score(y_test,preds)
    recall_score6 = recall_score(y_test,preds)
    matthews_corrcoef6 = matthews_corrcoef(y_test,preds)

    from imblearn.under_sampling import NearMiss
    undersample = NearMiss(version=1, n_neighbors=3)
    X_near, y_near = undersample.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_near, y_near, test_size=0.33, random_state=42)

    random.fit(X_train,y_train)

    preds = random.predict(X_test)

    average_precision7 = average_precision_score(y_test,preds)

    roc_auc_score7 =  roc_auc_score(y_test,preds)
    accuracy_score7 = accuracy_score(y_test,preds)
    precision_score7 = precision_score(y_test,preds)
    recall_score7 = recall_score(y_test,preds)
    matthews_corrcoef7 = matthews_corrcoef(y_test,preds)

    from imblearn.under_sampling import TomekLinks
    undersample = TomekLinks()
    # transform the dataset
    X_TomekLinks, y_TomekLinks = undersample.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_TomekLinks, y_TomekLinks, test_size=0.33, random_state=42)

    random.fit(X_train,y_train)

    preds = random.predict(X_test)

    average_precision8 = average_precision_score(y_test,preds)

    roc_auc_score8 =  roc_auc_score(y_test,preds)
    accuracy_score8 = accuracy_score(y_test,preds)
    precision_score8 = precision_score(y_test,preds)
    recall_score8 = recall_score(y_test,preds)
    matthews_corrcoef8 = matthews_corrcoef(y_test,preds)

    from imblearn.under_sampling import OneSidedSelection
    undersample = OneSidedSelection(n_neighbors=1, n_seeds_S=200)
    X_Oss, y_Oss = undersample.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_Oss, y_Oss, test_size=0.33, random_state=42)

    random.fit(X_train,y_train)

    preds = random.predict(X_test)

    average_precision9 = average_precision_score(y_test,preds)

    roc_auc_score9 =  roc_auc_score(y_test,preds)
    accuracy_score9 = accuracy_score(y_test,preds)
    precision_score9 = precision_score(y_test,preds)
    recall_score9 = recall_score(y_test,preds)
    matthews_corrcoef9 = matthews_corrcoef(y_test,preds)

    from imblearn.under_sampling import NeighbourhoodCleaningRule
    ncr = NeighbourhoodCleaningRule()
    X_NCL, y_NCL = ncr.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_Oss, y_Oss, test_size=0.33, random_state=42)
    random.fit(X_train,y_train)

    preds = random.predict(X_test)

    average_precision0 = average_precision_score(y_test,preds)

    roc_auc_score0 =  roc_auc_score(y_test,preds)
    accuracy_score0 = accuracy_score(y_test,preds)
    precision_score0 = precision_score(y_test,preds)
    recall_score0 = recall_score(y_test,preds)
    matthews_corrcoef0 = matthews_corrcoef(y_test,preds)
    nb_report = pd.DataFrame({
        'methods':['Random Oversampling','SMOTE','SVM SMOTE','ADASYN','CNN','ENN','Near_Miss','Tome_Links','OSS','NCL'],
        
        'matthews_corrcoef': [matthews_corrcoef1,matthews_corrcoef2,matthews_corrcoef3,matthews_corrcoef4,matthews_corrcoef5,matthews_corrcoef6,matthews_corrcoef7,matthews_corrcoef8,matthews_corrcoef9,matthews_corrcoef0],
        'PR_precision':[average_precision1,average_precision2,average_precision3,average_precision4,average_precision5,average_precision6,average_precision7,average_precision8,average_precision9,average_precision0],
        'precision_score': [precision_score1,precision_score2,precision_score3,precision_score4,precision_score5,precision_score6,precision_score7,precision_score8,precision_score9,precision_score0],
        'roc_auc_score': [roc_auc_score1,roc_auc_score2,roc_auc_score3,roc_auc_score4,roc_auc_score5,roc_auc_score6,roc_auc_score7,roc_auc_score8,roc_auc_score9,roc_auc_score0]

    })
    return nb_report


def turnover():
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier, ExtraTreesClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.model_selection import cross_val_predict
    from sklearn.model_selection import train_test_split
    df = pd.read_csv("C:/Users/AMMA/Downloads/turnover.csv", encoding = 'ISO-8859-1')
    from sklearn.preprocessing import LabelEncoder
    print('Categorical columns: ')
    for col in df.columns:
        if df[col].dtype == 'object':
            values = df[col].value_counts()
            values = dict(values)
            
            print(str(col))
            label = LabelEncoder()
            label = label.fit(df[col])
            df[col] = label.transform(df[col].astype(str))
            
            new_values = df[col].value_counts()
            new_values = dict(new_values)
            
            value_dict = {}
            i=0
            for key in values:
                value_dict[key] = list(new_values)[i]
                i+= 1
    X = df.drop(columns=['event'])
    y = df['event']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


    from imblearn.over_sampling import RandomOverSampler

    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X_train,y_train)



    xgb = XGBClassifier(
        n_estimators= 100,
        use_label_encoder= False,
        max_depth= 8,
        booster= 'gbtree',
        tree_method= 'hist',
        subsample= 0.5,
        colsample_bytree= 0.5,
        importance_type= 'gain',
        objective='binary:logistic',
        eval_metric='logloss',
        predictor= 'cpu_predictor',
        n_jobs= -1)

    xgb.fit(X_res,y_res)

    preds = xgb.predict(X_test)
  
    average_precision1 = average_precision_score(y_test,preds)

    roc_auc_score1 =  roc_auc_score(y_test,preds)
    accuracy_score1 = accuracy_score(y_test,preds)
    precision_score1 = precision_score(y_test,preds)
    recall_score1 = recall_score(y_test,preds)
    matthews_corrcoef1 = matthews_corrcoef(y_test,preds)

    from imblearn.over_sampling import SMOTE

    sm = SMOTE(random_state=42)
    X_smote, y_smote = sm.fit_resample(X,y)

    X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.33, random_state=42)

    xgb = XGBClassifier(
        n_estimators= 100,
        use_label_encoder= False,
        max_depth= 8,
        booster= 'gbtree',
        tree_method= 'hist',
        subsample= 0.5,
        colsample_bytree= 0.5,
        importance_type= 'gain',
        objective='binary:logistic',
        eval_metric='logloss',
        predictor= 'cpu_predictor',
        n_jobs= -1)

    xgb.fit(X_smote, y_smote)

    preds = xgb.predict(X_test)
  

    average_precision2 = average_precision_score(y_test,preds)

    roc_auc_score2 =  roc_auc_score(y_test,preds)
    accuracy_score2 = accuracy_score(y_test,preds)
    precision_score2 = precision_score(y_test,preds)
    recall_score2 = recall_score(y_test,preds)
    matthews_corrcoef2 = matthews_corrcoef(y_test,preds)


    from sklearn.ensemble import RandomForestClassifier

    random = RandomForestClassifier()

    from imblearn.over_sampling import SVMSMOTE
    sm = SVMSMOTE(random_state=42)
    X_ssmote, y_ssmote = sm.fit_resample(X,y)

    X_train, X_test, y_train, y_test = train_test_split(X_ssmote, y_ssmote, test_size=0.33, random_state=42)

    random.fit(X_train,y_train)

    preds = random.predict(X_test)

    average_precision3 = average_precision_score(y_test,preds)

    roc_auc_score3 =  roc_auc_score(y_test,preds)
    accuracy_score3 = accuracy_score(y_test,preds)
    precision_score3 = precision_score(y_test,preds)
    recall_score3 = recall_score(y_test,preds)
    matthews_corrcoef3 = matthews_corrcoef(y_test,preds)

    from imblearn.over_sampling import ADASYN

    ada = ADASYN(random_state=42)
    X_ada, y_ada = ada.fit_resample(X,y)

    X_train, X_test, y_train, y_test = train_test_split(X_ada, y_ada, test_size=0.33, random_state=42)

    random.fit(X_train,y_train)

    preds = random.predict(X_test)
 
    average_precision4 = average_precision_score(y_test,preds)

    roc_auc_score4 =  roc_auc_score(y_test,preds)
    accuracy_score4 = accuracy_score(y_test,preds)
    precision_score4 = precision_score(y_test,preds)
    recall_score4 = recall_score(y_test,preds)
    matthews_corrcoef4 = matthews_corrcoef(y_test,preds)


    from imblearn.under_sampling import CondensedNearestNeighbour

    undersample = CondensedNearestNeighbour(n_neighbors=1)
    X_cnn, y_cnn = undersample.fit_resample(X,y)

    X_train, X_test, y_train, y_test = train_test_split(X_cnn, y_cnn, test_size=0.33, random_state=42)

    random.fit(X_train,y_train)

    preds = random.predict(X_test)

    average_precision5 = average_precision_score(y_test,preds)

    roc_auc_score5 =  roc_auc_score(y_test,preds)
    accuracy_score5 = accuracy_score(y_test,preds)
    precision_score5 = precision_score(y_test,preds)
    recall_score5 = recall_score(y_test,preds)
    matthews_corrcoef5 = matthews_corrcoef(y_test,preds)

    from imblearn.pipeline import Pipeline
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.model_selection import cross_validate

    from imblearn.under_sampling import EditedNearestNeighbours
    undersample = EditedNearestNeighbours(n_neighbors=3)
    X_enn, y_enn = undersample.fit_resample(X, y)


    X_train, X_test, y_train, y_test = train_test_split(X_enn, y_enn, test_size=0.33, random_state=42)

    random.fit(X_train,y_train)

    preds = random.predict(X_test)

    average_precision6 = average_precision_score(y_test,preds)

    roc_auc_score6 =  roc_auc_score(y_test,preds)
    accuracy_score6 = accuracy_score(y_test,preds)
    precision_score6 = precision_score(y_test,preds)
    recall_score6 = recall_score(y_test,preds)
    matthews_corrcoef6 = matthews_corrcoef(y_test,preds)

    from imblearn.under_sampling import NearMiss
    undersample = NearMiss(version=1, n_neighbors=3)
    X_near, y_near = undersample.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_near, y_near, test_size=0.33, random_state=42)

    random.fit(X_train,y_train)

    preds = random.predict(X_test)

    average_precision7 = average_precision_score(y_test,preds)

    roc_auc_score7 =  roc_auc_score(y_test,preds)
    accuracy_score7 = accuracy_score(y_test,preds)
    precision_score7 = precision_score(y_test,preds)
    recall_score7 = recall_score(y_test,preds)
    matthews_corrcoef7 = matthews_corrcoef(y_test,preds)

    from imblearn.under_sampling import TomekLinks
    undersample = TomekLinks()
    # transform the dataset
    X_TomekLinks, y_TomekLinks = undersample.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_TomekLinks, y_TomekLinks, test_size=0.33, random_state=42)

    random.fit(X_train,y_train)

    preds = random.predict(X_test)

    average_precision8 = average_precision_score(y_test,preds)

    roc_auc_score8 =  roc_auc_score(y_test,preds)
    accuracy_score8 = accuracy_score(y_test,preds)
    precision_score8 = precision_score(y_test,preds)
    recall_score8 = recall_score(y_test,preds)
    matthews_corrcoef8 = matthews_corrcoef(y_test,preds)

    from imblearn.under_sampling import OneSidedSelection
    undersample = OneSidedSelection(n_neighbors=1, n_seeds_S=200)
    X_Oss, y_Oss = undersample.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_Oss, y_Oss, test_size=0.33, random_state=42)

    random.fit(X_train,y_train)

    preds = random.predict(X_test)

    average_precision9 = average_precision_score(y_test,preds)

    roc_auc_score9 =  roc_auc_score(y_test,preds)
    accuracy_score9 = accuracy_score(y_test,preds)
    precision_score9 = precision_score(y_test,preds)
    recall_score9 = recall_score(y_test,preds)
    matthews_corrcoef9 = matthews_corrcoef(y_test,preds)

    from imblearn.under_sampling import NeighbourhoodCleaningRule
    ncr = NeighbourhoodCleaningRule()
    X_NCL, y_NCL = ncr.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_Oss, y_Oss, test_size=0.33, random_state=42)
    random.fit(X_train,y_train)

    preds = random.predict(X_test)

    average_precision0 = average_precision_score(y_test,preds)

    roc_auc_score0 =  roc_auc_score(y_test,preds)
    accuracy_score0 = accuracy_score(y_test,preds)
    precision_score0 = precision_score(y_test,preds)
    recall_score0 = recall_score(y_test,preds)
    matthews_corrcoef0 = matthews_corrcoef(y_test,preds)
    nb_report = pd.DataFrame({
        'methods':['Random Oversampling','SMOTE','SVM SMOTE','ADASYN','CNN','ENN','Near_Miss','Tome_Links','OSS','NCL'],
        
        'matthews_corrcoef': [matthews_corrcoef1,matthews_corrcoef2,matthews_corrcoef3,matthews_corrcoef4,matthews_corrcoef5,matthews_corrcoef6,matthews_corrcoef7,matthews_corrcoef8,matthews_corrcoef9,matthews_corrcoef0],
        'PR_precision':[average_precision1,average_precision2,average_precision3,average_precision4,average_precision5,average_precision6,average_precision7,average_precision8,average_precision9,average_precision0],
        'precision_score': [precision_score1,precision_score2,precision_score3,precision_score4,precision_score5,precision_score6,precision_score7,precision_score8,precision_score9,precision_score0],
        'roc_auc_score': [roc_auc_score1,roc_auc_score2,roc_auc_score3,roc_auc_score4,roc_auc_score5,roc_auc_score6,roc_auc_score7,roc_auc_score8,roc_auc_score9,roc_auc_score0]

    })
    return nb_report



