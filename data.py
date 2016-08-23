import pandas as pd
import numpy as np
import yaml
import sklearn
import sklearn.cluster

def fetch():
    df = pd.read_excel('DNS 2015_16 (PUBLIC) Report.xlsx', 'results')
    return df

def preprocess(df):
    question_groups = {
        'unemployed':
        {
            'indicator': (df.iloc[:, 11] == 'Unemployed'),
            'questions':
                df.columns[12:14]
        },
        'student_or_learning_a_profession':
        {
            'indicator': (df.iloc[:, 11] == 'Student / Apprentice') ,
            'questions': df.columns[15:18]
        },
        'working':
            {
            'indicator': (df.iloc[:, 11].isin(['Freelance', 'Employee', 'Entrepreneur',
           'Company Owner']) | df.iloc[:, 15] == 1),
            'questions': df.columns[18:44]
        },
        'location_independent':
        {
            'indicator': df.iloc[:, 43] == 'Yes',
            'questions': df.columns[44:70]
        },
        'would_like':
        {
            'indicator': (~df.iloc[:,71].isnull()),
            'questions': df.columns[70:81]
        },
        'would_not_like':
        {
            'indicator': (~df.iloc[:,82].isnull()),
            'questions': df.columns[82:85]
        },
        'all':
        {
            'indicator': pd.Series(True, index=df.index),
            'questions':
                df.columns[1:12] | df.columns[85:93]
        }
    }
    df_types = pd.Series('mc', index=df.columns)
    for g_name, g_prop  in question_groups.items():
        for col in g_prop['questions']:
            try:
                _ = df.loc[~df[col].isnull(),col].astype(int)
                if df.loc[~df[col].isnull(),col].unique().shape[0] == 2:
                    df[col] = df[col].astype(bool)
                    df_types[col] = 'bool'
                else:
                    df_types[col] = 'star'
            except:
                if df.loc[~df[col].isnull(),col].unique().shape[0] == 1:
                    df[col] = (df[col] == df.loc[~df[col].isnull(),col].unique()[0])
                    df_types[col] = 'bool'
                elif df.loc[~df[col].isnull(),col].unique().shape[0] > 10:
                    df_types[col] = 'free'
            df.loc[~g_prop['indicator'], col] = np.nan
    df_types['#'] = 'index'
    return df, df_types, question_groups

def get_data():
    df = fetch()
    return preprocess(df)

def get_subsets(df, df_types):
    all_col = df.columns[1:].tolist()
    bool_col = df_types[df_types == 'bool'].index.tolist()
    free_col = df_types[df_types == 'free'].index.tolist()
    mc_col = df_types[df_types == 'mc'].index.tolist()
    star_col = df_types[df_types == 'star'].index.tolist()
    num = df.apply((lambda sr: sr.nunique()), axis=0)
    num_3 = num[num <= 3].index.tolist()
    return all_col, bool_col, free_col, mc_col, star_col, num_3

def get_interesting(df):
    iq1 = \
        df.columns[[3, 5, 9, 11, 30, 31, 43, 52, 61, 87, 88, 89, 90, 91]]
    iq2 = \
        df.columns[[3, 5, 9, 11, 30, 31, 43, 52, 61, 87, 88, 89, 90, 91]]
    return iq1, iq2


def foo(sr, pos=-1):
    sr.index = np.arange(sr.shape[0])
    return sr[sr == 1].index.tolist()[-1]


def cluster_pre(df, liqmc, sliq):

    locin = ['less than 1 week', '1-2 weeks', '2-4 weeks', '1-3 month', '3-6 month',
       '6-12 month', 'more than 12 months']

    dfp = df.copy()
    unordered = {col: df.loc[:, col].unique().tolist() for col in liqmc}
    with open('resources/unordered.yml', 'w') as f:
        yaml.dump(unordered, f, default_flow_style=False)
    with open('resources/ordered.yml', 'r') as f:
        ordered = yaml.load(f)
    str_to_num = {coln: {ans: i for i, ans in enumerate(ass)}
                  for coln, ass in ordered.items()}
    num_to_str = {coln: {i: ans for i, ans in enumerate(ass)}
                  for coln, ass in ordered.items()}
    for col in liqmc:
        dfp.loc[:,col] = dfp.loc[:,col] \
                            .apply(lambda x: str_to_num[col][x]) \
                            .astype(float)
    dfps = dfp.loc[:, sliq]
    # try:
    #     dfps['min_stay'] = dfps[locin].apply(foo, axis=1, pos=0)
    #     dfps['max_stay'] = dfps[locin].apply(foo, axis=1, pos=-1)
    #     dfps = dfps.drop(locin,  axis=1)
    # except:
    #     pass
    dfps = sklearn.preprocessing.Imputer(strategy='most_frequent').fit_transform(dfps)
    dfps = sklearn.preprocessing.Normalizer().fit_transform(dfps)
    # dfps = sklearn.preprocessing.RobustScaler().fit_transform(dfps)
    # dfps[:,1] = dfps[:,1] * 2
    return dfps
