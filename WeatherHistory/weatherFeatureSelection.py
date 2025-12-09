import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import f_oneway, chi2_contingency

df = pd.read_csv('weatherHistory.csv')

# print(df.info())
# print(df.describe())
# print(df.shape)
# print(df.isnull().sum())

# df_encode = df.copy()

# for col in df_encode.select_dtypes(include=["object"]).columns:
#     df_encode[col] = df_encode[col].astype('category').cat.codes

# df_encode = df_encode.dropna()

# x = df_encode.drop(columns=['Summary'])
# y = df_encode['Summary']

# mi = mutual_info_classif(x,y, discrete_features='auto', random_state=0)
# mi_score = pd.Series(mi , index = x.columns).sort_values(ascending=False)

# print("Mutual Information Scores:", mi_score)

num_col = df.select_dtypes(include=["int64", "float64"]).columns
cat_col = df.select_dtypes(include=["object"]).columns.drop('Summary')

print("ANOVA TEST RESULTS:")

for col in num_col:
    groups = [df[df['Summary'] == cat][col].dropna() for cat in df['Summary'].dropna().unique()]
    f , p = f_oneway(*groups)
    if p < 0.05:
        print(col, "-> keep feature")
    else:
        print(col,"-> remove feature")

print("CHI-SQUARE TEST RESULTS:")

for col in cat_col:
    table = pd.crosstab(df[col], df['Summary'])
    chi2, p , dof , expected = chi2_contingency(table)
    if p < 0.05:
        print(col ,"-> keep feature")
    else:
        print(col, "-> remove feature")

