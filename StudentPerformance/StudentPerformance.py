import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

df = pd.read_csv('StudentPerformanceFactors.csv')

num_col = df.select_dtypes(include=["int64","float64"]).columns.dropna()
cat_col = df.select_dtypes(include=["object"]).columns.dropna()

# ----------
# correlation
# ----------

print(df.describe())
print(df.info())
print((df.isnull().sum() / len(df)) * 100 )

df.dropna(subset=['Teacher_Quality','Parental_Education_Level','Distance_from_Home'], inplace=True)

corr = df[num_col].corr()
target_corr = corr['Exam_Score'].sort_values(ascending=False) 

plt.figure(figsize=(10,6))
plt.title('Correlation of Numerical Features with Exam Score', fontsize=16)

for col in num_col:
    sns.scatterplot(x=df[col], y=df['Exam_Score'])
    print(plt.show())

# ----------
# Anova test
# ----------

for col in cat_col:
    groups = [df[df[col] == cat]['Exam_Score'] for cat in df[col].unique()]   
    f, p = f_oneway(*groups)
    # print(col , "p->value:",p)
    if p < 0.05:
        print(col, "keep feature")
    else:
        print(col, "remove feature")
    
# ------------------
# Mutial information
# ------------------

from sklearn.feature_selection import mutual_info_regression

df_encode = df.copy()

for col in df_encode.select_dtypes(include=["object"]).columns:
    df_encode[col] = df_encode[col].astype('category').cat.codes

df_encode = df_encode.dropna()

x = df_encode.drop(columns=['Exam_Score']).dropna()
y = df_encode['Exam_Score'].dropna()

mi = mutual_info_regression(x,y)
mi_score = pd.Series(mi, index=x.columns).sort_values(ascending=False)
print(mi_score)

