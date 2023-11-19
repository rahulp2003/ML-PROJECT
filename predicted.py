import csv,os,re,sys,codecs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib,  statistics
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from collections import Counter
from sklearn.preprocessing import MinMaxScaler


reader=pd.read_csv("marketing_trn_data.csv") # my data is stored in a variable called reader

for i in range(len(reader)):
    if reader.iloc[i,2]=='YOLO' or reader.iloc[i,2]=='Absurd':
        print(i)
print(reader['Marital_Status'].value_counts())

data = reader.drop(['Dt_Customer','Year_Birth'],axis=1)
one_hot_encoded1= pd.get_dummies(data['Marital_Status'], prefix='Marital_Status')
one_hot_encoded2 = pd.get_dummies(data['Education'], prefix='Education')
encoded12 = pd.concat([one_hot_encoded2, one_hot_encoded1], axis=1)
df_encoded = pd.concat([data, encoded12], axis=1)
data = df_encoded.drop(['Marital_Status','Education'],axis=1)
data[' Income '] = data[' Income '].str.replace(r'\D', '', regex=True)
data[' Income '] = data[' Income '].astype(float)
data[" Income "] = data[" Income "].fillna(data[" Income "].mean())

target=pd.read_csv("marketing_trn_class_labels.csv",names = ["X","Y"])
target=target["Y"]

test=pd.read_csv("marketing_tst_data.csv")
print(test['Marital_Status'].value_counts())
test = test.drop(['Dt_Customer','Year_Birth'],axis=1)
one_hot_encoded1= pd.get_dummies(test['Marital_Status'], prefix='Marital_Status')
one_hot_encoded2 = pd.get_dummies(test['Education'], prefix='Education')
encoded12 = pd.concat([one_hot_encoded2, one_hot_encoded1], axis=1)
df_encoded = pd.concat([test, encoded12], axis=1)
test = df_encoded.drop(['Marital_Status','Education'],axis=1)
test[' Income '] = test[' Income '].str.replace(r'\D', '', regex=True)
test[' Income '] = test[' Income '].astype(float)
test[" Income "] = test[" Income "].fillna(test[" Income "].mean())


# print(test.isnull().sum())
# print(len(data))
# print(len(target))


clf=RandomForestClassifier()

pipeline = Pipeline([('scaler', MinMaxScaler()),
                    ('feature_selection', SelectKBest(chi2, k=18)),
                    ('clf',RandomForestClassifier(class_weight='balanced',criterion='entropy', n_estimators=30, max_depth=20)),
                ])


pipeline.fit(data,target) 

X_test = test
predictions = pipeline.predict(X_test)
predictions=pd.DataFrame({"Class":predictions})
predictions.to_csv('Predictions.csv',index=False)