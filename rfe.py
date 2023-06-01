# -*- coding: utf-8 -*-
"""
Created on Thu May 25 11:58:37 2023

@author: yildi
"""


import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score,f1_score,precision_score, r2_score
from sklearn import preprocessing
from sklearn.feature_selection import RFE, RFECV

data=pd.read_excel("student_por.xlsx")
# data.shape 
# print(data.isnull().any())
# print(data.isnull().sum())

d = {'yes':1,'no':0}
data['schoolsup']=data['schoolsup'].map(d)
data['famsup']=data['famsup'].map(d)
data['paid']=data['paid'].map(d)
data['activities']=data['activities'].map(d)
data['nursery']=data['nursery'].map(d)
data['higher']=data['higher'].map(d)
data['internet']=data['internet'].map(d)
data['romantic']=data['romantic'].map(d)


d={'F':1,'M':0}
data['sex']=data['sex'].map(d)

d={'teacher':0,'health':1,'services':2,'at_home':3,'other':4}
data['Mjob']=data['Mjob'].map(d)
data['Fjob']=data['Fjob'].map(d)

d={'home':0,'reputation':1,'course':2,'other':3}
data['reason']=data['reason'].map(d)

d={'mother':0,'father':1,'other':2}
data['guardian']=data['guardian'].map(d)

d={'A':0,'T':1}
data['Pstatus']=data['Pstatus'].map(d)

d={'R':0,'U':1}
data['address']=data['address'].map(d)

d={'LE3':0,'GT3':1}
data['famsize']=data['famsize'].map(d)

d={'GP':0,'MS':1}
data['school']=data['school'].map(d)

data['absences']=preprocessing.minmax_scale(data['absences'])

d={0:0,
   1:1,2:1,3:1,4:1,
   5:2,6:2,7:2,8:2,9:2,
   10:3,11:3,12:3,13:3,14:3,
   15:4,16:4,17:4,18:4,19:4,20:4}
data['G3']=data['G3'].map(d)
# ceviri=data.transpose()
# describe=data.describe()
# info=ceviri.head(33)
# print(ceviri.head(33))
# print(data.transpose().describe())

# print(data.dtypes)

# plt.figure(figsize=(18,16))
# sns.heatmap(data.corr(),annot=True)
# plt.show()
# x=data.drop(['age','schoolsup','famsup','paid','activities','nursery','internet','romantic','sex','Mjob','Fjob',
#  'reason','guardian','traveltime','absences','famrel','freetime','goout','Dalc','Walc','health','studytime','address','Medu','Fedu','Pstatus','famsize','G3'],axis=1)


# y=data['G3']
x=data.drop('G3',axis=1)

y=data['G3']

############           DESTEK VEKTÖR MAKİNELERİ          ################
from sklearn.svm import SVC
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)

# model = SVC()

# # Hiperparametre arama alanını tanımla
# param_grid = {
#     'C': [0.1, 1, 10],
#     'kernel': ['linear', 'rbf', 'sigmoid'],
#     'gamma': ['scale', 'auto']
# }

# # GridSearchCV'yi tanımla
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)

# # GridSearchCV'yi eğit
# grid_search.fit(X_train, y_train)

# # En iyi parametreleri ve skoru al
# best_params = grid_search.best_params_
# best_score = grid_search.best_score_

# print("En iyi parametreler:", best_params)
# print("En iyi skor:", best_score)

destek = SVC(kernel = "linear")
destek=RFECV(estimator=destek, step=1)
destek.fit(X_train,y_train)
y_pred = destek.predict(X_test)

selected_features = destek.support_
feature_names = [name for name, selected in zip(x.columns, selected_features) if selected]
print(feature_names)
print(destek.ranking_)

print("Başarı sonucu Destek Vektör:")
print('doğruluk:',(accuracy_score(y_test, y_pred))*100)
print('kesinlik:',(precision_score(y_test, y_pred, average='weighted'))*100)
print('recall:',(recall_score(y_test, y_pred, average='weighted'))*100)
print('f1:',(f1_score(y_test, y_pred, average='weighted'))*100)
print("****************************")

###########           EN YAKIN KOMŞU ALGORİTMASI        ###############
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Başarı sonucu K en yakın komşu:")
print('doğruluk:',(accuracy_score(y_test, y_pred))*100)
print('kesinlik:',(precision_score(y_test, y_pred, average='weighted'))*100)
print('recall:',(recall_score(y_test, y_pred, average='weighted'))*100)
print('f1:',(f1_score(y_test, y_pred, average='weighted'))*100)
print("****************************")

############            KARAR AĞACI ALGORİTMASI         #############
from sklearn.tree import DecisionTreeClassifier
decision = DecisionTreeClassifier(max_depth=10)
decision = RFE(estimator=decision, n_features_to_select=5, step=1)
decision.fit(X_train, y_train)
y_pred=decision.predict(X_test)
selected_features = decision.support_
feature_names = [name for name, selected in zip(x.columns, selected_features) if selected]
print(feature_names)

print("Başarı sonucu Karar Ağacı:")
print('doğruluk:',(accuracy_score(y_test, y_pred))*100)
print('kesinlik:',(precision_score(y_test, y_pred, average='weighted'))*100)
print('recall:',(recall_score(y_test, y_pred, average='weighted'))*100)
print('f1:',(f1_score(y_test, y_pred, average='weighted'))*100)
print("****************************")

#############         NAİVE BAYES ALGORİTMASI       ###############
from sklearn.naive_bayes import GaussianNB
naive = GaussianNB()
naive.fit(X_train, y_train)
y_pred=naive.predict(X_test)

print("Başarı sonucu Naive Bayes:")
print('doğruluk:',(accuracy_score(y_test, y_pred))*100)
print('kesinlik:',(precision_score(y_test, y_pred, average='weighted'))*100)
print('recall:',(recall_score(y_test, y_pred, average='weighted'))*100)
print('f1:',(f1_score(y_test, y_pred, average='weighted'))*100)
print("****************************")


############            RASTGELE ORMAN ALGORİTMASI           ############
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100, max_depth=15)
forest=RFE(estimator=forest, n_features_to_select=10,step=1)
forest.fit(X_train,y_train)
y_pred=forest.predict(X_test)

selected_features = forest.support_
feature_names = [name for name, selected in zip(x.columns, selected_features) if selected]
print(feature_names) 

print("Başarı sonucu Rastgele Orman:")
print('doğruluk:',(accuracy_score(y_test, y_pred))*100)
print('kesinlik:',(precision_score(y_test, y_pred, average='weighted'))*100)
print('recall:',(recall_score(y_test, y_pred, average='weighted'))*100)
print('f1:',(f1_score(y_test, y_pred, average='weighted'))*100)
print("****************************")

###############           XGBOOST ALGORİTMASI          #############
from xgboost import XGBClassifier
xgboost = XGBClassifier()
xgboost=RFE(estimator=xgboost,n_features_to_select=10,step=1)
xgboost.fit(X_train,y_train)
y_pred = destek.predict(X_test)

selected_features = xgboost.support_
feature_names = [name for name, selected in zip(x.columns, selected_features) if selected]
print(feature_names)

xgboost.fit(X_train, y_train)
y_pred=xgboost.predict(X_test)
print("Başarı sonucu XGBoost:")
print('doğruluk:',(accuracy_score(y_test, y_pred))*100)
print('kesinlik:',(precision_score(y_test, y_pred, average='weighted'))*100)
print('recall:',(recall_score(y_test, y_pred, average='weighted'))*100)
print('f1:',(f1_score(y_test, y_pred, average='weighted'))*100)
print("****************************")

###############            ADABOOST ALGORİTMASI           #############
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10))
ada=RFE(estimator=ada,n_features_to_select=2, step=1)
ada.fit(X_train, y_train)
y_pred=ada.predict(X_test)

selected_features = ada.support_
feature_names = [name for name, selected in zip(x.columns, selected_features) if selected]
print(feature_names)

print("Başarı sonucu AdaBoost:")
print('doğruluk:',(accuracy_score(y_test, y_pred))*100)
print('kesinlik:',(precision_score(y_test, y_pred, average='weighted'))*100)
print('recall:',(recall_score(y_test, y_pred, average='weighted'))*100)
print('f1:',(f1_score(y_test, y_pred, average='micro'))*100)
print("****************************")

#############              CATBOOST ALGORİTMASI         ##############
from catboost import CatBoostClassifier
cat=CatBoostClassifier(iterations=10)
cat=RFE(estimator=cat,n_features_to_select=2, step=1)
cat.fit(X_train, y_train)
y_pred=cat.predict(X_test)

selected_features = cat.support_
feature_names = [name for name, selected in zip(x.columns, selected_features) if selected]
print(feature_names)
print("Başarı sonucu CatBoost:")
print('doğruluk:',(accuracy_score(y_test, y_pred))*100)
print('kesinlik:',(precision_score(y_test, y_pred, average='weighted'))*100)
print('recall:',(recall_score(y_test, y_pred, average='weighted'))*100)
print('f1:',(f1_score(y_test, y_pred, average='weighted'))*100)


