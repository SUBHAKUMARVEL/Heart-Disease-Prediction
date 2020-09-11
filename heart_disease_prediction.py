# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 12:13:01 2019

@author: This pc
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 09:30:18 2019

@author: This pc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Other libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import accuracy_score
# Machine Learning
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

ch = pd.read_csv(r'C:\Users\This pc\Desktop\Mini project\heart.csv')
print(ch.head())

heat_map = sns.heatmap(ch, vmin=0, vmax=1, center=10)
plt.show()
plt.figure(figsize=(10, 5))
sns.heatmap(ch.corr(), annot=True)
sns.heatmap(ch.corr())
plt.show()

ch = pd.read_csv(r'heart.csv')
print(ch.head())
grp = sns.regplot(x='target', y='cp', data=ch, color='yellow')
plt.title("Heart disease")
plt.show()

target = ch[['target']]
#print(target)
cp = ch[['cp']]
#print(cp)
reg = linear_model.LinearRegression()
reg.fit(target, cp)
disease = pd.read_csv(r'C:\Users\This pc\Desktop\Mini project\heart1.csv')
print(disease.head(5))
p = reg.predict(disease)
print("predicted values")
print(p)
print('Variance score: {}'.format(reg.score(target, cp)))

print("Thalach-slope")
ch = pd.read_csv(r'C:\Users\This pc\Desktop\Mini project\heart.csv')
print(ch.head())
grp = sns.regplot(x='target', y='thalach', data=ch, color='yellow')
plt.title("Heart disease")
plt.show()

thalach= ch[['thalach']]
#print(thalach)
target= ch[['target']]
#print(slope)
reg = linear_model.LinearRegression()
reg.fit(target,thalach)
disease1 = pd.read_csv(r'C:\Users\This pc\Desktop\Mini project\thalach.csv')
print(disease1.head(5))
p1 = reg.predict(disease1)
print("predicted values")
print(p1)
print('Variance score: {}'.format(reg.score(target,thalach)))

'''k neighbors classifier'''
dataset = pd.read_csv(r'C:\Users\This pc\Desktop\Mini project\heart.csv')
y = dataset['target']
X = dataset.drop(['target'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

print("KNN")
knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(X_train, y_train)
    knn_scores.append(knn_classifier.score(X_test, y_test))
plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')
for i in range(1,21):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1, 21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')

knn =KNeighborsClassifier()
params = {'n_neighbors':[i for i in range(1,33,2)]}
model = GridSearchCV(knn,params,cv=10)
model.fit(X_train,y_train)
model.best_params_ 
predict = model.predict(X_test)

print('Accuracy Score: ',accuracy_score(y_test,predict))
print('Using k-NN we get an accuracy score of: ',
      round(accuracy_score(y_test,predict),5)*100,'%')

print(classification_report(y_test,predict))
#Get predicted probabilites from the model
y_probabilities = model.predict_proba(X_test)[:,1]
#Create true and false positive rates
false_positive_rate_knn,true_positive_rate_knn,threshold_knn = roc_curve(y_test,y_probabilities)
#Plot ROC Curve
plt.figure(figsize=(10,6))
plt.title('Revceiver Operating Characterstic')
plt.plot(false_positive_rate_knn,true_positive_rate_knn)
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
print("KNN")
print(roc_auc_score(y_test,y_probabilities))

'''Decisison tree'''
dataset = pd.read_csv(r'C:\Users\This pc\Desktop\Mini project\heart.csv')
y = dataset['target']
X = dataset.drop(['target'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

print("Decision tree")
dt_scores = []
for i in range(1, len(X.columns) + 1):
    dt_classifier = DecisionTreeClassifier(max_features = i, random_state = 0)
    dt_classifier.fit(X_train, y_train)
    dt_scores.append(dt_classifier.score(X_test, y_test))
plt.plot([i for i in range(1, len(X .columns) + 1)], dt_scores, color = 'green')
for i in range(1, len(X.columns) + 1):
    plt.text(i, dt_scores[i-1], (i, dt_scores[i-1]))
plt.xticks([i for i in range(1, len(X.columns) + 1)])
plt.xlabel('Max features')
plt.ylabel('Scores')
plt.title('Decision Tree Classifier scores for different number of maximum features')
dtree= DecisionTreeClassifier(random_state=7)
#Setting parameters for GridSearchCV
params = {'max_features': ['auto', 'sqrt', 'log2'],
          'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15], 
          'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11]}
tree_model = GridSearchCV(dtree, param_grid=params, n_jobs=-1)
tree_model.fit(X_train,y_train)
#Printing best parameters selected through GridSearchCV
tree_model.best_params_
predict = tree_model.predict(X_test)
from sklearn.metrics import accuracy_score
print('Accuracy Score : ',accuracy_score(y_test,predict))
print('Using Decision Tree we get an accuracy score of: ',
      round(accuracy_score(y_test,predict),5)*100,'%')
print(classification_report(y_test,predict))

target_probailities_tree = tree_model.predict_proba(X_test)[:,1]
#Create true and false positive rates
tree_false_positive_rate,tree_true_positive_rate,tree_threshold = roc_curve(y_test,
                                                             target_probailities_tree)
#Plot ROC Curve
sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
plt.title('Reciver Operating Characterstic Curve')
plt.plot(tree_false_positive_rate,tree_true_positive_rate)
plt.plot([0,1],ls='--')
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.show()