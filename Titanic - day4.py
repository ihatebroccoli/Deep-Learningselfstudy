# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 13:34:49 2020

@author: grago
"""
import tensorflow as tf
import numpy as np
import pandas as pd

import os

train_path = "C://Users//grago//Desktop//t_train.csv"
titan = pd.read_csv(train_path)
titan.drop("PassengerId", axis = 1, inplace = True)
titan.drop("Pclass", axis = 1, inplace = True)
titan.drop("Name", axis = 1, inplace = True)
titan.drop("SibSp", axis = 1, inplace = True)
titan.drop("Ticket", axis = 1, inplace = True)
titan.drop("Parch", axis = 1, inplace = True)
titan.drop("Fare", axis = 1, inplace = True)
titan.drop("Embarked", axis = 1, inplace = True)
titan.drop("Cabin", axis = 1, inplace = True)
median = titan["Age"].median()
titan["Age"].fillna(median, inplace = True)

test_path = "C://Users//grago//Desktop//t_test.csv"
test = pd.read_csv(test_path)
test.drop("PassengerId", axis = 1, inplace = True)
test.drop("Pclass", axis = 1, inplace = True)
test.drop("Name", axis = 1, inplace = True)
test.drop("SibSp", axis = 1, inplace = True)
test.drop("Ticket", axis = 1, inplace = True)
test.drop("Parch", axis = 1, inplace = True)
test.drop("Fare", axis = 1, inplace = True)
test.drop("Embarked", axis = 1, inplace = True)
test.drop("Cabin", axis = 1, inplace = True)
median = test["Age"].median()
test["Age"].fillna(median, inplace = True)

for i in range(len(titan["Sex"])):
    if (titan["Sex"][i] == 'female'):
        titan["Sex"][i] = 0
    else:
        titan["Sex"][i] = 1
     
for i in range(len(test["Sex"])):
    if (test["Sex"][i] == 'female'):
        test["Sex"][i] = 0
    else:
        test["Sex"][i] = 1

target = titan["Survived"]
target = (np.array(target)).reshape(-1)
titan.drop("Survived", axis = 1, inplace = True)
data = titan


from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
"""
ada1_clf = AdaBoostClassifier(
        DecisionTreeClassifier(),
        n_estimators= 200,
        algorithm='SAMME.R',
    )
ada2_clf = AdaBoostClassifier(
        RandomForestClassifier(),
        n_estimators= 200,
        algorithm='SAMME.R',
    )
ada3_clf = AdaBoostClassifier(
        SVC(),
        n_estimators= 200,
        algorithm='SAMME.R',
    )
ada4_clf = AdaBoostClassifier(
        SGDClassifier(),
        n_estimators= 200,
        algorithm='SAMME.R',
    )
"""

n_clfs = 1501

ada5_clf = []
prediction = np.array(np.zeros(418), dtype= 'int32')
print(prediction)

for i in range(n_clfs):
    ada5_clf.append(AdaBoostClassifier(
            DecisionTreeClassifier(),
            n_estimators= 600,
            algorithm='SAMME.R',
            learning_rate = 1
            
            ))

show = 1
print(len(ada5_clf))
for ada5 in ada5_clf: 
    print(show)
    show += 1
    ada5.fit(data, target)
    result = ada5.predict(test)
    prediction = prediction + np.array(result)
    


print(prediction)
#print(accuracy_score(target_test, y_pred))

prediction = 

for i in range(len(prediction)):
    if(prediction[i] >= n_clfs * 2 / 3 + 1):
        print(i+892,',',1)
    else:
        print(i+892,',',0)
