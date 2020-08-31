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


"""
from sklearn.ensemble import voting_classifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier

ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=2),
        n_estimators= 300,
        learning_rate = 0.3,
    )
ada_clf.fit(data, target)


 ##0.82
from sklearn.metrics import accuracy_score
y_pred = ada_clf.predict(test)
print(y_pred)
#print(accuracy_score(target_test, y_pred))
"""

X = tf.placeholder(tf.float32, (None, 2))
y = tf.placeholder(tf.int32, (None))

he_init = tf.variance_scaling_initializer()

n_h1 = 150
n_h2 = 300
n_h3 = 150
n_h4 = 200
n_h5 = 100
n_output = 2
learning_rate = 0.01 
n_iteration = 4000

def max_norm_regularizer(threshhold, axes = 1, collection = "max_norm"):
    def max_norm(weights):
        clipped = tf.clip_by_norm(weights, clip_norm = threshhold, axes = axes)
        clip_weights = tf.assign(weights, clipped)
        tf.add_to_collection(collection, clip_weights)
        return None
    return max_norm
max_normed = max_norm_regularizer(threshhold= 1.0)

hidden1 = tf.layers.dense(X, n_h1, kernel_initializer= he_init, activation = tf.nn.elu, kernel_regularizer= max_normed)
h1_drop = tf.layers.dropout(hidden1, rate = 0.5)
hidden2 = tf.layers.dense(h1_drop, n_h2, kernel_initializer= he_init, activation = tf.nn.elu, kernel_regularizer= max_normed)
h2_drop = tf.layers.dropout(hidden2, rate = 0.5)
hidden3 = tf.layers.dense(h2_drop, n_h3, kernel_initializer= he_init, activation = tf.nn.elu, kernel_regularizer= max_normed)
h3_drop = tf.layers.dropout(hidden3, rate = 0.5)
logits = tf.layers.dense(h3_drop, n_output)

xentrophy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = y)
loss = tf.reduce_mean(xentrophy)
train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)



    

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for iteration in range(n_iteration):
        sess.run(train_op, feed_dict = {X:data, y: target})
        if(iteration % 100 == 0):
            print(loss.eval(feed_dict = {X: data, y : target}))
    output = sess.run(logits, feed_dict = {X:test})
    
    for i in range(len(output)):
        if(output[i][0] > output[i][1]):
            print(0)
        else:
            print(1)
