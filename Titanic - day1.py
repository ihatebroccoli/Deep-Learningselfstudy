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



for i in range(len(titan["Sex"])):
    if (titan["Sex"][i] == 'female'):
        titan["Sex"][i] = 0
    else:
        titan["Sex"][i] = 1
     
from sklearn.model_selection import train_test_split
train, test = train_test_split(titan, test_size = 0.2)


target = titan["Survived"]
target = (np.array(target)).reshape(-1)
titan.drop("Survived", axis = 1, inplace = True)
data = titan

target_test = test["Survived"]
test.drop("Survived", axis = 1, inplace = True)
data_test = test


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
y_pred = ada_clf.predict(data_test)
print(accuracy_score(target_test, y_pred))
"""

X = tf.placeholder(tf.float32, (None, 2))
y = tf.placeholder(tf.int32, (None))

he_init = tf.variance_scaling_initializer()

n_h1 = 100
n_h2 = 300
n_h3 = 100
n_h4 = 200
n_h5 = 100
n_output = 2
learning_rate = 0.01 
n_iteration = 20000

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
logits = tf.layers.dense(hidden3, n_output)

xentrophy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = y)
loss = tf.reduce_mean(xentrophy)
train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for iteration in range(n_iteration):
        sess.run(train_op, feed_dict = {X:data, y: target})
        if(iteration % 100 == 0):
            print(loss.eval(feed_dict = {X: data, y : target}))
    output = sess.run(accuracy, feed_dict = {X:data_test, y: target_test})
    print(output)