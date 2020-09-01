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
#titan.drop("Pclass", axis = 1, inplace = True)
titan.drop("Name", axis = 1, inplace = True)
#titan.drop("SibSp", axis = 1, inplace = True)
titan.drop("Ticket", axis = 1, inplace = True)
#titan.drop("Parch", axis = 1, inplace = True)
titan.drop("Fare", axis = 1, inplace = True)
titan.drop("Embarked", axis = 1, inplace = True)
titan.drop("Cabin", axis = 1, inplace = True)
median = titan["Age"].median()
titan["Age"].fillna(median, inplace = True)

test_path = "C://Users//grago//Desktop//t_test.csv"
test = pd.read_csv(test_path)
test.drop("PassengerId", axis = 1, inplace = True)
#test.drop("Pclass", axis = 1, inplace = True)
test.drop("Name", axis = 1, inplace = True)
#test.drop("SibSp", axis = 1, inplace = True)
test.drop("Ticket", axis = 1, inplace = True)
#test.drop("Parch", axis = 1, inplace = True)
test.drop("Fare", axis = 1, inplace = True)
test.drop("Embarked", axis = 1, inplace = True)
test.drop("Cabin", axis = 1, inplace = True)
median = test["Age"].median()
test["Age"].fillna(median, inplace = True)

divider = titan["Age"].max()
titan["Age"] = titan["Age"] / 100
test["Age"] = test["Age"] / 100

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

from sklearn.model_selection import train_test_split
data, valid_X,target, valid_y = train_test_split(data,target, test_size = 0.1, shuffle = True)


he_init = tf.variance_scaling_initializer()



def max_norm_regularizer(threshhold, axes = 1, collection = "max_norm"):
    def max_norm(weights):
        clipped = tf.clip_by_norm(weights, clip_norm = threshhold, axes = axes)
        clip_weights = tf.assign(weights, clipped)
        tf.add_to_collection(collection, clip_weights)
        return None
    return max_norm
max_normed = max_norm_regularizer(threshhold= 1.0)


model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1000, input_shape = (5,)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(500, activation = 'elu'),
        #tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(250, activation = 'elu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(100, activation = 'elu'),
        tf.keras.layers.Dense(2, activation = 'softmax'),
        
    ])
model.compile(optimizer = 'Adam', loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
ckpt_path = "best_one.ckpt"
ckpt = tf.keras.callbacks.ModelCheckpoint(filepath = ckpt_path,
                          verbose = 1,
                          save_weights_only = True,
                          save_best_only = True,
                          monitor = 'val_loss'
                          )


model.fit(data, target, epochs = 1500, verbose = 1, 
          validation_data=(valid_X, valid_y), 
          callbacks=[ckpt],
          batch_size = 100
          )

model.load_weights(ckpt_path)

output = model.predict(test)
for i in range(len(output)):
    if(output[i][0] > output[i][1]):
        print(0)
    else:
        print(1)
