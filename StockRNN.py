"""
import urllib
import time
 
from urllib.request import urlopen
from bs4 import BeautifulSoup
  
stockItem = '005930'
 
url = 'http://finance.naver.com/item/sise_day.nhn?code='+ stockItem
html = urlopen(url) 
source = BeautifulSoup(html.read(), "html.parser")
 
maxPage=source.find_all("table",align="center")
mp = maxPage[0].find_all("td",class_="pgRR")
mpNum = int(mp[0].a.get('href')[-3:])
                                            
for page in range(1, mpNum+1):
  url = 'http://finance.naver.com/item/sise_day.nhn?code=' + stockItem +'&page='+ str(page)
  html = urlopen(url)
  source = BeautifulSoup(html.read(), "html.parser")
  srlists=source.find_all("tr")
  isCheckNone = None
   
  if((page % 1) == 0):
    time.sleep(1.50)
 
  for i in range(1,len(srlists)-1):
   if(srlists[i].span != isCheckNone):
     
    srlists[i].td.text
    print(srlists[i].find_all("td",align="center")[0].text, srlists[i].find_all("td",class_="num")[0].text )
   
참조: https://twpower.github.io/84-how-to-use-beautiful-soup
"""
import tensorflow as tf
import numpy as np
import pandas as pd

data = pd.read_csv("C://Users//grago//Desktop//dataset_samsung.csv")
test_data = pd.read_csv("C://Users//grago//Desktop//test.csv")
test = test_data.values[:, 0]
X_train = data.values[:, 0]
y_train = data.values[:, 1]

split_amount = 10
n_neurons = 250
n_inputs = 1
n_outputs = 1
n_steps = split_amount

X = tf.placeholder(shape = [None, n_steps, 1], dtype = tf.float32)
y = tf.placeholder(shape = [None, n_steps, 1], dtype = tf.float32)


split_size = int(len(X_train)/ split_amount)

_X = []
_y = []
X_test = []
for split in range (split_size):
     _X.append(np.array(X_train[split * split_amount : split * split_amount + split_amount]))
     _y.append(np.array(y_train[split * split_amount : split * split_amount + split_amount]))
     X_test.append(np.array(X_test[split * split_amount : split * split_amount + split_amount]))



_X = np.reshape(_X, (-1,n_steps,n_inputs))
_y = np.reshape(_y, (-1,n_steps,n_outputs))
X_test = np.reshape(X_test, (-1,n_steps,n_outputs))
X_train = _X
y_train = _y


cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicLSTMCell(num_units = n_neurons, activation = tf.nn.elu), output_size = n_outputs)
outputs, states = tf.nn.dynamic_rnn(cell, X , dtype = tf.float32)
learning_rate = 0.01
loss = loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_op = optimizer.minimize(loss)


n_iterations = 2000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for iteration in range(n_iterations):
        sess.run(training_op, feed_dict = {X: X_train, y: y_train})
        if iteration % 100 == 0:
            print (loss.eval(feed_dict = {X: X_train, y: y_train}))

    