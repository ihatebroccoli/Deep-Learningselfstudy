import tensorflow as tf
import numpy as np
from PIL import Image
import os
_trainlist = []
trainlist = []
labels = []
testlist = []

x_data, y_data = [],[]

count = 0
path = 'C:/Users/grago/recogface'
feelings = os.listdir(path)
for i in range(len(feelings)):
    _trainlist.append(os.listdir(path + '/' + feelings[i]))
    count = i
    for j in range(len(os.listdir(path + '/' + feelings[i]))):
        labels.append(count)

for i in range(6):
     for j in range(len(_trainlist[i])):
         img = Image.open(path + '/' + feelings[i] + '/' + _trainlist[i][j]).convert('L')
         img = img.resize((100,100))
         arr = np.array(img)
         x_data.append(arr)
         
         
datacount = len(labels)
x_data = np.array(x_data, dtype = 'float32')
y_data = np.array(labels, dtype = 'int32')
y_data = y_data.reshape(len(y_data), 1)

from keras.utils import to_categorical
y_data = to_categorical(y_data)
x_data = x_data.reshape((-1,100,100,1))
x_data /= 255

from sklearn.model_selection import train_test_split
x_train, x_further, y_train, y_further = train_test_split(x_data, y_data, test_size = 0.1)
x_validate, x_test, y_validate, y_test = train_test_split(x_further, y_further, test_size = 0.5)


from keras import layers
from keras import models

model = models.Sequential()

model.add(layers.Conv2D(32,(5,5), strides = (2,2), activation = 'relu', input_shape = (100,100,1)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(2,2),activation = 'relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(2,2),activation = 'relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation = 'relu'))
model.add(layers.Dense(6,activation = 'softmax'))

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 70, batch_size = 100, verbose = 1, validation_data = (x_validate, y_validate))
[loss, acc] = model.evaluate(x_test, y_test, verbose = 1)
print("ACC: " + str(acc))
