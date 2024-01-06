# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 12:26:33 2024

@author: hp
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten
import numpy as np
import matplotlib.pyplot as plt
import random

X_train=np.loadtxt("D:\dataset\input.csv",delimiter=',')
X_test=np.loadtxt("D:\dataset\input_test.csv",delimiter=',')
Y_train=np.loadtxt("D:\dataset\labels.csv",delimiter=',')
Y_test=np.loadtxt("D:\dataset\labels_test.csv",delimiter=',')


X_train=X_train.reshape(len(X_train),100,100,3)
X_test=X_test.reshape(len(X_test),100,100,3)
Y_train=Y_train.reshape(len(Y_train),1)
Y_test=Y_test.reshape(len(Y_test),1)

X_train=X_train/255.0
X_test=X_test/255.0
idx=random.randint(0,len(X_train))
plt.imshow(X_train[idx,:])

model=Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(100,100,3)),
    MaxPooling2D((2,2)),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64,activation='relu'),
    Dense(1,activation='sigmoid')
    ])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train,Y_train,epochs=5,batch_size=64)

model.evaluate(X_test,Y_test)

idx1=random.randint(0,len(X_train))
plt.imshow(X_train[idx1,:])
plt.show
y_pred=model.predict(X_test[idx1,:].reshape(1,100,100,3))
y_pred=y_pred>=0.5
if(y_pred):print("Dog")
else: print("Cat")
plt.show