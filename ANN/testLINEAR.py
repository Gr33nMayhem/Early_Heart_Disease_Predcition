import pandas as pd
import metrics
from sklearn.metrics import classification_report, confusion_matrix

cols_to_use = ['TenYearCHD']
Y = pd.read_csv('framingham.csv', usecols= cols_to_use)

data = pd.read_csv('framingham.csv')
data = data.fillna(0)
X = data.drop(labels='TenYearCHD', axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

import keras
from keras import backend as k
from  keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

model=Sequential([
	Dense(1024,input_shape=(15,),activation='linear'),
	Dense(512,activation='linear'),
	Dense(124,activation='linear'),
	Dense(64,activation='linear'),
	Dense(32,activation='linear'),
	Dense(16,activation='linear'),
	Dense(2,activation='softmax')
])

model.summary()

model.compile(Adam(lr=.000001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=10,validation_split=0.1,epochs=15,shuffle=True,verbose=2)

score=model.evaluate(X_test,Y_test,verbose=1)

print(score)
