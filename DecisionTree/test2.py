from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cols_to_use = ['TenYearCHD']
Y = pd.read_csv('framingham.csv', usecols= cols_to_use)

data = pd.read_csv('framingham.csv')
data = data.fillna(0)
X = data.drop(labels='TenYearCHD', axis=1)

X = data.drop(labels='prevalentHyp', axis=1)

X['age'] = ((X['age']-X['age'].min())/(X['age'].max()-X['age'].min()))

X['cigsPerDay'] = ((X['cigsPerDay']-X['cigsPerDay'].min())/(X['cigsPerDay'].max()-X['cigsPerDay'].min()))

X['totChol'] = ((X['totChol']-X['totChol'].min())/(X['totChol'].max()-X['totChol'].min()))

X['sysBP'] = ((X['sysBP']-X['sysBP'].min())/(X['sysBP'].max()-X['sysBP'].min()))

X['diaBP'] = ((X['diaBP']-X['diaBP'].min())/(X['diaBP'].max()-X['diaBP'].min()))

X['BMI'] = ((X['BMI']-X['BMI'].min())/(X['BMI'].max()-X['BMI'].min()))

X['heartRate'] = ((X['heartRate']-X['heartRate'].min())/(X['heartRate'].max()-X['heartRate'].min()))

X['glucose'] = ((X['glucose']-X['glucose'].min())/(X['glucose'].max()-X['glucose'].min()))


#Split the data into training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.5,random_state=0)

#Ready the Machine learning algorithm and set hyperparameters
tree = DecisionTreeClassifier(criterion='entropy')#,max_depth=1,random_state=0)

#Now we train the Model
tree.fit(X_train,Y_train)

#Make the prediction using the trained model
Y_predict = tree.predict(X_test)

#printing the results
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(Y_test,Y_predict))
print(classification_report(Y_test,Y_predict))