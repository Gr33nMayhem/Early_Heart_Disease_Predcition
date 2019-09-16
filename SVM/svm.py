import pandas as pd

cols_to_use = ['TenYearCHD']
Y = pd.read_csv('framingham.csv', usecols= cols_to_use)

data = pd.read_csv('framingham.csv')
data = data.fillna(0)
X = data.drop(labels='TenYearCHD', axis=1)

X=X.as_matrix()
Y=Y.as_matrix()

Y=Y.flatten()


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2)

print(X_train.shape)
print(X_test.shape)


from sklearn import svm

clf = svm.SVC(kernel='linear') # Linear Kernel
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))