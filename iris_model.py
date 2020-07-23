from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle

iris = datasets.load_iris()
print(iris)

X = iris.data
Y = iris.target

X_train,X_test,Y_train,Y_test = train_test_split(X,Y)

KNN = KNeighborsClassifier()
KNN = KNN.fit(X_train,Y_train)

SVC = SVC()
SVC = SVC.fit(X_train,Y_train)

rfc = RandomForestClassifier()
rfc = rfc.fit(X_train,Y_train)

pickle.dump(KNN,open('KNN.pkl','wb'))
pickle.dump(SVC,open('SVC.pkl','wb'))
pickle.dump(rfc,open('rfc.pkl','wb'))