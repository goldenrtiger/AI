# importing necessary libraries 
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 

import pandas as pd

# valid value: ['iris', 'letters']
config = dict()
config['dataset'] = 'letters'

def select_dataset():
    if config['dataset'] == 'iris':
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
    elif config['dataset'] == 'letters':
        df = pd.read_csv('./github/AI/ML/Multiclass/datasets/letter-recognition.data')
        X = df.values[:,1:].astype("float64")
        y = df.values[:,0]
    else:
        pass

    return X, y

X, y = select_dataset()

# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0) 

# training a DescisionTreeClassifier 
from sklearn.tree import DecisionTreeClassifier 
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train) 
dtree_predictions = dtree_model.predict(X_test) 

# creating a confusion matrix 
cm = confusion_matrix(y_test, dtree_predictions) 

print(f"Decision Tree -> confusion_matrix: \n {cm}")

# training a linear SVM classifier 
from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 

# model accuracy for X_test 
accuracy = svm_model_linear.score(X_test, y_test) 

# creating a confusion matrix 
cm = confusion_matrix(y_test, svm_predictions) 

print(f"SVM -> accuracy:{accuracy}, \n confusion_matrix:\n {cm}")

# training a KNN classifier 
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train) 

# accuracy on X_test 
accuracy = knn.score(X_test, y_test) 

# creating a confusion matrix 
knn_predictions = knn.predict(X_test) 
cm = confusion_matrix(y_test, knn_predictions) 

print(f"KNN -> accuracy:{accuracy}, \n confusion_matrix: \n {cm}")
