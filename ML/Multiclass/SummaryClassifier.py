# importing necessary libraries 
from sklearn import datasets 
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.model_selection import train_test_split 

import pandas as pd

# valid value: ['iris', 'letters']
config = dict()
config['dataset'] = 'letters'
max_features = None

def select_dataset():
    if config['dataset'] == 'iris':
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        max_features = 3
    elif config['dataset'] == 'letters':
        df = pd.read_csv('./datasets/letter-recognition.data')
        X = df.values[:,1:].astype("float64")
        y = df.values[:,0]
        max_features = 26
    else:
        pass

    return X, y

X, y = select_dataset()

# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0) 

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# training a DescisionTreeClassifier 
from sklearn.tree import DecisionTreeClassifier 
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train) 
dtree_predictions = dtree_model.predict(X_test) 

# model accuracy for X_test
accuracy = dtree_model.score(X_test, y_test) 

print(f"Decision Tree -> accuracy: {accuracy}")
print("Classification Report")
print(classification_report(y_test, dtree_predictions))

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# training a linear SVM classifier 
from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 

# model accuracy for X_test 
accuracy = svm_model_linear.score(X_test, y_test) 

print(f"SVM -> accuracy: {accuracy}")
print("Classification Report")
print(classification_report(y_test, svm_predictions))

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# training a KNN classifier 
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train) 

# accuracy on X_test 
accuracy = knn.score(X_test, y_test) 

# creating a confusion matrix 
knn_predictions = knn.predict(X_test) 

print(f"KNN -> accuracy: {accuracy} ")
print("Classification Report")
print(classification_report(y_test, knn_predictions))

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)

max_lr, score = None, 0
lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

gb_construct = lambda lr: GradientBoostingClassifier(n_estimators=50, learning_rate=lr, max_features=2, max_depth=8, random_state=0)

for lr in lr_list:
    gb_clf = gb_construct(lr)
    gb_clf.fit(X_train, y_train)
    ensemble_predictions = gb_clf.predict(X_test)

    score_train = gb_clf.score(X_train, y_train)
    score_test = gb_clf.score(X_test, y_test)

    print("Learning rate: ", lr)
    print("gbdt -> Accuracy score (training): {0:.3f}".format(score_train))
    print("gbdt -> Accuracy score (validation): {0:.3f}".format(score_test))

    if score_test > score: max_lr = lr; score = score_test;

optimum_gb_clf = gb_construct(max_lr)
optimum_gb_clf.fit(X_train, y_train)
gb_predictions = optimum_gb_clf.predict(X_test)
print("Classification Report")
print(classification_report(y_test, gb_predictions))

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
from xgboost import XGBClassifier
xgb_clf = XGBClassifier()
xgb_clf.fit(X_train, y_train)

# model accuracy for X_test 
accuracy = xgb_clf.score(X_test, y_test) 

xgb_predictions = xgb_clf.predict(X_test)
print(f"xgb -> accuracy: {accuracy}")
print("Classification Report")
print(classification_report(y_test, xgb_predictions))
