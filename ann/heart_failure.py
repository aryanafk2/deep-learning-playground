import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

import sklearn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.metrics import r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

#import and read the dataset
data = pd.read_csv("datasets/heart_failure_clinical_records_dataset.csv")
df = data.copy()

print(df.head(10))
print("\nShape:", df.shape)
df.info()

inp_data = df.drop("DEATH_EVENT", axis=1)
out_data = df["DEATH_EVENT"]

inp_data_scaled = MinMaxScaler().fit_transform(inp_data)

X_train, X_test, y_train, y_test = train_test_split(
    inp_data_scaled, 
    out_data, 
    test_size=0.2, 
    random_state=42
    )

print("X_train Shape : ", X_train.shape)
print("X_test Shape  : ", X_test.shape)
print("y_train Shape : ", y_train.shape)
print("y_test Shape  : ", y_test.shape)

from sklearn.neural_network import MLPClassifier

mlpc_model = MLPClassifier(random_state=42)
mlpc_model.fit(X_train, y_train)
y_pred = mlpc_model.predict(X_test)

print('Accuracy Score: {:.4f}'.format(accuracy_score(y_test, y_pred)))
print('SVC f1-score  : {:.4f}'.format(f1_score(y_pred, y_test)))
print('SVC precision : {:.4f}'.format(precision_score(y_pred, y_test)))
print('SVC recall    : {:.4f}'.format(recall_score(y_pred, y_test)))
print("\n",classification_report(y_pred, y_test))

activation = list(['identity', 'logistic', 'tanh', 'relu'])
solver = list(['lbfgs', 'sgd', 'adam'])
alpha = list([0.0001, 0.05])
learning_rate = list(['constant','adaptive'])
hidden_layer_sizes = list([(50,50,50), (50,100,50), (100,)])

param_grid = dict(
    activation = activation,
    solver = solver,
    alpha = alpha,
    learning_rate = learning_rate,
    hidden_layer_sizes = hidden_layer_sizes
)

mlp = MLPClassifier(max_iter=100)
clf = GridSearchCV(mlp, param_grid, cv=10, n_jobs=-1, verbose=2)
clf.fit(X_train, y_train)
clf.best_params_

best_params = clf.best_params_

mlpc_model = MLPClassifier(
    activation=best_params['activation'],
    alpha=best_params['alpha'],
    hidden_layer_sizes=best_params['hidden_layer_sizes'],
    learning_rate=best_params['learning_rate'],
    solver=best_params['solver'],
    random_state=42
)

mlpc_model.fit(X_train, y_train)

y_pred = mlpc_model.predict(X_test)

print("Accuracy Score: {:.4f}".format(
    accuracy_score(y_test, y_pred)
))

print("F1 Score: {:.4f}".format(
    f1_score(y_test, y_pred)
))

print("Precision: {:.4f}".format(
    precision_score(y_test, y_pred)
))

print("Recall: {:.4f}".format(
    recall_score(y_test, y_pred)
))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cf_matrix,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

scores = []

for i in range(500):

    X_train, X_test, y_train, y_test = train_test_split(
        inp_data_scaled,
        out_data,
        test_size=0.2
    )

    model = MLPClassifier(
        activation=best_params['activation'],
        alpha=best_params['alpha'],
        hidden_layer_sizes=best_params['hidden_layer_sizes'],
        learning_rate=best_params['learning_rate'],
        solver=best_params['solver'],
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    scores.append(
        accuracy_score(y_test, y_pred)
    )


plt.figure(figsize=(8,5))

plt.hist(scores, bins=20)

plt.title("Accuracy Distribution")
plt.xlabel("Accuracy")
plt.ylabel("Frequency")

plt.show()