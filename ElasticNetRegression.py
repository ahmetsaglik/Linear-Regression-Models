"""
The aim is to find the coefficients that minimize the sum of squares error 
    by applying a penalty to these coefficients. ElasticNet combines the 
    L1 and L2 approaches.
"""

from warnings import filterwarnings
filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet, ElasticNetCV


hit = pd.read_csv("datas/Hitters.csv")
df = hit.copy()
df = df.dropna()

## first five rows in data
print(df.head())

## informations of data
print(df.info())

## Statistical descriptions of the data
print(df.describe().T)

## dependent variable selected
y = df['Salary']

## Arguments selected
## One Hot Encoding
dummies = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
print(dummies)
## Creating arguments by adding categorical variables made in One Hot Encoding
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)


## The data was split into two for testing and training.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


## Model Building and Coefficients
enet_model = ElasticNet().fit(X_train, y_train)
print("ElasticNet Coefficients: " + str(enet_model.coef_))
print("ElasticNet Constant: " + str(enet_model.intercept_))


## Prediction and Test RMSE
y_pred = enet_model.predict(X_test)
print("Pre-CV Test RMSE: " + str(np.sqrt(mean_squared_error(y_test, y_pred))))


## Model Tuning 
enet_cv_model = ElasticNetCV(cv=10, random_state=0).fit(X_train, y_train)
print("Optimum Alpha Value: " + str(enet_cv_model.alpha_))


## Final Model and RMSE
enet_tuned = ElasticNet(alpha=enet_cv_model.alpha_).fit(X_train, y_train)

y_pred = enet_model.predict(X_test)
print("Post CV Test RMSE: " + str(np.sqrt(mean_squared_error(y_test, y_pred))))








