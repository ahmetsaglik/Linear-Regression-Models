"""
The aim is to find the coefficients that minimize the sum of squares error by applying a 
    penalty to these coefficients.

- It is proposed to overcome the disadvantage of Ridge regression to leave all relevant and irrelevant variables in the model.
- In Lasso, the coefficients are approached to zero.
- But the L1 norm makes some coefficients zero when λ is large enough. Thus, the variable selection is made.
- It is very important to choose λ correctly, CV is used here too.
- The Ridge and Lasso methods are not superior to each other.

"""
from warnings import filterwarnings
filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, LassoCV


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
lasso_model = Lasso(alpha=0.1).fit(X_train, y_train)
print(lasso_model.coef_)


## Trying diffrent lambda values
lambdas = 10**np.linspace(10,-2, 100)*0.5

lasso = Lasso()
coeffs = []

for i in lambdas:
    lasso.set_params(alpha = i)
    lasso.fit(X_train, y_train)
    coeffs.append(lasso.coef_)

ax = plt.gca()
ax.plot(lambdas*2, coeffs)
ax.set_xscale('log')

plt.xlabel('Lambda(Alpha) Values')
plt.ylabel('Coefficients/Weights')
plt.title('Lasso Coefficients as a Function of Regularization');
plt.show()

## Prediction and Test RMSE
y_pred = lasso.predict(X_test)
print("Pre-CV Test RMSE: " + str(np.sqrt(mean_squared_error(y_test, y_pred))))


## Lasso Model Tuning

from sklearn.linear_model import LassoCV
lasso_cv_model = LassoCV(alphas=None, 
                         cv=10,
                         normalize=True, 
                         max_iter=10000).fit(X_train, y_train)

## Tuning Result
print("Optimum Lambda Value for Lasso: " + str(lasso_cv_model.alpha_))


## Lasso Regression Final Model and Test RMSE
lasso_tuned = Lasso(alpha=lasso_cv_model.alpha_).fit(X_train, y_train)

y_pred = lasso.predict(X_test)
print("Post CV Test RMSE: " + str(np.sqrt(mean_squared_error(y_test, y_pred))))





