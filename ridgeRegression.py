"""
The aim is to find the coefficients that minimize the 
sum of squares error by applying a penalty to these coefficients.

- It is resistant to over-learning.
- It is biased but has low variance. (Sometimes we prefer biased models more)
- It is better than LS when there are too many parameters.
- It offers a solution to the curse of multidimensionality.
- It is effective when there is a multicollinearity problem.
- Builds a model with all variables. It does not remove irrelevant variables 
    from the model, but brings their coefficients closer to zero.
- λ plays a critical role. It allows to check the relative effects of two terms.
- λ için iyi bir değer bulunması önemlidir. Bunun için CV yöntemi kullanılır.

"""

import numpy as np
import pandas as pd
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, RidgeCV


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

## Coefficients
ridge_model = Ridge(alpha=0.1).fit(X_train, y_train) ## The alpha parameter is the lambda value in the formula.
print(ridge_model.coef_)


y_pred = ridge_model.predict(X_test)
print('Pre-CV Test RMSE:' + str(np.sqrt(mean_squared_error(y_test, y_pred))))


## Model Tuning
lambdas = 10**np.linspace(10,-2, 100)*0.5

ridge_model = Ridge()
coeffs = []

for i in lambdas:
    ridge_model.set_params(alpha = i)
    ridge_model.fit(X_train, y_train)
    coeffs.append(ridge_model.coef_)

ax = plt.gca()
ax.plot(lambdas, coeffs)
ax.set_xscale('log')

plt.xlabel('Lambda(Alpha) Values')
plt.ylabel('Coefficients/Weights')
plt.title('Ridge Coefficients as a Function of Regularization')
plt.show()


lambdas = 10**np.linspace(10,-2, 100)*0.5
ridge_cv = RidgeCV(alphas=lambdas, scoring='neg_mean_squared_error', normalize=True)

ridge_cv.fit(X_train, y_train)

print("Optimum lambda: " + str(ridge_cv.alpha_))

## Final Model
ridge_tuned = Ridge(alpha=ridge_cv.alpha_, normalize=True).fit(X_train, y_train)

print("Post CV Test RMSE: " + str(np.sqrt(mean_squared_error(y_test, ridge_tuned.predict(X_test)))))

































