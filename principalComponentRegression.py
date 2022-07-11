"""
It is based on the idea of establishing a regression model for the 
resulting components after dimension reduction is applied to the variables. 
Variables resulting from reduction are independent variables with each other. 
In this way, the multicollinearity problem is eliminated.

PCA (Principal Component Analysis): It is the idea of expressing the information 
contained in a certain number of variables with a smaller number of variables. 
As an example, data with 100 variables can be expressed with fewer variables 
with the least loss of information.

"""

### PCR Model Building

import pandas as pd

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
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


## When creating the PCA object, the number of reduced components can be entered as a parameter.
## If it is not entered as a parameter, it creates as many components as there are variables in the data set.
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

pca = PCA()

X_reduced_train = pca.fit_transform(scale(X_train))
print(X_reduced_train[1,:])
"""
Examination of the cumulative sums of the variance values of the components

Here is the result:
    
     We see how much the created components can explain the variance of the data set. 
     In other words, if we use only the first component, we can explain 38.18% of the variance 
     of the data set, 59.88% if we use the first two components, and 84.18% if we use the first 
     five components.

"""
import numpy as np

print(np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100))

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
pcr_model = lm.fit(X_reduced_train, y_train)

## Calculated constant
print(pcr_model.intercept_)

## Calculated coefficients
print(pcr_model.coef_)


## Prediction
y_pred = pcr_model.predict(X_reduced_train)
## Training Error Values
from sklearn.metrics import mean_squared_error, r2_score
## RMSE
print(np.sqrt(mean_squared_error(y_train, y_pred)))
## r2 score
print(r2_score(y_train, y_pred))


## Test Error Values
pca2 = PCA()
X_reduced_test = pca2.fit_transform(scale(X_test))
y_pred = pcr_model.predict(X_reduced_test)
## RMSE
print(np.sqrt(mean_squared_error(y_test, y_pred)))
## r2 score
print(r2_score(y_test, y_pred))


## PCR Model Tuning

## 10 katlı cross validation yapılandırması
from sklearn import model_selection
cv_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)

lm = LinearRegression()
RMSE = []

for i in np.arange(1, X_reduced_train.shape[1]+1):
    score = np.sqrt(-model_selection.cross_val_score(lm, 
                                                     X_reduced_train[:,:i], 
                                                     y_train.ravel(), 
                                                     cv=cv_10, 
                                                     scoring= "neg_mean_squared_error").mean())
    RMSE.append(score)

import matplotlib.pyplot as plt

plt.plot(RMSE, '-v')
plt.xlabel('Number of Components')
plt.ylabel('RMSE')
plt.title('PCR Model Tuning')
plt.show()











