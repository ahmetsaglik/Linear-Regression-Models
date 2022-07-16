"""
It is based on the idea that a regression model is established 
by reducing the variables to a smaller number of components that do not 
have multicollinearity problems between them.

- curse of multidimensionality p>n
- Multiple linear connection problem
- Like PCR, PLS finds linear combinations of independent variables. 
    These linear combinations are called components or latent.
- PLS is a special case of NIPALS, it iteratively tries to find the latent 
    relationship between the dependent variable and highly correlated variables.
- In PCR, linear combinations, ie components, are created in a way that maximally 
    sums up the variability in the independent variable space.
- This situation causes the lack of ability to explain the dependent variable.
- In PLS, on the other hand, the components are formed in a way that sums up the 
    covariance with the dependent variable as much as possible.
- If variables are not to be discarded and clarification is sought: PLS
- PLS can be viewed as a supervised size reduction procedure, PCR as an unattended size reduction procedure.
- There is a tuning parameter in both methods, which is the number of components.
- CV (Cross Validation) method is used to determine the optimum number of components.

"""

import numpy as np
import pandas as pd
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score


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


## Calculated coefficients
pls_model = PLSRegression().fit(X_train, y_train)
print("coefs: " + str(pls_model.coef_))



y_pred = pls_model.predict(X_train)
print('Pre-CV Training RMSE: ' + str(np.sqrt(mean_squared_error(y_train, y_pred))))
print('Pre-CV Training r2_score: ' + str(r2_score(y_train, y_pred)))

y_pred = pls_model.predict(X_test)
print('Pre-CV Test RMSE: ' + str(np.sqrt(mean_squared_error(y_test, y_pred))))
print('Pre-CV Test r2_score: ' + str(r2_score(y_test, y_pred)))

## PLS Model Tuning



cv_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)
RMSE = []

for i in np.arange(1, X_train.shape[1] + 1):
    pls = PLSRegression(n_components=i)
    score = np.sqrt(-model_selection.cross_val_score(pls, X_train, y_train, cv=cv_10, scoring="neg_mean_squared_error").mean())
    RMSE.append(score)

plt.plot(np.arange(1, X_train.shape[1]+1), np.array(RMSE), '-v', c='r')
plt.xlabel('Number of Components')
plt.ylabel('RMSE')
plt.title('PLS Model Tuning for Salary Estimation Model')
plt.show()


## Final Model
pls_final = PLSRegression(n_components=3).fit(X_train, y_train)
y_pred = pls_model.predict(X_train)
print('Post CV Training RMSE: ' + str(np.sqrt(mean_squared_error(y_train, y_pred))))
print('Post CV Training r2_score: ' + str(r2_score(y_train, y_pred)))
y_pred = pls_model.predict(X_test)
print('Post CV Test RMSE: ' + str(np.sqrt(mean_squared_error(y_test, y_pred))))
print('Post CV Test r2_score: ' + str(r2_score(y_test, y_pred)))






















