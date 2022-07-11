## Multiple Linear Regression
## It is used if the number of data affecting the result is more than one.

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

data = pd.read_csv("datas/insurance.csv")

print(data.columns)

## Spending estimate by age and number of children
expenses = data['expenses'].values.reshape(-1,1)  ## y axis
ageChildrens = data.iloc[:,[0,3]].values

## create a regression object
regression = LinearRegression()

## fitting the data into the model
regression.fit(ageChildrens, expenses)


## some sample predictions
print(regression.predict(np.array([[25,0]])))
print(regression.predict(np.array([[25,2]])))
print(regression.predict(np.array([[30,0]])))
print(regression.predict(np.array([[30,1]])))
print(regression.predict(np.array([[30,4]])))
print(regression.predict(np.array([[20,2]])))
print(regression.predict(np.array([[25,2]])))
print(regression.predict(np.array([[30,2]])))
print(regression.predict(np.array([[35,2]])))






