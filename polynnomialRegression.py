from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = pd.read_csv("sampleDatas/positions.csv")

x = data['Level'].values.reshape(-1,1)
y = data['Salary'].values.reshape(-1,1)

polyFeatures = PolynomialFeatures(degree=4)
levelPoly = polyFeatures.fit_transform(x)

regression = LinearRegression()
regression2 = LinearRegression()
regression.fit(levelPoly, y)
regression2.fit(x, y)

print(regression.predict(polyFeatures.fit_transform([[8.3]])))

plt.scatter(x,y)
plt.plot(x, regression2.predict(x), color='green', label='Linear Regression')
plt.plot(x, regression.predict(levelPoly), color='red', label='Polynomial Regression')
plt.legend(loc='best')
plt.show()



















