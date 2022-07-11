import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('sampleDatas/positions.csv')
level = data['Level'].values.reshape(-1,1)
salary = data['Salary'].values

regression = RandomForestRegressor(n_estimators=10, random_state=0) ## The number of trees in the forest. default value is 100
regression.fit(level, salary)

print(regression.predict([[8.3]]))