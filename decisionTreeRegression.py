import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor


data = pd.read_csv("sampleDatas/positions.csv")

level = data['Level'].values.reshape(-1,1)
salary = data.iloc[:, 2].values.reshape(-1,1)

regression = DecisionTreeRegressor()
regression.fit(level, salary)


print(regression.predict([[8.3]]))
print(regression.predict([[8.5]]))
print(regression.predict([[8.9]]))


plt.scatter(level, salary)
x = np.arange(min(level), max(level), 0.01).reshape(-1,1)
plt.plot(x, regression.predict(x), color='red', label='Decision Tree Regression')
plt.xlabel("Level")
plt.ylabel("Salary")
plt.title("Decision Tree Regression Model")
plt.show()




























