import pandas as pd
import numpy as np

ad = pd.read_csv("Advertising.csv", usecols=[1,2,3,4]) 
df = ad.copy()


## First Five Rows of Dataset
print("First Five Rows of Dataset: \n" + str(df.head(5)))

## general information about the dataset
print("general information about the dataset: \n" + str(df.info()))

## Statistical descriptions of the Data Set
print("Statistical descriptions of the Dataset: \n" + str(df.describe().T))

## Is there any null value in the dataset?
print("Is there any null value in the dataset? \n" + str(df.isnull().values.any()))

## correlation values between variables
print("correlation values between variables: \n" + str(df.corr()))

## Simple Linear Regression with statsmodels framework
## First way to construct simple linear regression
import statsmodels.api as sm

## Dependent and independent variables are selected for the model
X = df[["TV"]]
X = sm.add_constant(X) ## adding constant to dataframe for constant coefficient in function
y = df["sales"] 

linear_model_sm = sm.OLS(y,X)
model_sm = linear_model_sm.fit()
print(model_sm.summary())

print("Coefficients: \n" + str(model_sm.params))


## Second way to construct simple linear regression
import statsmodels.formula.api as smf
linear_model_smf = smf.ols("sales ~ TV", df)
model_smf = linear_model_smf.fit()
print(model_smf.summary())

print("Confidence internal: \n" + str(model_smf.conf_int()))

print("f_pvalue: %.5f" % model_smf.f_pvalue)
print("f_value: %.2f" % model_smf.fvalue)
print("t_value: %.3f" % model_smf.tvalues[0:1])
print("MSE: %.3f" % model_smf.mse_model)
print("r_squared: %.3f" % model_smf.rsquared)
print("adj r_squared: %.3f" % model_smf.rsquared_adj)


print("adj r_squared: %.3f" % model_smf.rsquared_adj)

## Obtaining the equation of the model
print('sales = ' + str('%.2f' % model_smf.params[0]) + ' + TV * ' + str('%.2f' % model_smf.params[1]))



## Simple Linear Regression with Scikit-Learn
from sklearn.linear_model import LinearRegression
df = ad.copy()
## The linear regression model in sklearn accepts data in matrix format.
## For this reason, the data was reshaped.
X = df.TV.values.reshape(-1,1)
y = df.sales.values.reshape(-1,1)

reg = LinearRegression()
model_sk = reg.fit(X=X, y=y)

print("Model Constant: " + str(model_sk.intercept_[0]))
print("TV Coef: " + str(model_sk.coef_[0][0]))

## r2 score
print("r2_score: " + str(model_sk.score(X,y)))


from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y, model_smf.fittedvalues)
print("MSE: " + str(mse))
print("RMSE: " + str(np.sqrt(mse)))

print("Remains of the model:")
print(model_smf.resid)







