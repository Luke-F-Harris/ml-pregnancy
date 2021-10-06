# imports 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection
import sklearn.linear_model


# 1000 births from North Carolina
# Features = father age, mother age, maturation of mother, weeks pregnant, 
#            visits, marital status, weight gained, low birth weight, gender,
#            habit, white mom

# I am going to inspect the link between weeks pregnant, and child's weight (pounds)
df = pd.read_csv('nc.csv')


# CSV contains null values
print("\nNumber of NaN: ", sum(np.isnan(df.weeks)))
df = df[~np.isnan(df.weeks)]
df.info()

# Shows the jointplot correlations
#sns.jointplot(data=df, x="weeks", y="weight")
#plt.show()

# Training the model - simple linear regression
regr = sklearn.linear_model.LinearRegression() # instantiate the model

# set values
x = df.weeks.values
X_fit = np.c_[x, x**2, x**3]
y_fit = df.weight.values

# fit the model, print the coefficents 
regr.fit(X_fit,y_fit)
regr.coef_


plt.scatter(x,y_fit)
plt.xlabel("Weeks")
plt.ylabel("Weight")
plt.title("Weeks vs Weight born in Pregnancy")

# x values to predict with
pred_weeks = np.linspace(x.min(), x.max(), 30)
X_pred = np.c_[pred_weeks, pred_weeks**2, pred_weeks**3]

# y values to predict
pred_weight = regr.predict(X_pred)
plt.plot(pred_weeks, pred_weight, color="red")
plt.show()


