import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


data=pd.read_csv('test.csv')
x=data[['x']]
y=data['y']
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=7)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
plt.scatter(X_test, y_test, label="Actual Data", color="red")
plt.plot(X_test, y_pred, label="Regression Line", color="yellow", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()