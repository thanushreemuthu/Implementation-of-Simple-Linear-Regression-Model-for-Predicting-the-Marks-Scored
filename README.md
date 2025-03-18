# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and read the dataframe.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Thanushree M
RegisterNumber:  212224240169

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('/content/student_scores.csv')
print(dataset.head())
print(dataset.tail())

dataset.info()

#assigning hours to X & scores to Y
X = dataset.iloc[:,:-1].values
print(X)
Y = dataset.iloc[:,-1].values
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_pred = reg.predict(X_test)
print(Y_pred)
print(Y_test)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

plt.scatter(X_train,Y_train,color='Purple')
plt.plot(X_train,reg.predict(X_train),color='Red')
plt.title ("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test ,Y_test,color='Red')
plt.plot(X_test,reg.predict(X_test),color='Purple')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:

## Head Values and Tail values
![image](https://github.com/user-attachments/assets/666e8e73-7f1c-43c9-9777-09a3e874f601)
## Dataset info
![image](https://github.com/user-attachments/assets/a0156edd-3343-4931-bc24-99a3fb1663b8)
## X and Y values
![image](https://github.com/user-attachments/assets/d941f888-5a22-4249-abd5-9e37f8727d0f)

![image](https://github.com/user-attachments/assets/b1ef6ff2-bb8c-4b38-8530-a0ca6f6580c9)
## Graph plot for training data
![image](https://github.com/user-attachments/assets/3dc684ee-1239-4ad1-8e86-50bf1d971c95)
## Graph plot for test data
![image](https://github.com/user-attachments/assets/5b1fbcef-675d-4586-b371-9dea839bd21b)
## MSE, MAE and RMSE
![image](https://github.com/user-attachments/assets/37d6dcab-5a58-4f61-8a46-c49a4aa501ba)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
