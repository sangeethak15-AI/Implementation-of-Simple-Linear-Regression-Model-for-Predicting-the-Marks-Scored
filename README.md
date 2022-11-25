# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.
```
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sangeetha.K
RegisterNumber: 212221230085 
*/
import pandas as pd
import numpy as np
df=pd.read_csv('student_scores.csv')
print(df)

X=df.iloc[:,:-1].values
Y=df.iloc[:,1].values
print(X,Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error

plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title(' Training set (Hours Vs Scores)')
plt.xlabel('Hours')
plt.ylabel('Scores')

plt.scatter(X_test,Y_test,color='green')
plt.plot(X_test,reg.predict(X_test),color='purple')
plt.title(' Training set (Hours Vs Scores)')
plt.xlabel('Hours')
plt.ylabel('Scores')

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)

```

## Output:
![mx1](https://user-images.githubusercontent.com/93992063/204019608-654d26f1-c81a-44df-b42d-817318f73ef9.png)

![30a](https://user-images.githubusercontent.com/93992063/204019633-5fc697ab-05fb-46a9-bfee-644afde0e55e.png)

![30b](https://user-images.githubusercontent.com/93992063/204019673-cbeed266-bf04-4b96-8ae0-8d5d3cdbcbb1.png)


![6](https://user-images.githubusercontent.com/93992063/204019987-20bca8cc-48a2-42ca-b7fa-60383a383946.png)

![mx2](https://user-images.githubusercontent.com/93992063/204019692-0c85bcb7-9046-482e-8673-8e2d88410ac2.png)

![mx3](https://user-images.githubusercontent.com/93992063/204019706-70ebcf35-6912-4b76-84de-d1ce33f7905a.png)

![mx4](https://user-images.githubusercontent.com/93992063/204019738-14cef619-c430-488c-be15-092304272713.png)

![mx5](https://user-images.githubusercontent.com/93992063/204019760-1fa0b638-70b0-4a1b-a686-2f34e9716185.png)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
