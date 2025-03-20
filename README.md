# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import necessary libraries (e.g., pandas, numpy,matplotlib).
2.Load the dataset and then split the dataset into training and testing sets using sklearn library.
3.Create a Linear Regression model and train the model using the training data (study hours as input, marks scored as output).
4.Use the trained model to predict marks based on study hours in the test dataset.
5.Plot the regression line on a scatter plot to visualize the relationship between study hours and marks scored.
```

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())

x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)


```


## Output:
![image](https://github.com/user-attachments/assets/f120d70e-da68-4c98-8c77-daf77230e191)
![image](https://github.com/user-attachments/assets/3fcb5ac5-1cf0-4484-8197-af57e2d32c10)
![image](https://github.com/user-attachments/assets/97537201-b440-4549-8d9b-586bfbb19956)
![image](https://github.com/user-attachments/assets/8e53a2fd-095a-49ce-ae29-4a0d24e4e759)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
