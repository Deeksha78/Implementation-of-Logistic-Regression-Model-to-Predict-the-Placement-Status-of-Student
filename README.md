# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Deeksha P
RegisterNumber: 212222040031 
*/
```
```
import pandas as pd
data=pd.read_csv('/Placement_Data(1).csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test,y_pred)#Accuracy Score = (TP+TN)/(TP+FN+TN+FP)
#accuracy_score(y_true,y_pred,normalize=False)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion #11+24=35 -correct predictions,5+3=8 incorrect predictions

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
Placement data

![image](https://github.com/Deeksha78/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128116204/1740e3fd-ea17-4112-b8ec-669ccf92682c)

Salary data

![image](https://github.com/Deeksha78/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128116204/a16610f1-7f4d-4a54-926b-79349553f4de)

Checking the null() function

![image](https://github.com/Deeksha78/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128116204/9710a79d-a20d-4af0-babd-afe0c7123ecc)

Data Duplicate

![image](https://github.com/Deeksha78/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128116204/75d820b0-22a4-4951-8cc9-9d7d713abafd)

Print data

![image](https://github.com/Deeksha78/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128116204/81cbf560-6477-4c04-bfbd-98a76c24cf05)

Data-Status

![image](https://github.com/Deeksha78/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128116204/85d9e019-dac3-4d85-b967-c75ddf74a438)

y_prediction array

![image](https://github.com/Deeksha78/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128116204/30067106-b374-4b5a-86eb-1dd3d04ac88d)

Accuracy value

![image](https://github.com/Deeksha78/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128116204/e7ae42e0-a7bd-4d6e-bda4-635f5dd7948c)

Confusion array

![image](https://github.com/Deeksha78/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128116204/71e6d7f2-cd0b-47e9-87da-31cf53f38649)

Classification report

![image](https://github.com/Deeksha78/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128116204/18e222de-5075-49d0-a9b8-14799c5eb2ee)

Prediction of LR

![image](https://github.com/Deeksha78/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128116204/076d0b5c-8d5f-4c2b-bdb5-034539d9cf3f)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
