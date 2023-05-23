import numpy as np
import pandas as pd
#Load and Explore the Dataset
ds=pd.read_csv(r"C:\Users\HP\Desktop\INTERNSHIP 5 MARCH\ml project\sale prediction ml project 07\Advertising Budget and Sales.csv")
print(ds)

print(ds.head(5))
print(ds.tail(5))
print(ds.shape)
### Get summary of statistics of the data
print(ds.describe)
print(ds.info())
print(ds.isnull().sum())


## Separate the predictor and the target variable
x = ds.iloc[:,:-1]
y = ds.iloc[:,-1]

print(y)
### Visualize Correlation
import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(ds, height=2.5)
plt.subplots(figsize=(11, 9))
plt.scatter(ds["TV Ad Budget ($)"], ds["Sales ($)"])
plt.xlabel("TV")
plt.ylabel("Sales")
plt.title("Sales vs TV")

plt.subplots(figsize=(11, 9))
plt.scatter(ds["Radio Ad Budget ($)"], ds["Sales ($)"])
plt.xlabel("Radio")
plt.ylabel("Sales ($) ")
plt.title("Sales vs Radio")

plt.subplots(figsize=(11, 9))
plt.scatter(ds["Newspaper Ad Budget ($)"], ds["Sales ($)"])
plt.xlabel("Newspaper")
plt.ylabel("Sales ($) ")
plt.title("Sales vs Radio")
plt.show()
### Train-Test Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=11)
print("x_train size:", x_train.shape)
print("y_train size:", y_train.shape)
print("\nx_test size:", x_test.shape)
print("y_test size:", y_test.shape)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(x_train,y_train)


pre=regressor.predict(x_test)


from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.metrics import r2_score, mean_absolute_error
print("R2:", r2_score(pre, y_test)*100)
print("MAE:", mean_absolute_error(pre, y_test))
"""print("confusion_matrix is :{0}%".format(confusion_matrix(y_test,pre)))
print(accuracy_score(pre,y_test)*100)
print(classification_report(pre,y_test))"""