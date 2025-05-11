# Titanic-survival-analysis
Titanic Survival Analysis project, where we explore real passenger data from the infamous Titanic voyage to understand what factors influenced survival. Using data science techniques, we analyze variables like age, gender, passenger class, and fare to uncover trends and build predictive models.

Importing the Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

Data Collection & processing
# load the data from csv file to Pandas DataFrame
titanic_data = pd.read_csv('/content/tested.csv')
# printing the first 5 rows of the dataframe
titanic_data.head()
# number of rows and columns
titanic_data.shape
# getting some informations about the data
titanic_data.info()
# check the number of missing values in each column
titanic_data.isnull().sum()

Handling the missing values
# drop the "Cabin" column from the dataframe
titanic_data = titanic_data.drop(columns='Cabin', axis=1)
# replacing the missing values in "Age" column with mean value
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)
#replacing the missing values in "Fare" column with mean value
titanic_data['Fare'].fillna(titanic_data['Fare'].mean(), inplace=True)

# check the number of missing values in each column
titanic_data.isnull().sum()

Data Analysis
# getting some statistical measures about the data
titanic_data.describe()
# finding the number of people survived and not survived
titanic_data['Survived'].value_counts()

Data Visualization
sns.set()
# making a count plot for "Survived" column
sns.countplot(x='Survived', data=titanic_data)

titanic_data['Sex'].value_counts()
# making a count plot for "Sex" column
sns.countplot(x='Sex', data=titanic_data)
# number of survivors Gender wise
sns.countplot(x='Sex', hue='Survived', data=titanic_data)
# making a count plot for "Pclass" column
sns.countplot(x='Pclass', data=titanic_data)

# number of survivors based on Pclass
sns.countplot(x='Pclass', hue='Survived', data=titanic_data)

Encoding the Categorical Columns
# converting the categorical columns
titanic_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
titanic_data.head()

Separating features & Target
x = titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
y = titanic_data['Survived']
print(x)
print(y)

Splitting the data into training data & test data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=2)
print(x.shape, x_train.shape, x_test.shape)

Model Training

Logistic Regression
model = LogisticRegression()
# training the Logistic Regression model with training data
model.fit(x_train, y_train)

Model Evaluation
Accuracy Score
# accuracy on training data
x_train_prediction = model.predict(x_train)
print(x_train_prediction)

training_data_accuracy = accuracy_score(y_train, x_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)

# accuracy on test data
x_test_prediction = model.predict(x_test)
print(x_test_prediction)


test_data_accuracy = accuracy_score(y_test, x_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)
