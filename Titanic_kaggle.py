"""Determining who will survive the Titanic"""
import numpy as np
import pandas as pd

# importing everything from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv('data/kaggle_titanic/train.csv')
test_df = pd.read_csv('data/kaggle_titanic/test.csv')

# FEATURE ENGINEERING

# Setting the value of Male to 1 and value of female to 0
train_df['Male'] = [0 if value == 'female' else 1 for value in train_df['Sex']]
test_df['Male'] = [0 if value == 'female' else 1 for value in test_df['Sex']]

# setting the null values to median
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)

# There's a null value in the test_df dataset, so I set it to the median
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)

# cutting values into different bins
train_df['Faregrp'] = pd.cut(train_df['Fare'], [0, 25, 50, 80, 1000], labels=[1,2,3,4], include_lowest=True)
test_df['Faregrp'] = pd.cut(test_df['Fare'], [0, 25, 50, 80, 1000], labels=[1,2,3,4], include_lowest=True)

# Mapping values in the Embarked column so that it can be manipulated later on
train_df['Embarkedgrp'] = np.where(train_df['Embarked'] == 'S', 1,
                                   np.where(train_df['Embarked'] == 'C', 2,
                                   np.where(train_df['Embarked'] == 'Q', 3, 4)))

test_df['Embarkedgrp'] = np.where(test_df['Embarked'] == 'S', 1,
                                   np.where(test_df['Embarked'] == 'C', 2,
                                   np.where(test_df['Embarked'] == 'Q', 3, 4)))

# extracting relevant information from name
train_df['Namegrp'] = train_df['Name'].str.extract(r"(,\s\w*\W)")
test_df['Namegrp'] = test_df['Name'].str.extract(r"(,\s\w*\W)")

train_df['Namegrp'] = np.where(train_df['Namegrp'] == ', Ms.', ', Miss.', train_df['Namegrp'])
train_df['Namegrp'] = np.where(train_df['Namegrp'] == ', Mme.', ', Mrs.', train_df['Namegrp'])

test_df['Namegrp'] = np.where(test_df['Namegrp'] == ', Ms.', ', Miss.', test_df['Namegrp'])
test_df['Namegrp'] = np.where(test_df['Namegrp'] == ', Mme.', ', Mrs.', test_df['Namegrp'])

# 1 = Mrs.
# 2 = Miss.
# 3 = Mr.
# 4 = Other
train_df['Namegrp'] = np.where(train_df['Namegrp'] == ', Mrs.', 1, np.where(train_df['Namegrp'] == ', Miss.', 2, \
                                  np.where(train_df['Namegrp'] == ', Mr.', 3, 4)))

test_df['Namegrp'] = np.where(test_df['Namegrp'] == ', Mrs.', 1, np.where(test_df['Namegrp'] == ', Miss.', 2, \
                                  np.where(test_df['Namegrp'] == ', Mr.', 3, 4)))

# dropping all values that won't be used
train_df = train_df.drop(['Sex', 'Ticket', 'Embarked', 'Fare', 'Name', 'Cabin'], axis=1)
test_df = test_df.drop(['Sex', 'Ticket', 'Embarked', 'Fare', 'Name', 'Cabin'], axis=1)

# Adding Passenger ID to the test dataframe
PassengerId=test_df['PassengerId']
test_df.drop(labels=['PassengerId'],inplace=True,axis=1)

# TRAINING MODELS

# Splitting the data and setting up the train and test sets
from sklearn.model_selection import train_test_split
X = train_df[['Pclass', 'Age', 'SibSp', 'Parch', 'Male', 'Faregrp', 'Embarkedgrp', 'Namegrp']]
Y = train_df['Survived']
X_train, X_test , y_train , y_test = train_test_split(X,Y,test_size=0.25, random_state=52)

# Checking logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
Y_pred_logreg = logreg.predict(X_test)
logreg_score = logreg.score(X_train, y_train)

# Checking SVM
svc = SVC()
svc.fit(X_train, y_train)
Y_pred_svc = svc.predict(X_test)
svc_score = svc.score(X_train, y_train)

# Checking Naive Bayes
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)
y_pred_nb = naive_bayes.predict(X_test)
nb_score = naive_bayes.score(X_train, y_train)

# Checking Random Forest
RFC = RandomForestClassifier()
RFC.fit(X_train, y_train)
Y_pred_rfc = RFC.predict(X_test)
rfc_score = RFC.score(X_train, y_train)

# EXPORTING DATA

# prepping the data to be exported
prediction=RFC.predict(test_df)
sub=pd.DataFrame({'PassengerId':PassengerId,'Survived':prediction})

# exporting as a csv
sub.to_csv('Submission.csv',index=False)