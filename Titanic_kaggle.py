"""Determining who will survive the Titanic"""
import numpy as np
import pandas as pd

# importing everything from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def _main():
    train_df, test_df = _data_intake()

    # going through all of the training data first
    fe_training = _feature_engineering_training(train_df)
    X, y, X_train, X_test, y_train, y_test = _splitting_data(fe_training)
    log_reg = _logistic_regression(X, y, X_train, X_test, y_train, y_test)
    svm = _svm(X, y, X_train, X_test, y_train, y_test)
    nb = _naive_bayes(X, y, X_train, X_test, y_train, y_test)
    rf = _random_forest(X, y, X_train, X_test, y_train, y_test)


    # going through the testing data
    fe_testing = _feature_engineering_testing(test_df)
    X, y, X_train, X_test, y_train, y_test = _splitting_data(fe_testing)
    test_log_reg = _logistic_regression(X, y, X_train, X_test, y_train, y_test)
    test_svm = _svm(X, y, X_train, X_test, y_train, y_test)
    test_nb = _naive_bayes(X, y, X_train, X_test, y_train, y_test)
    test_rf = _random_forest(X, y, X_train, X_test, y_train, y_test)
    exported_data = _exporting_data(test_rf)

    return exported_data


def _data_intake():
    train_df = pd.read_csv('data/kaggle_titanic/train.csv')
    test_df = pd.read_csv('data/kaggle_titanic/test.csv')
    return train_df, test_df

# FEATURE ENGINEERING
def _feature_engineering_training(train_df):
    # Setting the value of Male to 1 and value of female to 0
    train_df['Male'] = [0 if value == 'female' else 1 for value in train_df['Sex']]

    # setting the null values to median
    train_df['Age'].fillna(train_df['Age'].median(), inplace=True)

    # cutting values into different bins
    train_df['Faregrp'] = pd.cut(train_df['Fare'], [0, 25, 50, 80, 1000], labels=[1, 2, 3, 4], include_lowest=True)

    # Mapping values in the Embarked column so that it can be manipulated later on
    train_df['Embarkedgrp'] = np.where(train_df['Embarked'] == 'S', 1,
                                       np.where(train_df['Embarked'] == 'C', 2,
                                       np.where(train_df['Embarked'] == 'Q', 3, 4)))

    # extracting relevant information from name
    train_df['Namegrp'] = train_df['Name'].str.extract(r"(,\s\w*\W)")

    train_df['Namegrp'] = np.where(train_df['Namegrp'] == ', Ms.', ', Miss.', train_df['Namegrp'])
    train_df['Namegrp'] = np.where(train_df['Namegrp'] == ', Mme.', ', Mrs.', train_df['Namegrp'])

    # 1 = Mrs.
    # 2 = Miss.
    # 3 = Mr.
    # 4 = Other
    train_df['Namegrp'] = np.where(train_df['Namegrp'] == ', Mrs.', 1, np.where(train_df['Namegrp'] == ', Miss.', 2, \
                                                                                np.where(train_df['Namegrp'] == ', Mr.',
                                                                                         3, 4)))

    # dropping all values that won't be used
    train_df = train_df.drop(['Sex', 'Ticket', 'Embarked', 'Fare', 'Name', 'Cabin'], axis=1)



def _feature_engineering_testing(test_df):
    # Setting the value of Male to 1 and value of female to 0
    test_df['Male'] = [0 if value == 'female' else 1 for value in test_df['Sex']]

    # setting the null values to median
    test_df['Age'].fillna(test_df['Age'].median(), inplace=True)

    # There's a null value in the test_df dataset, so I set it to the median
    test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)

    # cutting values into different bins
    test_df['Faregrp'] = pd.cut(test_df['Fare'], [0, 25, 50, 80, 1000], labels=[1,2,3,4], include_lowest=True)

    # Mapping values in the Embarked column so that it can be manipulated later on
    test_df['Embarkedgrp'] = np.where(test_df['Embarked'] == 'S', 1,
                                       np.where(test_df['Embarked'] == 'C', 2,
                                       np.where(test_df['Embarked'] == 'Q', 3, 4)))

    # extracting relevant information from name
    test_df['Namegrp'] = test_df['Name'].str.extract(r"(,\s\w*\W)")

    test_df['Namegrp'] = np.where(test_df['Namegrp'] == ', Ms.', ', Miss.', test_df['Namegrp'])
    test_df['Namegrp'] = np.where(test_df['Namegrp'] == ', Mme.', ', Mrs.', test_df['Namegrp'])

    # 1 = Mrs.
    # 2 = Miss.
    # 3 = Mr.
    # 4 = Other
    test_df['Namegrp'] = np.where(test_df['Namegrp'] == ', Mrs.', 1, np.where(test_df['Namegrp'] == ', Miss.', 2, \
                                  np.where(test_df['Namegrp'] == ', Mr.', 3, 4)))

    # dropping all values that won't be used
    test_df = test_df.drop(['Sex', 'Ticket', 'Embarked', 'Fare', 'Name', 'Cabin'], axis=1)

    # Adding Passenger ID to the test dataframe
    PassengerId=test_df['PassengerId']
    test_df.drop(labels=['PassengerId'],inplace=True,axis=1)


# TRAINING MODELS

def _splitting_data(df):
    # Splitting the data and setting up the train and test sets
    X = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Male', 'Faregrp', 'Embarkedgrp', 'Namegrp']]
    y = df['Survived']
    X_train, X_test , y_train , y_test = train_test_split(X,Y,test_size=0.25, random_state=52)
    return X, y, X_train, X_test, y_train, y_test


def _logistic_regression(X, y, X_train, X_test, y_train, y_test):
    # Checking logistic regression
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    Y_pred_logreg = logreg.predict(X_test)
    logreg_score = logreg.score(X_train, y_train)
    return logreg_score


def _svm(X, y, X_train, X_test, y_train, y_test):
    # Checking SVM
    svc = SVC()
    svc.fit(X_train, y_train)
    Y_pred_svc = svc.predict(X_test)
    svc_score = svc.score(X_train, y_train)
    return svc_score


def _naive_bayes(X, y, X_train, X_test, y_train, y_test):
    # Checking Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    nb_score = nb.score(X_train, y_train)
    return nb_score

def _random_forest(X, y, X_train, X_test, y_train, y_test):
    # Checking Random Forest
    RFC = RandomForestClassifier()
    RFC.fit(X_train, y_train)
    Y_pred_rfc = RFC.predict(X_test)
    rfc_score = RFC.score(X_train, y_train)
    return rfc_score

# EXPORTING DATA
def _exporting_data(test_df):
    # prepping the data to be exported
    RFC = RandomForestClassifier()
    prediction=RFC.predict(test_df)
    sub=pd.DataFrame({'PassengerId':PassengerId,'Survived':prediction})

    # exporting as a csv
    return sub.to_csv('Submission.csv',index=False)