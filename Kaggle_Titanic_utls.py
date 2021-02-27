import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score




def base_pip(model, data, labels, imputer_s='median'):
    # Pipeline + model
    # Returns (score on train, score on test)
    numeric_features = ['Age', 'SibSp', 'Fare', 'Parch']
    
    if (imputer_s == 'median'):
        imputer = SimpleImputer(strategy='median')
    elif (imputer_s == 'mean'):
        imputer = SimpleImputer(strategy='mean')
    elif (imputer_s == 'zeros'):
        imputer = SimpleImputer(strategy='constant', fill_value=0)
    elif (imputer_s == 'function'):
        imputer = IterativeImputer()
        
    numeric_transformer = Pipeline(steps=[
        ('imputer', imputer),
        ('scaler', StandardScaler())])

    categorical_features = ['Embarked', 'Sex', 'Pclass', 'Title']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])

    
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.15,
                                                                random_state=0, stratify=labels)

    
    clf.fit(X_train, y_train)
    score_train, score_test = clf.score(X_train, y_train), clf.score(X_test, y_test)
    
    return score_train, score_test