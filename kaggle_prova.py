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
#from sklearn.metrics import accuracy, make_scorer




def base_pip(model, data, labels, imputer_s='median', option='train_test', 
    param_grid=None, cv=3, numeric_features = ['Age', 'SibSp', 'Fare', 'Parch'],
    categorical_features = ['Embarked', 'Sex', 'Pclass', 'Title'], X_unknown=None):
    # Pipeline + model or pipeline + grid search CV
    # Returns (score on train, score on test) or (best_params, best_score, best_estimator)
    

    #numeric_features = ['Age', 'SibSp', 'Fare', 'Parch']
    
    # Mutual bit
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

    #categorical_features = ['Embarked', 'Sex', 'Pclass', 'Title']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])


    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

  



    ########### ATTENZIONE: LE DUE RIGHE DI SEGUITO DEVI METTERLE DOPO IF OPTION == TRAIN TEST!!!!! ##########
    ########### IN OPTION == GRID SEARCH DEVI INVECE APPLICARE LA PIPELINE A TUTTO IL DATASET (A DATA) 
    # X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.15,
    #                                                                random_state=0, stratify=labels)
    # X_train_tr = pipeline.fit_transform(X_train)
    # X_test_tr = pipeline.transform(X_test)

    
    clf = model
    


    # Baseline
    if (option == 'train_test'):            
    
        data = pipeline.fit_transform(data)
        X_train_tr, X_test_tr, y_train, y_test = train_test_split(data, labels, test_size=0.15,
                                                                    random_state=0, stratify=labels)
        # X_train_tr = pipeline.fit_transform(X_train)
        # X_test_tr = pipeline.transform(X_test)
        print('train_tr shape',X_train_tr.shape)
        clf.fit(X_train_tr, y_train)
        score_train, score_test = clf.score(X_train_tr, y_train), clf.score(X_test_tr, y_test)
    
        return score_train, score_test





    # Grid Search
    elif (option == 'grid_search'):
        
        X_train_tr = pipeline.fit_transform(data)  ### NOTA: QUI STAI FITTANDO SU TUTTI I DATI E TRASFORMANDO, MENTRE PRIMA STAVI FITTANDO SUL TRAIN E TRASFORMANDO IL TEST: ASSICURATI CHE NON SIA UN PROBLEMA  
        y_train = labels
        print('train_tr shape with gscv', X_train_tr.shape)
        gscv = GridSearchCV(clf, cv=cv, n_jobs=-3, param_grid=param_grid, scoring='accuracy', verbose=1)
        gscv.fit(X_train_tr, y_train)
        best_params = gscv.best_params_
        best_score = gscv.best_score_
        best_estimator = gscv.best_estimator_
        #cv_results = gscv.cv_results_

        return {'best_param': best_params, 'best_score': best_score, 
        'best_estimator': best_estimator, 'pipeline': pipeline, 
        'pipeline_params': pipeline.get_params()}


    else:

        raise ValueError("No valid option provided, insert 'test_train' or 'grid_search'")
        return 0