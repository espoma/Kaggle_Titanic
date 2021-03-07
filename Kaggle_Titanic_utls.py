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


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'




def find_features_by_type(data):

	numerical_features = list(data.columns[data.dtypes != 'object'])
	categorical_features = list(data.columns[data.dtypes == 'object'])

	return numerical_features, categorical_features


def get_features_by_type(data):

	num, cat = find_features_by_type(data)

	return data.loc[:, num], data.loc[:, cat]


def preprocessing(data_num, data_cat, imputer_num='median', scaler=StandardScaler(),
	imputer_cat='mode', transformer='dummies'):

	if (imputer_num == 'median'):
		value_num = data_num.median()
	elif (imputer_num == 'mean'):
		value_num = data_num.mean()
	elif (imputer_num == 'zero'):
		value_num = 0
	else:
		raise ValueError('Invalid option for numerical imputer')

	if (imputer_cat == 'mode'):
		value_cat = data_cat.mode()
	else:
		raise ValueError('Invalid option for categorical imputer')

	data_num.fillna(value_num, inplace=True)
	data_cat.fillna(value_cat, inplace=True)

	data_num_scaled = scaler.fit_transform(data_num)

	if (transformer == 'dummies'):
		data_cat_dummy = pd.get_dummies(data_cat)

	data_ = np.c_[data_num_scaled, data_cat_dummy]

	return data_



def model_trial(data, labels, model):

	X_train, X_test, y_train, y_test = train_test_split(data, labels,
                            random_state=1, test_size=0.25, stratify=labels)

	model.fit(X_train, y_train)
	dict_ = {'train_score': model.score(X_train, y_train),
			  'test_score': model.score(X_test, y_test)}

	return dict_








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
        X_train_tr, X_test_tr, y_train, y_test = train_test_split(data, labels, test_size=0.20,
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













# def imputer_func(data, num_imputer_s='median', cat_imputer_s='most_frequent',
#     numerical_features = ['Age', 'SibSp', 'Fare', 'Parch'],
#     categorical_features = ['Embarked', 'Sex', 'Pclass', 'Title']):


#     if (num_imputer_s == 'median'):
#         num_imputer = SimpleImputer(strategy='median')
#     elif (num_imputer_s == 'mean'):
#         num_imputer = SimpleImputer(strategy='mean')
#     elif (num_imputer_s == 'zeros'):
#         num_imputer = SimpleImputer(strategy='constant', fill_value=0)
#     elif (num_imputer_s == 'function'):
#         num_imputer = IterativeImputer()    
#     else:
#         raise ValueError('Invalid imputer method provided')
#         return 0

#     cat_imputer = SimpleImputer(strategy=cat_imputer_s)

#     numerical_pip = Pipeline(steps=[('imputer', num_imputer)])
#     categorical_pip = Pipeline(steps=[('imputer', cat_imputer)])


#     imputation = ColumnTransformer(
#     transformers=[
#         ('num', numerical_pip, numerical_features),
#         ('cat', categorical_pip, categorical_features)])


#     data_transformed = Pipeline(steps=[('imputation', imputation)]).fit_transform(data)
#     data_transformed = 

#     return data_transformed




def transformer_func(data, 
    numerical_features = ['Age', 'SibSp', 'Fare', 'Parch'],
    categorical_features = ['Embarked', 'Sex', 'Pclass', 'Title']):

    
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    numerical_pip = Pipeline(steps=[('transformer', numerical_transformer)])
    categorical_pip = Pipeline(steps=[('transformer', categorical_transformer)])


    imputation = ColumnTransformer(
    transformers=[
        ('num', numerical_pip, numerical_features),
        ('cat', categorical_pip, categorical_features)])


    data_transformed = Pipeline(steps=[('imputation', imputation)]).fit_transform(data)

    return data_transformed




def base_pip(model, data, labels, option='train_test', param_grid=None, 
    cv=3, numerical_features = ['Age', 'SibSp', 'Fare', 'Parch'],
    categorical_features = ['Embarked', 'Sex', 'Pclass', 'Title']):
    # Pipeline + model or pipeline + grid search CV
    # Returns (score on train, score on test) or (best_params, best_score, best_estimator)
    
    
    clf = model
    


    # Baseline
    if (option == 'train_test'):            
    
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

        #print('train_tr shape with gscv', data.shape)
        gscv = GridSearchCV(clf, cv=cv, n_jobs=-3, param_grid=param_grid, scoring='accuracy', verbose=1)
        gscv.fit(data, labels)
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