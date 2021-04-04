import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pandas.plotting import scatter_matrix
import pickle as pkl
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
from sklearn.ensemble import RandomForestClassifier, \
BaggingClassifier, VotingClassifier, ExtraTreesClassifier


#from sklearn.metrics import accuracy, make_scorer

PATH_DATA = Path('.')


# Customize print statements {https://stackoverflow.com/questions/8924173/how-do-i-print-bold-text-in-python}
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



def prepare_test_data(test, columns_to_drop=['Cabin', 'Ticket', 'PassengerId']):

	test_ids = test.PassengerId
	test_ = test.drop(columns_to_drop, axis=1)

	test_title = get_title(test_)
	test_['Title'] = test_title
	test_.drop('Name', axis=1, inplace=True)

	return test_, test_ids


def get_title(data, col='Name'):
    
    title =  data[col].apply(lambda x:x.split(',')[1].split('.')[0].strip())
    
    return title


def find_features_by_type(data):

	numerical_features = list(data.columns[data.dtypes != 'object'])
	categorical_features = list(data.columns[data.dtypes == 'object'])

	return numerical_features, categorical_features


def get_features_by_type(data):

	num, cat = find_features_by_type(data)

	return data.loc[:, num], data.loc[:, cat]




def preprocessing(data_num, data_cat, imputer_num='median',
				scaler=StandardScaler(), imputer_cat='mode', transformer='dummies'):

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


	if (scaler == 'no') or (scaler == None):
		pass
	else:
		data_num = pd.DataFrame(scaler.fit_transform(data_num))



	if (transformer == 'no') or (transformer == None): 
		pass
	elif (transformer == 'dummies'):
		data_cat = pd.get_dummies(data_cat)
	else:
		raise ValueError('Invalid option for the categorical transformer')


	data_ = pd.concat([data_num, data_cat], axis=1, join='inner')


	return data_, value_num, value_cat




def test_preprocessing(test, value_num, value_cat, scaler=StandardScaler(), transformer='dummies'):

	test_num, test_cat = get_features_by_type(test)

	test_num.fillna(value_num, inplace=True)
	test_cat.fillna(value_cat, inplace=True)

	test_num_scaled = pd.DataFrame(scaler.fit_transform(test_num))

	if (transformer == 'dummies'):
		test_cat_dummy = pd.get_dummies(test_cat)
	else:
		raise ValueError('Invalid option for the categorical test transformer')


	test_ = pd.concat([test_num_scaled, test_cat_dummy], axis=1, join='inner')

	return test_



def model_trial(data, labels, model):

	X_train, X_test, y_train, y_test = train_test_split(data, labels,
                            random_state=123, test_size=0.25, stratify=labels)

	model.fit(X_train, y_train)
	dict_ = {'train_score': model.score(X_train, y_train),
			  'test_score': model.score(X_test, y_test)}

	return dict_





def save_model(model, name_model, overwrite='no', PATH_DATA=PATH_DATA):
    """
    Saves a sklearn model

    Parameters
    ------------
    model: sklearn model, it's the model to save;
    name_model: string, it's the name of the pickle file;
    overwrite: string, whether to overwrite ('yes') or not ('no');
    PATH_DATA: pathlib object, the relative path to save the file to.

    Returns:
    - 0 if executed correctly;
    - 1 if "overwrite" is not 'yes' or 'no'.
    """
    if (overwrite == 'yes'):
        with open(PATH_DATA / f"{name_model}.pkl", 'wb') as pkl_file:
            pkl.dump(model, pkl_file)
    elif (overwrite == 'no'):
        if (not Path(PATH_DATA / name_model).is_file()):
            with open(PATH_DATA / f"{name_model}.pkl", 'wb') as pkl_file:
                pkl.dump(model, pkl_file)
    else:
        raise ValueError("specify whether to overwrite ('yes') or not ('no')")
        return 1

    return 0



def load_model(name_model, PATH_DATA=PATH_DATA):
    """
    Loads a sklearn model

    Parameters
    name_model: string, name of the sklearn model to load;
    PATH_DATA: pathlib object, the relative path to save the file to.

    Returns:
    - sklearn model, if it exists in PATH_DATA;
    - None if the sklearn model doesn't exist.
    """
    try:
        model_loaded = pkl.load(open(PATH_DATA / f'{name_model}.pkl', 'rb'))
        return model_loaded
    except FileNotFoundError:
        return None




def create_subfile_titanic(test_aligned, test_ids, model, name_model, PATH_DATA=PATH_DATA):
	"""
    Creates a submission file ready for kaggle
    
    Parameters
    ------------
    test_aligned: pandas dataframe, the test set aligned to the training set;
    test_ids: list or pandas series, the passengers' id;
    model: sklearn model, fit to the training set;
    name_model: string, name of the sklearn model;
    PATH_DATA: pathlib object, relative path to create the submission file to.
    
    Returns:
    - 0 if executed correctly;
    - 1 if test_ids is not a list or a pandas series.
    """

	model_loaded = load_model(name_model)
	if (model_loaded == None):
		predictions = model.predict(test_aligned)
	else:
		predictions = model_loaded.predict(test_aligned)

	if isinstance(test_ids, pd.Series):
		submission = np.c_[test_ids.values, predictions]
	elif isinstance(test_ids, (np.array, list)):
		submission = np.c_[test_ids, predictions]
	else:
		raise TypeError('test_ids is not a valid type')
		return 1

	pd.DataFrame(submission).to_csv(PATH_DATA / \
		f"submission_{name_model}_titanic.csv", header=['PassengerId', 'Survived'],\
		index=False)


	return 0




def try_model(X, y, model, name_model, random_state=100, print_score=True, dump_model='yes', PATH_DATA=PATH_DATA):
    """
    Fits a sklearn model to X, y (or loads it if exists in PATH_DATA) and 
    prints the train and test score. It also saves it if "save_model" = 'yes'

    Parameters
    ------------
    X: pandas dataframe, training set;
    y: pandas series, target variable;
    model: sklearn model, to fit on the training set;
    name_model: string, name of the sklearn model;
    random_state: int, random_state of train_test_split;
    score: bool, whether to print the score or not;
    save_model: string, whether to save the model or not;
    PATH_DATA: pathlib object, the relative path to save the file to.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y,\
            test_size=0.25, random_state=random_state)

    model_loaded = load_model(name_model)
    if (model_loaded == None):
        model.fit(X_train, y_train)
        print(model)
        if (dump_model == 'yes'):
            save_model(model, name_model)   
        if (print_score):
            print(f'model: {name_model}')
            print(f'train score: {model.score(X_train, y_train):.4f}\ntest score: {model.score(X_test, y_test):.4f}')

    else:
        if (print_score):
            print(f'model: {name_model}')
            print(f'train score: {model_loaded.score(X_train, y_train):.4f}\ntest score: {model_loaded.score(X_test, y_test):.4f}')






def try_model_old(X, y, model, name_model, PATH_DATA, random_state=100, score=True, save_model='yes'):

    X_train, X_test, y_train, y_test = train_test_split(X, y,\
            test_size=0.25, random_state=random_state, stratify=y)

    # If model is stored in the folder, load it and return the model
    try:
        model_loaded = pkl.load(open(PATH_DATA / f'{name_model}.pkl', 'rb'))
        if (score):
            print(f'model: {name_model}')
            print(f'train score: {model_loaded.score(X_train, y_train):.4f}\ntest score: {model_loaded.score(X_test, y_test):.4f}')


    # Check if the model was trained on a different PCA decomposition
    except ValueError:
        print(f'ValueError: most likely an existent {name_model} was trained on a different number of components. Delete existing {name_model} and retrain it with the current PCA-decomposed training set')


    # If model not stored in the folder, train it on X_train, y_train
    except FileNotFoundError:
        model.fit(X_train, y_train)
        print(model)
        if (not Path(PATH_DATA / name_model).is_file()) and (save_model == 'yes'):
            with open(PATH_DATA / f"{name_model}.pkl", 'wb') as pkl_file:
                pkl.dump(model, pkl_file)
    
        if (score):
            print(f'model: {name_model}')
            print(f'train score: {model.score(X_train, y_train):.4f}\ntest score: {model.score(X_test, y_test):.4f}')




