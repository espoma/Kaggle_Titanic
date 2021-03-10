import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pandas.plotting import scatter_matrix

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
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy, make_scorer


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




def preprocessing(data_num, data_cat, imputer_num='median', scaling='yes',
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



	data_num_scaled = pd.DataFrame(scaler.fit_transform(data_num))

	if (transformer == 'dummies'):
		data_cat_dummy = pd.get_dummies(data_cat)
	else:
		raise ValueError('Invalid option for the categorical transformer')

	data_ = pd.concat([data_num_scaled, data_cat_dummy], axis=1, join='inner')


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











