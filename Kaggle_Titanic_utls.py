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
	else:
		raise ValueError('Invalid option for the categorical transformer')

	data_ = np.c_[data_num_scaled, data_cat_dummy]

	return data_



def model_trial(data, labels, model):

	X_train, X_test, y_train, y_test = train_test_split(data, labels,
                            random_state=1, test_size=0.25, stratify=labels)

	model.fit(X_train, y_train)
	dict_ = {'train_score': model.score(X_train, y_train),
			  'test_score': model.score(X_test, y_test)}

	return dict_











