import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
from classifiers import *
import seaborn as sns
import scipy.stats as _stats
import numpy as np
import time
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from subprocess import call
from sklearn.ensemble import RandomForestClassifier



""" MAIN """
register_matplotlib_converters()
data = pd.read_csv('data/pd.csv',  index_col = 'id', header = 1)


selected_data = feature_selection(data, 0.8)

y: np.ndarray = selected_data.pop('class').values #class
X: np.ndarray = selected_data.values
labels = pd.unique(y)

# Dta scaling
X = minMax_data(X)

# Data split
trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

# Normalization
X_normalized = normalization(X)

trnX_normalized = normalization(trnX)
tstX_normalized = normalization(tstX)


#knn(trnX_normalized,trnY,tstX_normalized,tstY)
#knn_feature_selection(selected_data)
#knn_cross_validation(X_normalized, y)

# Decision trees
#decision_trees(trnX_normalized, trnY, tstX_normalized, tstY)
#decision_tree_draw(trnX_normalized, trnY)
#decision_trees_cross_validation(X_normalized, y)

# Random Forests
#random_forests(trnX_normalized, trnY, tstX_normalized, tstY)
#random_forests_cross_validation(X_normalized, y)

# Gradient Boosting
gradient_boosting(trnX_normalized, trnY, tstX_normalized, tstY)
gradient_boosting_cross_validation(X_normalized, y)



