import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
from classifiers import *
import seaborn as sns
import scipy.stats as _stats
import numpy as np
import time
import csv
#import xgboost as xgb
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
from sklearn.feature_selection import SelectKBest, f_classif
from mlxtend.frequent_patterns import apriori, association_rules #for ARM


""" MAIN """
register_matplotlib_converters()
data = pd.read_csv('data/pd.csv',  index_col = 'id', header = 1)

#print(data.columns)


selected_data = feature_selection(data, 0.8)


def sep_data(data):
    y: np.ndarray = data.pop('class').values #class
    X: np.ndarray = data.values

    X_columns = data.columns
    labels = pd.unique(y)

    return y,X, X_columns


"""
y, X = sep_data(selected_data)
# Dta scaling
X = minMax_data(X)
#y = sep_data(selected_data)[0]
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
#gradient_boosting(trnX_normalized, trnY, tstX_normalized, tstY)
#gradient_boosting_cross_validation(X_normalized, y)


dtrain = xgb.DMatrix(trnX_normalized, label=trnY)
dtest = xgb.DMatrix(tstX, label=tstY)

param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
num_round = 2

print('running cross validation')
xgb.cv(param, dtrain, num_round, nfold=5,
       metrics={'error'}, seed=0,
       callbacks=[xgb.callback.print_evaluation(show_stdv=True)])

print('running cross validation, disable standard deviation display')
# do cross validation, this will print result out as
# [iteration]  metric_name:mean_value
res = xgb.cv(param, dtrain, num_boost_round=10, nfold=5,
             metrics={'error'}, seed=0,
             callbacks=[xgb.callback.print_evaluation(show_stdv=False),
                        xgb.callback.early_stop(3)])
print(res)
"""



y, X, X_columns = sep_data(data)
#print(X_columns)
#print(type(X_columns))
X_collumns_name = X_columns.tolist()

X_df = pd.DataFrame(X, columns=X_collumns_name)
#print(X_df.columns)
#print(type(X_df.columns))


selector = SelectKBest(f_classif, k=10)
X_new = selector.fit_transform(X_df, y)
names = X_df.columns.values[selector.get_support()]

scores = selector.scores_[selector.get_support()]
#names_scores = list(zip(names, scores))
#ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'F_Scores'])
#ns_df_sorted = ns_df.sort_values(['F_Scores', 'Feat_names'], ascending = [False, True])


X_KBest_df = pd.DataFrame(X_new, columns=names)


X_KBest_copy = X_KBest_df.copy()
for col in X_KBest_copy:
    X_KBest_copy[col] = pd.cut(X_KBest_copy[col], 3, labels=['0','1','2'])
#X_KBest_copy.head(5)
#print(type(X_KBest_copy))
#print(X_KBest_copy.shape)
#print(X_KBest_copy.head(5))



dummylist = []
for att in X_KBest_copy:
    if att in ['a01','a02']: X_KBest_copy[att] = X_KBest_copy[att].astype('category')
    dummylist.append(pd.get_dummies(X_KBest_copy[[att]]))
dummified_df = pd.concat(dummylist, axis=1)
#print(dummified_df.head(5))



minsup = 0.35
frequent_itemsets = apriori(dummified_df, min_support=minsup, use_colnames=True)



minconf = 0.9
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=minconf)
rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
ex = rules[(rules['antecedent_len']>=2)]


export = ex.to_csv(r'out.csv', index=None, header=True)


