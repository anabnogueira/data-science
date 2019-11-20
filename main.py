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
import time, warnings
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle, islice
from sklearn import datasets, metrics, cluster, mixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, accuracy_score
import time, warnings, numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from statistics import mean



""" MAIN """
register_matplotlib_converters()
data = pd.read_csv('data/pd.csv', index_col='id', header=1)


#######################################################################################################################
#### DATA PROCESSING #################################################################################################
#######################################################################################################################


def sep_data(data):
    """
    divide in x and y (removing class)
    returns X, y, and the colunms
    """

    y: np.ndarray = data.pop('class').values #class
    X: np.ndarray = data.values

    X_columns = data.columns
    labels = pd.unique(y)

    return y, X, X_columns



def best_number_features_NB(X, y, dataset):
    if dataset == 1:
        nr_features = [10, 20, 30, 40, 50, 60, 70]
    elif dataset == 2:
        nr_features = [10, 20, 30, 40, 50]
    yvalues = []

    for n in nr_features:
        X_new = SelectKBest(k=n).fit_transform(X, y)
        classifier = GaussianNB()
        scores = cross_val_score(classifier, X_new, y, cv=3)
        print(scores)
        print(scores.mean())
        #print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        yvalues.append(scores.mean())


    plt.plot(nr_features, yvalues, color='g')
    plt.xlabel('Number of features')
    plt.ylabel('Mean scores')
    plt.title('Mean scores for number of features through Naive Bayes classifier')
    plt.show()


"select kBest and save columns names FOR UNSUPERVISED"
def select_Kbest(X, k):
    "select the K best an returns a data frame"
    X_df = pd.DataFrame(X, columns=X_columns_name)
    selector = SelectKBest(f_classif, k=k)
    X_new = selector.fit_transform(X_df, y)
    names = X_df.columns.values[selector.get_support()]
    scores = selector.scores_[selector.get_support()]
    X_KBest_df = pd.DataFrame(X_new, columns=names)

    return X_KBest_df



######################################################################################################################
####   CLASSIFICATION      ##############################################################################
######################################################################################################################
"1º fazer isto para eliminar redundancias"

show_classBalance(data, "Class Balance - 1st dataset")

selected_data = feature_selection(data, 0.9)

y, X, X_columns = sep_data(selected_data)

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.85, stratify=y)

# Dta scaling
trnX = minMax_data(trnX)
tstX = minMax_data(tstX)

compareNB(trnX, tstX, trnY, tstY, "NB classifiers")


trnX_normalized = normalization(trnX)
tstX_normalized = normalization(tstX)


#comparar os 3 e ver scores com o naive bayes
X_smoted, Y_smoted = smote(trnX_normalized,trnY) # USAR ESTE
X_over, Y_over = oversample(trnX_normalized, trnY)
X_under, Y_under = undersample(trnX_normalized, trnY)


print("puro")
NB_crossValidation(trnX_normalized,trnY)
print("smote")
NB_crossValidation(X_smoted,Y_smoted)
print("over")
NB_crossValidation(X_over,Y_over)
print("under")
NB_crossValidation(X_under,Y_under)


# Normalize smote
X_sm_normalized = normalization(X_smoted)

# Normalize Over
X_ov_normalized = normalization(X_over)

# Normalize under
X_ud_normalized = normalization(X_under)


print("smote")
NB_crossValidation(X_sm_normalized,Y_smoted)
print("over")
NB_crossValidation(X_ov_normalized,Y_over)
print("under")
NB_crossValidation(X_ud_normalized,Y_under)


# Uses NB to show confusion matrix

# NB with oversampling without normalisation
#trnX, trnY = oversample(trnX, trnY)
#gaussianNB(trnX, tstX, trnY, tstY, labels=[0, 1])
#compareNB(trnX, tstX, trnY, tstY, "NB classifiers with Oversampling")
"""
# NB with undersampling without normalisation
trnX, trnY = undersample(trnX, trnY)
gaussianNB(trnX, tstX, trnY, tstY, labels, "Gaussian NB with Undersampling")
compareNB(trnX, tstX, trnY, tstY, "NB classifiers with Undersampling")

# NB with undersampling without normalisation
trnX, trnY = smote(trnX, trnY)
gaussianNB(trnX, tstX, trnY, tstY, labels, "Gaussian NB with SMOTE")
compareNB(trnX, tstX, trnY, tstY, "NB classifiers with SMOTE")

"""


#print(X)
#y = sep_data(selected_data)[0]
# Data split

# Normalization
#X_normalized = normalization(X)

#cnf_mtx = gaussianNB(trnX, tstX, trnY, tstY, "labels", "nome")



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



######################################################################################################################
####   ASSOCIATION RULES  #############################################################################################
######################################################################################################################
#Data preperation for ASSOCIATION RULES  """
""""
y, X, X_columns = sep_data(selected_data)

X_columns_name = X_columns.tolist()

#X_df = pd.DataFrame(X, columns=X_columns_name)


#X_df = select_Kbest(X_df, best_nr_features)

#best_number_features_NB(X, y, 1)
#X_k_best_df = select_Kbest(X, 20)
#support_cut_qcut_compare(X_k_best_df)
#lift_cut_qcut_compare(X_k_best_df)

X_df_cut = cut(X_k10_best_df, 3, ['0','1','2'])
dummified_df_cut = dummyfication(X_df_cut)
freqt_assRule_mining(dummified_df_cut)

X_df_qcut = qcut(X_k10_best_df, 3, ['0','1','2'])
dummified_df_qcut = dummyfication(X_df_qcut)
freqt_assRule_mining(dummified_df_qcut)

"""




######################################################################################################################
####   CLUSTER   #############################################################################################
######################################################################################################################


#CHAMAR AQUI AS FUNÇÕES DO CLUSTER

"""
*********** 2ND DATASET ***********
***********************************

"""

def second_dataSet():
    # add header column
    header = []
    for i in range(0, 54):
        header.append(str(i))
    header.append('class')
    dataset_two = pd.read_csv('data/covtype.csv', header=None, names=header)
    return dataset_two

datasetTwo = second_dataSet()

show_classBalance(datasetTwo, "Class Balance - 2nd dataset")

#heatmap(datasetTwo)

y2, X2, X2_columns = sep_data(datasetTwo)
#print(X2.shape)

#X2_df = pd.DataFrame(X2, columns=X2_columns)

#filtered_X2 = filter_columns(X2_df, 0.6)
#print(filtered_X2.shape)


# Data split
trnX2, tstX2, trnY2, tstY2 = train_test_split(X2, y2, train_size=0.7, stratify=y2)
"""
#X_smoted, Y_smoted = smote(trnX2,trnY2)
#X_over, Y_over = oversample(trnX2, trnY2)
X_under, Y_under = undersample(trnX2, trnY2)
#X_under2, Y_under2 = under_ClusterCentroids(trnX2, trnY2) #JA NAO E, MUDAR 

#clf = GaussianNB()
#scores1 = cross_val_score(clf, X_under, Y_under, cv=5)

best_number_features_NB(X_under, Y_under, 2)

"""

"""
def stratifiedShuffleSplit(X,y):
    accValue_list = []

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
    sss.get_n_splits(X_smoted, Y_smoted)
    for train_index, test_index in sss.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accValue = accuracy_score(y_test, y_pred)
        #print(accValue)
        accValue_list.append(accValue)

    mean_value = mean(accValue_list)
    return mean_value


acc_valueSmote = stratifiedShuffleSplit(X_smoted, Y_smoted)
print(acc_valueSmote)

acc_valueOver = stratifiedShuffleSplit(X_over, Y_over)
print(acc_valueOver)

acc_undersample = stratifiedShuffleSplit(X_under, Y_under)
print(acc_undersample)
"""













