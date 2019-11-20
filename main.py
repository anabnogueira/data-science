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

#######################################################################################################################
#### FUNCTION FOR PROCESSING OF UNSUPERVISED  #########################################################################
#######################################################################################################################


def best_number_features_NB(X, y):
    nr_features = [10, 20, 30, 40, 50, 60, 70]

    for n in nr_features:
        X_new = SelectKBest(k=n).fit_transform(X, y)
        classifier = GaussianNB()
        scores = cross_val_score(classifier, X_new, y, cv=3)
        #print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        yvalues.append(scores.mean())


    plt.plot(nr_features, yvalues, color='g')
    plt.xlabel('Number of features')
    plt.ylabel('Mean scores')
    plt.title('Mean scores for number of features through Naive Bayes classifier')
    plt.show()


"select kBest and save columns names FOR UNSUPERVISED Assocation RUles"
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
####   CLASSIFICATION      ###########################################################################################
######################################################################################################################

"""
*********** 1st DATASET ***************************************************************************************************

"""
data = pd.read_csv('data/pd.csv', index_col='id', header=1)
#print(data.groupby('class').size())

def processing_and_classification_1st(data):

    selected_data = feature_selection(data, 0.9)

    y, X, X_columns = sep_data(selected_data)

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.85, stratify=y)

    # Dta scaling
    trnX = minMax_data(trnX)
    tstX = minMax_data(tstX)

    #compareNB(trnX, tstX, trnY, tstY, "NB classifiers")

    trnX_normalized = normalization(trnX)
    tstX_normalized = normalization(tstX)



    #comparar os 3 e ver scores com o naive bayes
    X_smoted, Y_smoted = smote(trnX_normalized,trnY) # USAR ESTE
    #X_over, Y_over = oversample(trnX_normalized, trnY)
    #X_under, Y_under = undersample(trnX_normalized, trnY)


    "Naive Bayes using training and test set"
    #compareNB(trnX_normalized, tstX_normalized, trnY, tstY, "NB classifiers with normalization")


    #compareNB(X_smoted, tstX_normalized, Y_smoted, tstY, "NB classifiers with smote")
    #compareNB(X_over, tstX_normalized, Y_over, tstY, "NB classifiers with oversampling")
    #compareNB(X_under, tstX_normalized, Y_under, tstY, "NB classifiers with undersampling")

    "confusion matrix"
    #gaussianNB(X_under, tstX_normalized, Y_under, tstY, labels = [0, 1])


    "KNN with cross validation for the training data "
    #knn_feature_selection(selected_data)

    #knn_cross_validation(X_smoted, Y_smoted) #with smote normalized
    #knn_cross_validation(trnX_normalized,trnY) #normalized
    #knn_cross_validation(trnX,trnY) #without normalization


    " Decision trees "
    #decision_tree_draw(trnX_normalized, trnY)
    #decision_trees_cross_validation(trnX_normalized,trnY)
    #decision_trees_cross_validation(trnX,trnY)

    #decision_trees_cross_validation(X_smoted, Y_smoted)

    #decision_trees_feature_selection(data, 0.015, 25, 'accuracy')


    " Random Forests "
    #random_forests_cross_validation(trnX_normalized, trnY)
    #random_forests_cross_validation(X_smoted, Y_smoted)

    " xgBOOST"
    #xgboost(X_smoted, Y_smoted, tstX, tstY)

    # Gradient Boosting
    #gradient_boosting_cross_validation(X_normalized, trnY)


"""
*********** 2ND DATASET ***************************************************************************************************

"""

def second_dataSet():
    # add header column
    header = []
    for i in range(0, 54):
        header.append(str(i))
    header.append('class')
    dataset_two = pd.read_csv('data/covtype.csv', header=None, names=header)
    return dataset_two


def processing_2nd(dataset_two):

    #datasetTwo = second_dataSet()

    # show_classBalance(datasetTwo, "Class Balance - 2nd dataset")
    # heatmap(datasetTwo)

    y2, X2, X2_columns = sep_data(dataset_two)
    # print(X2.shape)

    X2_df = pd.DataFrame(X2, columns=X2_columns)

    # nao fazemos feature slection no segundo data set
    # filtered_X2 = filter_columns(X2_df, 0.9)
    # print(filtered_X2.shape)

    # Data split
    trnX2, tstX2, trnY2, tstY2 = train_test_split(X2, y2, train_size=0.7, stratify=y2)

    # Dta scaling
    trnX2 = minMax_data(trnX2)
    tstX2 = minMax_data(tstX2)

    trnX2_normalized = normalization(trnX2)
    tstX2_normalized = normalization(tstX2)

    # X2_smoted, Y2_smoted = smote(trnX2_normalized,trnY2)
    # X2_over, Y2_over = oversample(trnX2_normalized, trnY2)
    # X2_under2, Y2_under2 = undersample_AllNN(trnX2_normalized, trnY2) #JA NAO E, MUDAR

    X2_under, Y2_under = undersample(trnX2_normalized, trnY2)  # USAMOS ESTE

    return X2_under, Y2_under, tstX2, tstY2


def classification_2nd():
    datasetTwo = second_dataSet()
    X2_under, Y2_under, tstX2, tstY2 = processing_2nd(datasetTwo)

    #clf = GaussianNB()
    #scores1 = cross_val_score(clf, X_under, Y_under, cv=5)

    "confusion matrix"
    #gaussianNB(X2_under, tstX2_normalized, Y2_under, tstY2, labels = [1,2,3,4,5,6,7])

    "KNN with cross validation for the training data "
    #knn_cross_validation(X2_under, Y2_under) #with smote normalized

    " Decision trees "
    #decision_trees_cross_validation(X2_under, Y2_under)

    " Random Forests "
    #random_forests_cross_validation(X2_under, Y2_under)

    " xgBOOST"
    #xgboost(X2_under,Y2_under, tstX2, tstY2)



"RUN"
processing_and_classification_1st(data)
classification_2nd()



######################################################################################################################
####   ASSOCIATION RULES  #############################################################################################
######################################################################################################################
#Data preperation for ASSOCIATION RULES  """
""""
y, X, X_columns = sep_data(selected_data)

X_columns_name = X_columns.tolist()

#X_df = pd.DataFrame(X, columns=X_columns_name)


#X_df = select_Kbest(X_df, best_nr_features)

#best_number_features_NB(X, y)
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

