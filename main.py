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
    yvalues = []
    for n in nr_features:
        X_new = SelectKBest(k=n).fit_transform(X, y)
        classifier = GaussianNB()
        scores = cross_val_score(classifier, X_new, y, cv=3)
        #print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        yvalues.append(scores.mean())


    plt.plot(nr_features, yvalues, color='g')
    plt.xlabel('Number of features')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for number of features with t = 0.8')
    plt.show()




######################################################################################################################
####   CLASSIFICATION      ###########################################################################################
######################################################################################################################

"""
*********** 1st DATASET ***************************************************************************************************

"""
data = pd.read_csv('data/pd.csv', index_col='id', header=1)
#print(data.groupby('class').size())

def processing_and_classification_1st(data):
    #group_formant = data.iloc[:,25:29]
    #heatmap(group_formant)

    #show_classBalance(data, "Class distribution")
    print(data.shape)
    selected_data = feature_selection(data, 0.9)

    y, X, X_columns = sep_data(selected_data)
    #print(len(X_columns))

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.85, stratify=y) #so para NB

    # Dta scaling
    #trnX = minMax_data(trnX) #NB
    #tstX = minMax_data(tstX) #BB

    X_minMax =  minMax_data(X)

    #compareNB(trnX, tstX, trnY, tstY, "NB classifiers")

    #comparar os 3 e ver scores com o naive bayes
    X_smoted, Y_smoted = smote(X_minMax,y) # USAR ESTE
    #X_over, Y_over = oversample(trnX_normalized, trnY)
    #X_under, Y_under = undersample(trnX_normalized, trnY)

    "Naive Bayes using training and test set"
    #compareNB(X_smoted, tstX, Y_smoted, tstY, "NB classifiers with smote")
    #compareNB(X_over, tstX_normalized, Y_over, tstY, "NB classifiers with oversampling")
    #compareNB(X_under, tstX_normalized, Y_under, tstY, "NB classifiers with undersampling")

    "confusion matrix"
    #gaussianNB(X_smoted, tstX, Y_smoted, tstY, labels = [0, 1])

    "KNN with cross validation for the training data "
    #knn_cross_validation(X,y)
    #knn_cross_validation(X_minMax,y)
    #knn_cross_validation(X_smoted, Y_smoted)

    " Decision trees "
    #decision_tree_draw(X_smoted, Y_smoted)
    #decision_trees_cross_validation(X_smoted,Y_smoted)
    #decision_trees_cross_validation(trnX,trnY)

    #decision_trees_feature_selection(data, 0.02, 25, 'accuracy')

    " Random Forests "
    #random_forests_cross_validation(X_smoted, Y_smoted)
    #random_forest_feature_selection(data, 20, 50, 'accuracy')

    " xgBOOST"
    #xgboost(X_smoted, Y_smoted)



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

    # show_classBalance(datasetTwo, "Class Balance - 2nd dataset")
    # heatmap(datasetTwo)

    y2, X2, X2_columns = sep_data(dataset_two)
    # print(X2.shape)

    X2_df = pd.DataFrame(X2, columns=X2_columns)

    # Data split
    trnX2, tstX2, trnY2, tstY2 = train_test_split(X2, y2, train_size=0.7, stratify=y2) #NB

    # Dta scaling
    trnX2 = minMax_data(trnX2)
    tstX2 = minMax_data(tstX2)


    # X2_smoted, Y2_smoted = smote(trnX2_normalized,trnY2)
    # X2_over, Y2_over = oversample(trnX2_normalized, trnY2)
    # X2_under2, Y2_under2 = undersample_AllNN(trnX2_normalized, trnY2) #JA NAO E, MUDAR

    X2_under, Y2_under = undersample(trnX2, trnY2)  # USAMOS ESTE

    return X2_under, Y2_under, tstX2, tstY2


def classification_2nd():
    datasetTwo = second_dataSet()

    #show_classBalance(datasetTwo, "Class distribution")


    X2_under, Y2_under, tstX2, tstY2 = processing_2nd(datasetTwo)

    #clf = GaussianNB()
    #scores1 = cross_val_score(clf, X_under, Y_under, cv=5)

    "confusion matrix"
    #gaussianNB(X2_under, tstX2, Y2_under, tstY2, labels = [1,2,3,4,5,6,7])

    "KNN with cross validation for the training data "
    #knn_cross_validation(X2_under, Y2_under) #with smote normalized

    " Decision trees "
    #decision_trees_cross_validation(X2_under, Y2_under)
    #decision_tree_draw(X2_under, Y2_under)


    " Random Forests "
    #random_forests_cross_validation(X2_under, Y2_under)

    " xgBOOST"
    #xgboost(X2_under,Y2_under)



" Run for supervised "
#processing_and_classification_1st(data)
#classification_2nd()



######################################################################################################################
####   ASSOCIATION RULES  #############################################################################################
######################################################################################################################
#Data preperation for ASSOCIATION RULES  """

"""
*********** 1st DATASET ***************************************************************************************************

"""

"select kBest and save columns names FOR UNSUPERVISED Assocation RUles"
def select_Kbest(X_df, y, k):
    "select the K best an returns a data frame"

    selector = SelectKBest(f_classif, k=k)
    X_new_df = selector.fit_transform(X_df, y)

    #vou arranjar agora o nome das colunas
    names_columns = X_df.columns.values[selector.get_support()]
    #scores = selector.scores_[selector.get_support()]

    #crio data frame com dados e nomes colunas
    X_KBest_df = pd.DataFrame(X_new_df, columns=names_columns)

    return X_KBest_df




def associationRules_1st(data):
    group_test = data.iloc[:, 1:4]
    group_test2 = data.iloc[:, 6:8]

    group_baseline = data.iloc[:, 1:22]
    group_intensity = data.iloc[:, 22:25]
    group_formant = data.iloc[:, 25:29]
    group_bandwidth = data.iloc[:, 29:33]
    group_vocalfold = data.iloc[:, 33:55]
    group_mfcc = data.iloc[:, 55:139]
    group_wavelet = data.iloc[:, 139:-1]

    groupA = data.iloc[:, 1:33]
    groups = [group_formant]

    y, X, X_columns = sep_data(data)
    X_columns_name = X_columns.tolist()

    """for X_df in groups:
        #X: np.ndarray = group.values
        #X_df = pd.DataFrame(X, columns=X_columns_name)
        print("x_df")
        X_df_cut = cut(X_df, 3, ['0', '1', '2'])
        dummified_df_cut = dummyfication(X_df_cut)
        #freqt_assRule_mining(dummified_df_cut)
        assocRules_plot(dummified_df_cut)
"""
    X_df = pd.DataFrame(X, columns=X_columns_name)
    #print(X_df)

    #X_k_best_df = select_Kbest(X_df, y, 20)
    #print(X_k_best_df)

    "vejo os graficos para escolher cut ou qcut "
    support_cut_qcut_compare(X_df)
    lift_cut_qcut_compare(X_df)

    " agora com o qcut imprimo a tabea de AR"
    """X_df_qcut = qcut(X_df, 3, ['0','1','2'])
    print(X_df_qcut)
    dummified_df_qcut = dummyfication(X_df_qcut)
    print(dummified_df_qcut)
    freqt_assRule_mining(dummified_df_qcut)
    assocRules_plot(dummified_df_qcut)"""


def associationRules_2nd():
    datasetTwo = second_dataSet()

    y2, X2, X2_columns = sep_data(datasetTwo)
    X2_columns_name = X2_columns.tolist()

    trnX2, tstX2, trnY2, tstY2 = train_test_split(X2, y2, train_size=0.05, stratify=y2) #NB

    X2_df = pd.DataFrame(trnX2, columns=X2_columns_name)

    X2_df_cut = cut(X2_df, 3, ['0', '1', '2'])
    dummified2_df_cut = dummyfication(X2_df_cut)
    #freqt_assRule_mining(dummified_df_cut)
    assocRules_plot(dummified2_df_cut)



" Run for Association Rules "
#associationRules_1st(data)
associationRules_2nd()



######################################################################################################################
####   CLUSTER   #############################################################################################
######################################################################################################################


"""
*********** 1st DATASET ***************************************************************************************************
"""

def unsupervised_1st(data):

    #new_data = data.groupby(['id']).mean()

    y, X, X_columns = sep_data(data)
    #y, X, X_columns = sep_data(data)


    X = minMax_data(X)

    X_columns_name = X_columns.tolist()

    X_df = pd.DataFrame(X, columns=X_columns_name)

    #kmeans_NrClusters_inertia(X)

    # return y_pred to be used in pca graph
    #y_pred_clustering = k_means_sillhoutte(X, 8)

    #k_means_adjusted_rand_score(X, y, 8)

    #X_k2_best_df = select_Kbest(X_df, y, 2)
    #X_k2_best: np.ndarray = X_k2_best_df.values

    #clusters_plot(X)
    #k_means(X)

    #pca_graph(X, y_pred_clustering)

    #pca_variance(X)




"""
*********** 2nd DATASET ***************************************************************************************************
"""


def unsupervised_2nd():
    datasetTwo = second_dataSet()
    y2, X2, X2_columns = sep_data(datasetTwo)

    trnX2, tstX2, trnY2, tstY2 = train_test_split(X2, y2, train_size=0.5, stratify=y2) #NB
    print(len(trnX2))

    X2 = minMax_data(trnX2)

    X2_columns_name = X2_columns.tolist()

    #X2_under, Y2_under = undersample(X2_normalized, y2)


    #X2_df = pd.DataFrame(X2_under, columns=X2_columns_name)

    #kmeans_NrClusters_inertia(X2)

    # return y_pred to be used in pca graph
    #y2_pred_clustering = k_means_sillhoutte(X2, 10)

    #k_means_adjusted_rand_score(X2, y2, 7)

    #X2_k2_best_df = select_Kbest(X2_df, Y2_under, 2)
    #X2_k2_best: np.ndarray = X2_k2_best_df.values

    #pca_graph(X2, y2_pred_clustering)




" Run for cluster "
#unsupervised_1st(data)
#unsupervised_2nd()
