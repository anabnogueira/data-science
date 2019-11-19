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
    plt.ylabel('Mean scores')
    plt.title('Mean scores for number of features through Naive Bayes classifier')
    plt.show()


"select kBest and save columns names FOR UNSUPERVISED"
def select_Kbest(X, k):
    "select the K best an returns a data frame"
    X_df = pd.DataFrame(X, columns=X_collumns_name)
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

show_classBalance(data, "Class Balance")

selected_data = feature_selection(data, 0.9)

y, X, X_columns = sep_data(selected_data)

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.85, stratify=y)

# Dta scaling
trnX = minMax_data(trnX)
tstX = minMax_data(tstX)

print("puro")
NB_crossValidation(trnX,trnY)


X_smoted, Y_smoted = smote(trnX,trnY)
X_over, Y_over = oversample(trnX, trnY)
X_under, Y_under = undersample(trnX, trnY)

print("smote")
NB_crossValidation(X_smoted,Y_smoted)
print("over")
NB_crossValidation(X_over,Y_over)
print("under")
NB_crossValidation(X_under,Y_under)


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

""""
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


"ASSOCIATION RULES USING APRIORI"


######################################################################################################################
####   ASSOCIATION RULES USING APRIORI  ##############################################################################
######################################################################################################################
def cut(X_df, bins, labels):
    X_copy = X_df.copy()
    for col in X_copy:
        X_copy[col] = pd.cut(X_copy[col], bins, labels=labels)
    return X_copy

def qcut(X_df, quantils, label):
    X_copy = X_df.copy()
    for col in X_copy:
        X_copy[col] = pd.qcut(X_copy[col], quantils, labels = label)
    return X_copy


"DUMMY"
def dummyfication(X_df):
    dummylist = []
    for att in X_df:
        dummylist.append(pd.get_dummies(X_df[[att]]))
    dummified_df = pd.concat(dummylist, axis=1)

    return dummified_df


def freqt_assRule_mining(dummified_df):

    frequent_itemsets = {}
    minpaterns = 300
    minsup = 1.0

    while minsup > 0:

        minsup = minsup * 0.9
        frequent_itemsets = apriori(dummified_df, min_support=minsup, use_colnames=True)
        if len(frequent_itemsets) >= minpaterns:
            #print("Minimum support:", minsup)
            break
    #print("Number of found patterns:", len(frequent_itemsets))

    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    frqt = frequent_itemsets[(frequent_itemsets['length'] >= 3)]

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.9)
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    #print(rules)
    assoc_rules = rules[(rules['antecedent_len'] >= 2)]
    #export = assoc_rules.to_csv(r'assoc_rules.csv', index=None, header=True)
    #print(assoc_rules)
    mean_support = assoc_rules["support"].mean()
    mean_lift = assoc_rules["lift"].mean()

    return mean_support, mean_lift



def support_cut_qcut_compare(X_df):
    bins = [2,3,4,5,6,7,8,9,10]
    labels = ['0','1']
    label = 1
    avg_supports_cut = []
    avg_supports_qcut = []

    for bin in bins:
        X_df_cut = cut(X_df, bin, labels)
        dummy_df_cut = dummyfication(X_df_cut)
        mean_support_cut, mean_lift_cut = freqt_assRule_mining(dummy_df_cut)
        avg_supports_cut.append(mean_support_cut)

        X_df_qcut = qcut(X_df, bin, labels)
        dummy_df_qcut = dummyfication(X_df_qcut)
        mean_support_qcut, mean_lift_qcut = freqt_assRule_mining(dummy_df_qcut)
        avg_supports_qcut.append(mean_support_qcut)

        bin += 1
        label += 1
        labels.append(str(label))


    plt.plot(bins, avg_supports_cut, color='g')
    plt.plot(bins, avg_supports_qcut, color='orange')
    plt.xlabel('Number of bins/ quantiles')
    plt.ylabel('Average support')
    plt.title('Average support values through cut and qcut')
    plt.show()


def lift_cut_qcut_compare(X_df):
    bins = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    labels = ['0', '1']
    label = 1
    avg_lift_cut = []
    avg_lift_qcut = []

    for bin in bins:
        X_df_cut = cut(X_df, bin, labels)
        dummy_df_cut = dummyfication(X_df_cut)
        mean_support_cut, mean_lift_cut = freqt_assRule_mining(dummy_df_cut)
        avg_lift_cut.append(mean_lift_cut)

        X_df_qcut = qcut(X_df, bin, labels)
        dummy_df_qcut = dummyfication(X_df_qcut)
        mean_support_qcut, mean_lift_qcut = freqt_assRule_mining(dummy_df_qcut)
        avg_lift_qcut.append(mean_lift_qcut)

        bin += 1
        label += 1
        labels.append(str(label))

    plt.plot(bins, avg_lift_cut, color='g')
    plt.plot(bins, avg_lift_qcut, color='orange')
    plt.xlabel('Number of bins/ quantiles')
    plt.ylabel('Average lift')
    plt.title('Average lift values through cut and qcut')
    #plt.show()



######################################################################################################################
####   ASSOCIATION RULES  #############################################################################################
######################################################################################################################
#Data preperation for ASSOCIATION RULES  """
""""
y, X, X_columns = sep_data(selected_data)

X_collumns_name = X_columns.tolist()

#X_df = pd.DataFrame(X, columns=X_collumns_name)


#X_df = select_Kbest(X_df, best_nr_features)

#best_number_features_NB(X,y)
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
####   CLUSTERING  #############################################################################################
######################################################################################################################
"""""
X_normalized = normalization(X)
#print(X_normalized)



#K MEANS
def kmeans_NrClusters_inertia(X):
    #finds best nr of clusters with inertia values
    #shows graph
    #returns best nr of clusters
    
    list_inertia_values = []
    nr_clusters_list = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    for nr_cluster in nr_clusters_list:
        kmeans_model = cluster.KMeans(n_clusters=nr_cluster, random_state=1).fit(X)
        #y_pred = kmeans_model.labels_
        list_inertia_values.append(kmeans_model.inertia_)

    #shows graph
    plt.title("K-Means and number of clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")

    plt.plot(nr_clusters_list, list_inertia_values, linewidth=4)
    #plt.show()
    
    
def k_means_sillhoutte(X, nr_cluster):
    kmeans_model = cluster.KMeans(n_clusters=nr_cluster, random_state=1).fit(X)
    y_pred = kmeans_model.labels_
    print("Silhouette:", metrics.silhouette_score(X, y_pred))
    # return y_pred to be used in pca graph
    return y_pred

def k_means_adjusted_rand_score(y_true, nr_cluster):
    kmeans_model = cluster.KMeans(n_clusters=nr_cluster, random_state=1).fit(X) 
    y_pred = kmeans_model.labels_
    print("Adjusted Rand Score =", adjusted_rand_score(y_true, y_pred))


kmeans_NrClusters_inertia(X_normalized)

# return y_pred to be used in pca graph
y_pred_clustering = k_means_sillhoutte(X_normalized,6)

k_means_adjusted_rand_score(y, 6)

X_k2_best_df = select_Kbest(X_normalized,2)
#print(X_k2_best_df)

X_k2_best: np.ndarray = X_k2_best_df.values
#print(X_k2_best)

def clusters_plot(X_k2_best):
    # 1b compute clustering with Means
    k_means = KMeans(init='k-means++', n_clusters=6, n_init=10)
    t0 = time.time()
    k_means.fit(X_k2_best)
    t_batch = time.time() - t0

    # 1c compute clustering with MiniBatchKMeans
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=45, n_init=10, max_no_improvement=10, verbose=0)
    t0 = time.time()
    mbk.fit(X_k2_best)
    t_mini_batch = time.time() - t0


    # 2 plotting differences result
    fig = plt.figure(figsize=(8, 3))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    colors = ['#EB507F', '#137DD2', '#F6E051', '#4EACC5', '#FF9C34', '#4E9A06']

    k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
    mbk_means_cluster_centers = np.sort(mbk.cluster_centers_, axis=0)
    k_means_labels = pairwise_distances_argmin(X_k2_best, k_means_cluster_centers)
    mbk_means_labels = pairwise_distances_argmin(X_k2_best, mbk_means_cluster_centers)
    order = pairwise_distances_argmin(k_means_cluster_centers, mbk_means_cluster_centers)


    # 2a KMeans
    ax = fig.add_subplot(1, 3, 1)
    n_clusters = 6
    for k, col in zip(range(n_clusters), colors):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        ax.plot(X_k2_best[my_members,0], X_k2_best[my_members,1],  'w', markerfacecolor=col, marker='o')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
    ax.set_title('KMeans')
    ax.set_xticks(())
    ax.set_yticks(())
    plt.text(-3.5, 1.8,'train time: %.2fs\ninertia: %f' % (t_batch, k_means.inertia_))


    # 2b MiniBatchKMeans
    ax = fig.add_subplot(1, 3, 2)
    for k, col in zip(range(n_clusters), colors):
        my_members = mbk_means_labels == order[k]
        cluster_center = mbk_means_cluster_centers[order[k]]
        ax.plot(X_k2_best[my_members,0], X_k2_best[my_members,1],'w',markerfacecolor=col,marker='.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
    ax.set_title('MiniBatchKMeans')
    ax.set_xticks(())
    ax.set_yticks(())
    plt.text(-3.5,1.8,'train time: %.2fs\ninertia: %f' % (t_mini_batch, mbk.inertia_))

    # 2c difference between solutions
    different = (mbk_means_labels == 4)
    ax = fig.add_subplot(1, 3, 3)
    for k in range(n_clusters):
        different += ((k_means_labels == k) != (mbk_means_labels == order[k]))

    identic = np.logical_not(different)
    ax.plot(X_k2_best[identic, 0], X_k2_best[identic, 1], 'w',markerfacecolor='#bbbbbb', marker='.')
    ax.plot(X_k2_best[different, 0], X_k2_best[different, 1], 'w', markerfacecolor='m', marker='.')
    ax.set_title('Difference')
    ax.set_xticks(())
    ax.set_yticks(())

    #plt.show()


#clusters_plot(X_k2_best)


# plot best 2 pca components colored with k-means clustering
def pca_graph(X, y_clustered):

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['Principal Component 1', 'Principal Component 2'])

    # turn numpy into dataframe and concat
    y_pred_clustering_df = pd.DataFrame({'target': y_clustered})
    finalDf = pd.concat([principalDf, y_pred_clustering_df], axis = 1)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 12)
    ax.set_ylabel('Principal Component 2', fontsize = 12)
    ax.set_title('K-means clustering with 2 Principal Components', fontsize = 16)

    targets = [0, 1, 2, 3, 4, 5]
    target_labels = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']
    colors = ['#00A0B0', '#6A4A3C', '#CC333F', '#EB6841', '#EDC951', '#252525']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'Principal Component 1'], finalDf.loc[indicesToKeep, 'Principal Component 2'], c = color, s = 75, marker='o')
    ax.legend(target_labels)
    ax.grid()

    #plt.show()

#pca_graph(X_normalized, y_pred_clustering)


# plot best 2 pca components colored with k-means clustering
def pca_graph_kmeans(X):

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['Principal Component 1', 'Principal Component 2'])

    kmeans_model = cluster.KMeans(n_clusters=6, random_state=1).fit(principalDf)
    y_clustered = kmeans_model.labels_

    # turn numpy into dataframe and concat
    y_pred_clustering_df = pd.DataFrame({'target': y_clustered})
    finalDf = pd.concat([principalDf, y_pred_clustering_df], axis = 1)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 12)
    ax.set_ylabel('Principal Component 2', fontsize = 12)
    ax.set_title('K-means clustering with 2 Principal Components', fontsize = 16)

    targets = [0, 1, 2, 3, 4, 5]
    target_labels = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5']
    colors = ['#00A0B0', '#6A4A3C', '#CC333F', '#EB6841', '#EDC951', '#252525']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'Principal Component 1'], finalDf.loc[indicesToKeep, 'Principal Component 2'], c = color, s = 75, marker='o')
    ax.legend(target_labels)
    ax.grid()

    #plt.show()


#pca_graph_kmeans(X_normalized)

"""


"""def xgboosting(X):
    # Dta scaling
    X = minMax_data(X)
    #y = sep_data(selected_data)[0]
    # Data split
    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

    # Normalization
    X_normalized = normalization(X)

    trnX_normalized = normalization(trnX)
    tstX_normalized = normalization(tstX)

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

#xgboosting(X)
"""

""""
def pca_variance(X):
    pca_components = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10"]

    pca = PCA(n_components=10)
    principalComponents = pca.fit_transform(X)
    variance_ratio = pca.explained_variance_ratio_

    plt.figure()
    plt.title('Principal Components')
    plt.ylabel("Variance")
    plt.bar(pca_components, variance_ratio, color="#4287f5")
    #plt.show()


#pca_variance(X_normalized)
"""

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

show_classBalance(datasetTwo, "Class Balance 2nd dataset")

#heatmap(datasetTwo)

y2, X2, X2_columns = sep_data(datasetTwo)
#print(X2.shape)

X2_df = pd.DataFrame(X2, columns=X2_columns)

filtered_X2 = filter_columns(X2_df, 0.6)
#print(filtered_X2.shape)


# Data split
"""trnX, tstX, trnY, tstY = train_test_split(X_2, y_2, train_size=0.7, stratify=y_2)

#X_smoted, Y_smoted = smote(trnX,trnY)
#X_over, Y_over = oversample(trnX, trnY)
X_under, Y_under = undersample(trnX, trnY)
#X_under2, Y_under2 = under_ClusterCentroids(trnX, trnY)

#clf = GaussianNB()
#scores1 = cross_val_score(clf, X_under, Y_under, cv=5)

best_number_features_NB(X_under,Y_under)
"""

"""def stratifiedShuffleSplit(X,y):
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













