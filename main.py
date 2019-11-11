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
from sklearn.metrics import silhouette_score, adjusted_rand_score
import time, warnings, numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs


""" MAIN """
register_matplotlib_converters()
data = pd.read_csv('data/pd.csv',  index_col = 'id', header = 1)

selected_data = feature_selection(data, 0.8)



def sep_data(data):
    """divide in x and y (removing class
    returns X, y, and the colunms
    """

    y: np.ndarray = data.pop('class').values #class
    X: np.ndarray = data.values

    X_columns = data.columns
    labels = pd.unique(y)

    return y,X, X_columns


"""Data preparation with func feature_selection(0.8)"""

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


""" Data preperation for ASSOCIATION RULES  """
y, X, X_columns = sep_data(data)

X_collumns_name = X_columns.tolist()

X_df = pd.DataFrame(X, columns=X_collumns_name)


"select kBest and save columns names"

def select_Kbest(X, k):
    X_df = pd.DataFrame(X, columns=X_collumns_name)
    selector = SelectKBest(f_classif, k=k)
    X_new = selector.fit_transform(X_df, y)
    names = X_df.columns.values[selector.get_support()]
    scores = selector.scores_[selector.get_support()]
    X_KBest_df = pd.DataFrame(X_new, columns=names)

    return X_KBest_df

""""def cut_qcut_compare():
    X_k10_best_cut = X_k10_best_df.copy()
    X_k10_best_qcut = X_k10_best_df.copy()
    bins = 2
    labels = ['0', '1']
    max_bins = 10

    for i in range(2, max_bins):
        for col in X_k10_best_cut:
            X_k10_best_cut[col] = pd.cut(X_k10_best_cut[col], bins, labels=labels)
            X_k10_best_qcut[col] = pd.cut(X_k10_best_qcut[col], bins, labels=labels)
            bins += 1
            labels
"""

#export1 = X_KBest_df.to_csv(r'x_Kbest.csv', index=None, header=True)

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


"DYMMY"
def dummyfication(X_df):
    dummylist = []
    for att in X_df:
        dummylist.append(pd.get_dummies(X_df[[att]]))
    dummified_df = pd.concat(dummylist, axis=1)
    #export2 = dummified_df.to_csv(r'dummified_df.csv', index=None, header=True)
    
    return dummified_df



"ASSOCIATION RULES USING APRIORI"
def assRules_w_apriori(dummified_df):
    #min_support_list = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    #mean_supp_list = []
    #mean_lift_list = []
    sup = 0.35
    #for minsup in minsup_list:
    frequent_itemsets = apriori(dummified_df, min_support=sup, use_colnames=True)

    minconf = 0.9
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=minconf)
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    assoc_rules = rules[(rules['antecedent_len']>=2)] #[0:10]
    print(assoc_rules)
    mean_support = assoc_rules["support"].mean()
    #mean_supp_list.append(mean_support)

    mean_lift = assoc_rules["lift"].mean()
    #mean_lift_list.append(mean_lift)

    return mean_support, mean_lift


def cut_qcut_avg_support(X_df):
    label = ['0','1','2']


    X_df_cut = cut(X_df, 3, label)
    dummy_df_cut = dummyfication(X_df_cut)
    mean_support_cut, mean_lift_cut = assRules_w_apriori(dummy_df_cut)
    print(mean_support_cut)
    print(mean_lift_cut)


    X_df_qcut = qcut(X_df, 3, label)
    dummy_df_qcut = dummyfication(X_df_qcut)
	#mean_support_cut, mean_lift_cut = assRules_w_apriori(dummy_df_cut)
	#mean_support_qcut, mean_lift_qcut = assRules_w_apriori(dummy_df_qcut)
    #print(mean_support_qcut)
    #print(mean_lift_qcut)


X_k10_best_df = select_Kbest(X, 10)
cut_qcut_avg_support(X_k10_best_df)




#dummified = dummyfication(X_copy)
#assRules_w_apriori(dummified)



"CLUSTERING"
"""
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
    plt.ylabel("Number of Inertia")

    plt.plot(nr_clusters_list, list_inertia_values)
    #plt.show()
    
    
def k_means_sillhoutte(X, nr_cluster):
    kmeans_model = cluster.KMeans(n_clusters=nr_cluster, random_state=1).fit(X)
    y_pred = kmeans_model.labels_
    print("Silhouette:", metrics.silhouette_score(X, y_pred))

def k_means_adjusted_rand_score(y_true, nr_cluster):
    kmeans_model = cluster.KMeans(n_clusters=nr_cluster, random_state=1).fit(X) 
    y_pred = kmeans_model.labels_
    print("Adjusted Rand Score =", adjusted_rand_score(y_true, y_pred))


kmeans_NrClusters_inertia(X_normalized)

k_means_sillhoutte(X_normalized,6)

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

    plt.show()


clusters_plot(X_k2_best)"""