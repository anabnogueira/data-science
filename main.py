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

#export1 = X_KBest_df.to_csv(r'x_Kbest.csv', index=None, header=True)

X_k10_best_df = select_Kbest(X, 10)


X_KBest_copy = X_k10_best_df.copy()
for col in X_KBest_copy:
    X_KBest_copy[col] = pd.cut(X_KBest_copy[col], 3, labels=['0','1','2'])

#export2 = X_KBest_copy.to_csv(r'x_Kbest_copy.csv', index=None, header=True)

"DYMMY"
dummylist = []
for att in X_KBest_copy:
    dummylist.append(pd.get_dummies(X_KBest_copy[[att]]))
dummified_df = pd.concat(dummylist, axis=1)
#print(dummified_df.head(5))
export2 = dummified_df.to_csv(r'dummified_df.csv', index=None, header=True)


"ASSOCIATION RULES USING APRIORI"
minsup_list = [0.35, 0.65]
sup = 0.35
for minsup in minsup_list:
    frequent_itemsets = apriori(dummified_df, min_support=sup, use_colnames=True)

    minconf = 0.9
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=minconf)
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    ex = rules[(rules['antecedent_len']>=2)][0:10]

#export = ex.to_csv(r'out.csv', index=None, header=True)


"CLUSTERING"

X_normalized = normalization(X)
#print(X_normalized)

#K MEANS
def kmeans_NrClusters_inertia(X):
    """finds best nr of clusters with inertia values
    shows graph
    returns best nr of clusters
    """
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
    plt.show()
    
    
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


k2_best = select_Kbest(X_normalized,2)
print(k2_best)



#print("Calinski Harabaz:",metrics.calinski_harabasz_score(X, y_pred))
#print("Davies Bouldin:",metrics.davies_bouldin_score(X, y_pred))
#print("Silhouette per instance:",metrics.silhouette_samples(X, y_pred))


