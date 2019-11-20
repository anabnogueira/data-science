#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 10:12:16 2019

@author: vilmaneves
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import functions as func
import seaborn as sns
import scipy.stats as _stats
import numpy as np
import time
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.metrics import accuracy_score, pairwise_distances_argmin
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from imblearn.under_sampling import RandomUnderSampler, InstanceHardnessThreshold, ClusterCentroids, EditedNearestNeighbours, AllKNN
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from subprocess import call
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn import datasets, metrics, cluster, mixture
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, accuracy_score



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



######################################################################################################################
####   DATA EXPLORATION   #############################################################################################
######################################################################################################################

def heatmap(data):
    fig = plt.figure(figsize=[12, 12])
    corr_mtx = data.corr()
    sns.heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
    plt.title('Correlation analysis')
    plt.show()


def show_classBalance(data, title):
    target_count = data['class'].value_counts()
    plt.figure()
    plt.title(title)
    classes, count = zip(*sorted(zip(target_count.index, target_count.values)))

    ls = []
    for el in classes:
        ls = ls + [str(el)]

    plt.bar(ls, count, color="#4287f5")
    #plt.show()
    print("\n")

    """min_class = target_count.idxmin()
    ind_min_class = target_count.index.get_loc(min_class)
    print('Minority class:', target_count[ind_min_class])
    print('Majority class:', target_count[1-ind_min_class])
    print('Proportion:', round(target_count[ind_min_class] / target_count[1-ind_min_class], 2), ': 1')"""


'''Balanceamento para haver o memso nr records para classe 1 e 0'''


def show_smote_over_under_sample(un_data):  # compara tecnicas de balaceamento
    target_count = un_data['class'].value_counts()
    min_class = target_count.idxmin()
    ind_min_class = target_count.index.get_loc(min_class)

    RANDOM_STATE = 42
    values = {'Original': [target_count.values[ind_min_class], target_count.values[1 - ind_min_class]]}

    df_class_min = un_data[un_data['class'] == min_class]
    df_class_max = un_data[un_data['class'] != min_class]

    df_under = df_class_max.sample(len(df_class_min))
    values['UnderSample'] = [target_count.values[ind_min_class], len(df_under)]

    df_over = df_class_min.sample(len(df_class_max), replace=True)
    values['OverSample'] = [len(df_over), target_count.values[1 - ind_min_class]]

    smote = SMOTE(ratio='minority', random_state=RANDOM_STATE)
    y = un_data.pop('class').values
    X = un_data.values
    _, smote_y = smote.fit_sample(X, y)
    smote_target_count = pd.Series(smote_y).value_counts()
    values['SMOTE'] = [smote_target_count.values[ind_min_class], smote_target_count.values[1 - ind_min_class]]

    plt.figure()
    func.multiple_bar_chart(plt.gca(),
                            [target_count.index[ind_min_class], target_count.index[1 - ind_min_class]],
                            values, 'Target', 'frequency', 'Class balance')
    plt.show()


######################################################################################################################
####   DIMENSIONALITY   #############################################################################################
######################################################################################################################



def filter_columns(group, coef):
    corr_mtx = group.corr()
    
    #print("Pairs of parameters with Pearson Coefficient larger than " + str(coef))
    #print(60*"-")
    columns = np.full((corr_mtx.shape[0],), True, dtype=bool)
    for i in range(corr_mtx.shape[0]):
        for j in range(i+1, corr_mtx.shape[0]):
            if coef > 0 and (corr_mtx.iloc[i,j] >= coef):
                #print(group.columns[i] + " - " + group.columns[j])
                if columns[j]:
                    columns[j] = False

    selected_columns = group.columns[columns]
    result = group[selected_columns]
    return result


def feature_selection(data, thresh):
    group_baseline = data.iloc[:,1:22]
    group_intensity = data.iloc[:,22:25]
    group_formant = data.iloc[:,25:29]
    group_bandwidth = data.iloc[:,29:33]
    group_vocalfold = data.iloc[:,33:55]
    group_mfcc = data.iloc[:,55:139]
    group_wavelet = data.iloc[:,139:-1]

    new_baseline = filter_columns(group_baseline, thresh)
    new_intensity = filter_columns(group_intensity, thresh)
    new_form = filter_columns(group_formant, thresh)
    new_band = filter_columns(group_bandwidth, thresh)
    new_vocalfold = filter_columns(group_vocalfold, thresh)
    new_mfcc = filter_columns(group_mfcc, thresh)
    new_wavelet = filter_columns(group_wavelet, thresh)

    frames = [new_baseline,new_intensity,new_form,new_band,new_vocalfold,new_mfcc,new_wavelet, data['class']]
    selected_data = pd.concat(frames, axis = 1)

    return selected_data


'''Normalization'''
def minMax_data(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    new_data = scaler.transform(data)

    return new_data


'''Normalization'''
def normalization(X):
    transfX = Normalizer().fit(X)
    new_X = transfX.transform(X)

    return new_X


######################################################################################################################
####   BALANCING   #############################################################################################
######################################################################################################################


'''SMOTE'''
def smote(trnX,trnY):
    #print('Dataset shape %s' % Counter(trnY))
    sm = SMOTE()
    trnX_smoted, trnY_smoted = sm.fit_resample(trnX, trnY)
    #print('Resampled dataset shape %s' % Counter(trnY_smoted))
    return trnX_smoted, trnY_smoted


def undersample(trnX, trnY):
    #print('Dataset shape %s' % Counter(trnY))
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(trnX, trnY)
    #print('Resampled dataset shape %s' % Counter(y_resampled))
    return X_resampled, y_resampled

def undersample_AllNN(trnX, trnY):
    rus = AllKNN(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(trnX, trnY)
    return X_resampled, y_resampled
    

def oversample(trnX, trnY):
    #print('Dataset shape %s' % Counter(trnY))
    rus = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(trnX, trnY)
    #print('Resampled dataset shape %s' % Counter(y_resampled))
    return X_resampled, y_resampled



######################################################################################################################
####   CLASSIFIERS    ###############################################################################################
######################################################################################################################


" ************ NAive Bayes *************"

def NB_crossValidation(X,y):
    clfG = GaussianNB()
    clfM = MultinomialNB()
    clfB = BernoulliNB()
    print("Naive Bayes")
    print("\tGaussian")
    scores = cross_val_score(clfG, X, y, cv=10)
    print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("\tMultinomial")
    scores = cross_val_score(clfM, X, y, cv=10)
    print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("\tBernoulli")
    scores = cross_val_score(clfB, X, y, cv=10)
    print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



def gaussianNB(trnX, tstX, trnY, tstY, labels):
    clf = GaussianNB()
    clf.fit(trnX, trnY)
    prdY = clf.predict(tstX)
    cnf_mtx = metrics.confusion_matrix(tstY, prdY, labels)

    func.plot_confusion_matrix(cnf_mtx, tstY, prdY, labels, "Confusion Matrix with Naive Bayes")
    plt.show()
    return cnf_mtx


def compareNB(trnX, tstX, trnY, tstY, title):
    estimators = {'GaussianNB': GaussianNB(),
                  'MultinomialNB': MultinomialNB(),
                  'BernoulyNB': BernoulliNB()}

    xvalues = []
    yvalues = []
    for clf in estimators:
        xvalues.append(clf)
        estimators[clf].fit(trnX, trnY)
        prdY = estimators[clf].predict(tstX)
        yvalues.append(metrics.accuracy_score(tstY, prdY))

    plt.figure()
    func.bar_chart(plt.gca(), xvalues, yvalues, title, '', 'accuracy', percentage=True)
    plt.show()


" ************ KNN *************"


def knn(trnX,trnY,tstX,tstY):
    nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 25, 50, 75, 100]
    dist = ['manhattan', 'euclidean', 'chebyshev']
    values = {}
    for d in dist:
        yvalues = []
        for n in nvalues:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            knn.fit(trnX, trnY)
            prdY = knn.predict(tstX)
            #print(metrics.accuracy_score(tstY, prdY))
            yvalues.append(metrics.accuracy_score(tstY, prdY))
        values[d] = yvalues

    plt.figure()
    func.multiple_line_chart(plt.gca(), nvalues, values, 'KNN variations by number of neighbours', 'n', 'accuracy', percentage=True)
    plt.show()


def knn_cross_validation(X, y):
    nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 25, 50, 75, 100] # 1st data set
    #nvalues = [1,2,3,4,5,6,10,20,50,10,200,300,400,700, 1000]  # 2nd data set

    dist = ['manhattan', 'euclidean', 'chebyshev']
    values = {}
    for d in dist:
        yvalues = []
        for n in nvalues:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            #print(d + " distance - " + str(n) + " neighbours")
            scores = cross_val_score(knn, X, y, cv=10)
            #print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            yvalues.append(scores.mean())
        values[d] = yvalues
        #print("\n")
    plt.figure()
    func.multiple_line_chart(plt.gca(), nvalues, values, 'KNN variations by number of neighbours', 'n', 'accuracy', percentage=True)
    plt.show()


def knn_feature_selection(data):
    #nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 25, 50, 75, 100]
    nvalues = [1, 3, 5, 15, 25, 50, 200]
    dist = 'manhattan'
    coefs = [0.2, 0.4, 0.5, 0.7, 0.8, 0.9]
    original = data

    values = {}
    for n in nvalues:
        yvalues = []
        for c in coefs:
            data = original
            data = feature_selection(data, c)

            y: np.ndarray = data.pop('class').values
            X: np.ndarray = data.values

            knn = KNeighborsClassifier(n_neighbors=n, metric=dist)
            scores = cross_val_score(knn, X, y, cv=10)

            yvalues.append(scores.mean())
        values[n] = yvalues
        
    plt.figure()
    func.multiple_line_chart(plt.gca(), coefs, values, 'KNN variations by number of neighbours + CV', 'n', 'accuracy', percentage=True)
    plt.show()



" ************ Decision Trees *************"

def decision_trees(trnX, trnY, tstX, tstY):
    min_samples_leaf = [.05, .025, .01, .0075, .005, .0025, .001]
    max_depths = [5, 10, 25, 50]
    criteria = ['entropy', 'gini']

    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(16, 4), squeeze=False)
    for k in range(len(criteria)):
        f = criteria[k]
        values = {}
        for d in max_depths:
            yvalues = []
            for n in min_samples_leaf:
                tree = DecisionTreeClassifier(min_samples_leaf=n, max_depth=d, criterion=f)
                tree.fit(trnX, trnY)
                prdY = tree.predict(tstX)
                yvalues.append(metrics.accuracy_score(tstY, prdY))
            values[d] = yvalues
        func.multiple_line_chart(axs[0, k], min_samples_leaf, values, 'Decision Trees with %s criteria' % f,
                                 'nr estimators',
                                 'accuracy', percentage=True)

    plt.show()


def decision_trees_cross_validation(X, y):
    min_samples_leaf = [.05, .025, .01, .0075, .005, .0025, .001]
    max_depths = [5, 10, 25, 50]
    criteria = ['entropy', 'gini']

    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(16, 4), squeeze=False)
    for k in range(len(criteria)):
        f = criteria[k]
        values = {}
        for d in max_depths:
            yvalues = []
            for n in min_samples_leaf:
                tree = DecisionTreeClassifier(min_samples_leaf=n, max_depth=d, criterion=f)
                scores = cross_val_score(tree, X, y, cv=5)
                yvalues.append(scores.mean())
            values[d] = yvalues
        func.multiple_line_chart(axs[0, k], min_samples_leaf, values, 'Decision Trees with %s criteria' % f,
                                 'nr estimators',
                                 'accuracy', percentage=True)

    plt.show()


def decision_tree_draw(trnX, trnY):
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(trnX, trnY)

    dot_data = export_graphviz(tree, out_file='dtree.dot', filled=True, rounded=True, special_characters=True)  
    # Convert to png
    call(['dot', '-Tpng', 'dtree.dot', '-o', 'dtree.png', '-Gdpi=600'])

    time.sleep(5)
    plt.figure(figsize = (14, 18))
    plt.imshow(plt.imread('dtree.png'))
    plt.axis('off')
    plt.show()


def decision_trees_feature_selection(data, min_samples_leaf, max_depth, metric):
    nr_features = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    recall_values = []

    plt.figure()
    plt.title('Decision Trees and Feature Selection')
    plt.xlabel('Feature Selection')
    plt.ylabel(metric)

    for d in nr_features:
        selected_data = feature_selection(data, d)
        y, X, _ = sep_data(selected_data)
        tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, max_depth=max_depth, criterion='entropy')
        recall = cross_val_score(tree, X, y, cv=5, scoring=metric)
        #print(nr_features)
        #print(recall.mean())

        recall_values.append(recall.mean())

    plt.plot(nr_features, recall_values, color="#4287f5")
    plt.show()





" ************ Random Forest *************"

def random_forests(trnX, trnY, tstX, tstY):
    n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300]
    max_depths = [5, 10, 25, 50]
    max_features = ['sqrt', 'log2']

    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    for k in range(len(max_features)):
        f = max_features[k]
        values = {}
        for d in max_depths:
            yvalues = []
            for n in n_estimators:
                rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
                rf.fit(trnX, trnY)
                prdY = rf.predict(tstX)
                yvalues.append(metrics.accuracy_score(tstY, prdY))
            values[d] = yvalues
        func.multiple_line_chart(axs[0, k], n_estimators, values, 'Random Forests with %s features'%f, 'nr estimators', 
                                 'accuracy', percentage=True)
        
    plt.show()


def random_forests_cross_validation(X, y):
    n_estimators = [5, 10, 25, 50, 75]
    max_depths = [5, 10, 25, 50, 75]
    max_features = ['sqrt', 'log2']

    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    for k in range(len(max_features)):
        f = max_features[k]
        values = {}
        for d in max_depths:
            yvalues = []
            for n in n_estimators:
                rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
                scores = cross_val_score(rf, X, y, cv=10)
                yvalues.append(scores.mean())
            values[d] = yvalues

        func.multiple_line_chart(axs[0, k], n_estimators, values, 'Random Forests with %s features' % f,
                                 'nr estimators',
                                 'accuracy', percentage=True)
    plt.show()




" ************ Gradient Boosting *************"

def gradient_boosting(trnX, trnY, tstX, tstY):

    lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
    n_estimators = [5, 10, 25, 50, 75]
    max_features = ['sqrt', 'log2']
    #max_depths = [2,5,7,10,12,14,17,19,20,23,25,28]

    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    
    for k in range(len(max_features)):
        f = max_features[k] 
        values = {}      
        for estimators in n_estimators:
            yvalues = []
            for learning_rate in lr_list:
                    
                gb_clf = GradientBoostingClassifier(n_estimators=estimators, learning_rate=learning_rate, max_features=f, max_depth=25, random_state=0)
                gb_clf.fit(trnX, trnY)
                prdY = gb_clf.predict(tstX)
                yvalues.append(metrics.accuracy_score(tstY, prdY))

            values[estimators] = yvalues
              
        func.multiple_line_chart(axs[0, k], lr_list, values, 'Gradient Boosting with %s features' % f,
                                 'learning_rate',
                                 'accuracy', percentage=True)    
    
    plt.show()


def gradient_boosting_cross_validation(X,y):

    lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
    n_estimators = [5, 10, 25, 50, 75]
    max_features = ['sqrt', 'log2']
    #max_depths = [2,5,7,10,12,14,17,19,20,23,25,28]

    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    for k in range(len(max_features)):
        f = max_features[k] 
        values = {}      
        for estimators in n_estimators:
            yvalues = []
            for learning_rate in lr_list:
                gb_clf = GradientBoostingClassifier(n_estimators=estimators, learning_rate=learning_rate, max_features=f, max_depth=25, random_state=0)
                scores = cross_val_score(gb_clf, X, y, cv=10)
                yvalues.append(scores.mean())

            values[estimators] = yvalues

        func.multiple_line_chart(axs[0, k], lr_list, values, 'Gradient Boosting CV with %s features' % f,
                                 'learning_rate',
                                 'accuracy', percentage=True)
    plt.show()


def xgboost(trnX, trnY, tstX, tstY):
    # fit model no training data
    model = XGBClassifier()
    model.fit(trnX, trnY)
    #print(model)


    # make predictions for test data
    y_pred = model.predict(tstX)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(tstY, predictions)

    print("Accuracy: %.2f%%" % (accuracy * 100.0))





""""
def xgboosting(X):
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
####   CLUSTERING  #############################################################################################
######################################################################################################################


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

def k_means_adjusted_rand_score(X, y_true, nr_cluster):
    kmeans_model = cluster.KMeans(n_clusters=nr_cluster, random_state=1).fit(X) 
    y_pred = kmeans_model.labels_
    print("Adjusted Rand Score =", adjusted_rand_score(y_true, y_pred))




def clusters_plot(X_k2_best):
    # 1b compute clustering with Means
    k_means = KMeans(init='k-means++', n_clusters=6, n_init=10)
    t0 = time.time()
    k_means.fit(X_k2_best)
    t_batch = time.time() - t0

    # 1c compute clustering with MiniBatchKMeans
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=6, batch_size=45, n_init=10, max_no_improvement=10, verbose=0)
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
        ax.plot(X_k2_best[my_members,0], X_k2_best[my_members,1],  'w', markerfacecolor=col, marker='.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=4)
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
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=4)
    ax.set_title('MiniBatchKMeans')
    ax.set_xticks(())
    ax.set_yticks(())
    plt.text(-3.5,1.8,'train time: %.2fs\ninertia: %f' % (t_mini_batch, mbk.inertia_))

    # 2c difference between solutions
    """
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
    """
    plt.show()



""""
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