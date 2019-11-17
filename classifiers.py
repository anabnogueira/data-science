#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 10:12:16 2019

@author: vilmaneves
"""

import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import functions as func
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
from sklearn.ensemble import GradientBoostingClassifier



######################################################################################################################
####   DATA PROCESSISG   #############################################################################################
######################################################################################################################

def heatmap(data):
    fig = plt.figure(figsize=[12, 12])
    corr_mtx = data.corr()
    sns.heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
    plt.title('Correlation analysis')
    plt.show()
    
    
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


def normalization(X):
    transf = Normalizer().fit(X)
    new_X = transf.transform(X)

    return new_X


def show_classBalance(data):
    target_count = data['class'].value_counts()
    plt.figure()
    plt.title('Class balance')
    plt.bar(target_count.index, target_count.values)
    plt.show()
    print("\n")

    min_class = target_count.idxmin()
    ind_min_class = target_count.index.get_loc(min_class)

    print('Minority class:', target_count[ind_min_class])
    print('Majority class:', target_count[1-ind_min_class])
    print('Proportion:', round(target_count[ind_min_class] / target_count[1-ind_min_class], 2), ': 1')
    


'''Balanceamento para haver o memso nr records para classe 1 e 0'''
def show_smote_over_under_sample(un_data): #compara tecnicas de balaceamento
    target_count = unbalace_data['class'].value_counts()
    min_class = target_count.idxmin()
    ind_min_class = target_count.index.get_loc(min_class)
    
    RANDOM_STATE = 42
    values = {'Original': [target_count.values[ind_min_class], target_count.values[1-ind_min_class]]}
    
    df_class_min = un_data[un_data['class'] == min_class]
    df_class_max = un_data[un_data['class'] != min_class]
    
    df_under = df_class_max.sample(len(df_class_min))
    values['UnderSample'] = [target_count.values[ind_min_class], len(df_under)]
    
    df_over = df_class_min.sample(len(df_class_max), replace=True)
    values['OverSample'] = [len(df_over), target_count.values[1-ind_min_class]]
    
    smote = SMOTE(ratio='minority', random_state=RANDOM_STATE)
    y = un_data.pop('class').values
    X = un_data.values
    _, smote_y = smote.fit_sample(X, y)
    smote_target_count = pd.Series(smote_y).value_counts()
    values['SMOTE'] = [smote_target_count.values[ind_min_class], smote_target_count.values[1-ind_min_class]]
    
    plt.figure()
    func.multiple_bar_chart(plt.gca(), 
                            [target_count.index[ind_min_class], target_count.index[1-ind_min_class]], 
                            values, 'Target', 'frequency', 'Class balance')
    plt.show()


'''SMOTE'''
def smote(trnX,trnY):
    print('Dataset shape %s' % Counter(trnY))
    sm = SMOTE()
    trnX_smoted, trnY_smoted = sm.fit_resample(trnX, trnY)
    print('Resampled dataset shape %s' % Counter(trnY_smoted))
    return trnX_smoted, trnY_smoted


def undersample(trnX, trnY):
    print('Dataset shape %s' % Counter(trnY))
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(trnX, trnY)
    print('Resampled dataset shape %s' % Counter(y_resampled))
    return X_resampled, y_resampled
    

def oversample(trnX, trnY):
    print('Dataset shape %s' % Counter(trnY))
    rus = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(trnX, trnY)
    print('Resampled dataset shape %s' % Counter(y_resampled))
    return X_resampled, y_resampled



######################################################################################################################
####   CLASSIFIERS    ###############################################################################################
######################################################################################################################

def gaussianNB(trnX, tstX, trnY, tstY, labels, name):
    clf = GaussianNB()
    clf.fit(trnX, trnY)
    prdY = clf.predict(tstX)
    cnf_mtx = metrics.confusion_matrix(tstY, prdY, labels)

    func.plot_confusion_matrix(cnf_mtx, tstY, prdY, labels, name)

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
    nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 25, 50, 75, 100]
    dist = ['manhattan', 'euclidean', 'chebyshev']
    values = {}
    for d in dist:
        yvalues = []
        for n in nvalues:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            print(d + " distance - " + str(n) + " neighbours")
            scores = cross_val_score(knn, X, y, cv=10)
            print("\tAccuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            yvalues.append(scores.mean())
        values[d] = yvalues
        print("\n")
    plt.figure()
    func.multiple_line_chart(plt.gca(), nvalues, values, 'KNN variations by number of neighbours + CV', 'n', 'accuracy', percentage=True)
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
                scores = cross_val_score(tree, X, y, cv=10)
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
