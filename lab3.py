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



register_matplotlib_converters()
data = pd.read_csv('data/pd.csv',  index_col = 'id', header = 1)

group_baseline = data.iloc[:,1:22]
group_intensity = data.iloc[:,22:25]
group_formant = data.iloc[:,25:29]
group_bandwidth = data.iloc[:,29:33]
group_vocalfold = data.iloc[:,33:55]
group_mfcc = data.iloc[:,55:139]
group_wavelet = data.iloc[:,139:-1]


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



'''Normalization'''
def minMax_data(data):

    scaler = MinMaxScaler()
    scaler.fit(data)
    new_data = scaler.transform(data)

    return new_data

def normalization(X):

    #transf = Normalizer().fit(X)
    #df_nr = pd.DataFrame(transf.transform(X, copy=True), columns=X.columns)

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
def smote(trnX,trY):
    sm = SMOTE(random_state=42)
    trnX_smoted, trnY_smoted = sm.fit_resample(trnX, trnY)
    return trnX_smoted, trnY_smoted

def undersample(trnX, trnY):
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(trnX, trnY)
    return X_resampled, y_resampled
    
def oversample(trnX, trnY):
    rus = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(trnX, trnY)
    return X_resampled, y_resampled

def gaussianNB(trnX, tstX, trnY, tstY, labels):
    clf = GaussianNB()
    clf.fit(trnX, trnY)
    prdY = clf.predict(tstX)
    cnf_mtx = metrics.confusion_matrix(tstY, prdY, labels)

    func.plot_confusion_matrix(cnf_mtx, tstY, prdY, labels)

    return cnf_mtx

def compareNB(trnX,tstX, trnY,tstY):
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
    func.bar_chart(plt.gca(), xvalues, yvalues, 'Comparison of Naive Bayes Models', '', 'accuracy', percentage=True)
    plt.show()

def knn(trnX,trnY,tstX,tstY):
    nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    dist = ['manhattan', 'euclidean', 'chebyshev']
    values = {}
    for d in dist:
        yvalues = []
        for n in nvalues:
            knn = KNeighborsClassifier(n_neighbors=n, metric=d)
            knn.fit(trnX, trnY)
            prdY = knn.predict(tstX)
            yvalues.append(metrics.accuracy_score(tstY, prdY))
        values[d] = yvalues

    plt.figure()
    func.multiple_line_chart(plt.gca(), nvalues, values, 'KNN variants', 'n', 'accuracy', percentage=True)
    plt.show()



""" MAIN """

'''Feature Selection'''
new_baseline = filter_columns(group_baseline, 0.90)
new_intensity = filter_columns(group_intensity, 0.95)
new_form = filter_columns(group_formant, 0.90)
new_band = filter_columns(group_bandwidth, 0.90)
new_vocalfold = filter_columns(group_vocalfold, 0.90)
new_mfcc = filter_columns(group_mfcc, 0.90)
new_wavelet = filter_columns(group_wavelet, 0.90)



frames = [new_baseline,new_intensity,new_form,new_band,new_vocalfold,new_mfcc,new_wavelet, data['class']]
selected_data = pd.concat(frames, axis = 1) #junto todos os grupos


y: np.ndarray = selected_data.pop('class').values #class
X: np.ndarray = selected_data.values
labels = pd.unique(y)

new_X = minMax_data(X)


trnX, tstX, trnY, tstY = train_test_split(new_X, y, train_size=0.7, stratify=y)
""""
print("Gaussian")
clfG = GaussianNB()
clfM = MultinomialNB()
clfB = BernoulliNB()
scores = cross_val_score(clfG, new_X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(clfM, new_X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scores = cross_val_score(clfB, new_X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
"""

x1, y1 = oversample(trnX, trnY)
x2, y2 = undersample(trnX, trnY)


print("Train set, before")
print(trnX.shape)
print("Train set, oversample")
print(x1.shape)
print("Train set, undersample")
print(x2.shape)







trnX_normalized = normalization(trnX)
tstX_normalized = normalization(tstX)

gaussianNB(trnX, tstX, trnY, tstY, labels)
gaussianNB(trnX_normalized, tstX_normalized, trnY, tstY, labels)
'''
compareNB(trnX,trnY,tstY)
compareNB(trnX_normalized,trnY,tstY)

'''
knn(trnX_normalized,trnY,tstX,tstY)


"""
TO DO:
-OverSample
-undersample
-cross variar cv 

"""