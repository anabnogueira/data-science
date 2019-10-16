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



'''Feature Selection'''
new_baseline = filter_columns(group_baseline, 0.90)
new_intensity = filter_columns(group_intensity, 0.95)
new_form = filter_columns(group_formant, 0.90)
new_band = filter_columns(group_bandwidth, 0.90)
new_vocalfold = filter_columns(group_vocalfold, 0.90)
new_mfcc = filter_columns(group_mfcc, 0.90)
new_wavelet = filter_columns(group_wavelet, 0.90)



'''Normalization'''
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



frames = [new_baseline,new_intensity,new_form,new_band,new_vocalfold,new_mfcc,new_wavelet, data['class']]
selected_data = pd.concat(frames, axis = 1) #junto todos os grupos

y: np.ndarray = selected_data.pop('class').values #class
X: np.ndarray = selected_data.values
labels = pd.unique(y)

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

trnX_normalized = normalization(trnX)
tstX_normalized = normalization(tstX)

for x in trnX_normalized:
    print(x)
    if x < 0:
        print(x)
        print("negative")


"""
'''SMOTE'''
sm = SMOTE(random_state=42)
trnX_smoted, trnY_smoted = sm.fit_resample(trnX_normalized, trnY)



clf = GaussianNB()
clf.fit(trnX_smoted, trnY_smoted)
prdY = clf.predict(tstX_normalized)
cnf_mtx = metrics.confusion_matrix(tstY, prdY, labels)
print( cnf_mtx)
func.plot_confusion_matrix(cnf_mtx, tstY, prdY, labels)


estimators = {'GaussianNB': GaussianNB(), 
              'MultinomialNB': MultinomialNB(), 
              'BernoulyNB': BernoulliNB()}

xvalues = []
yvalues = []
for clf in estimators:
    print(clf)
    print("xvalues before", xvalues)
    xvalues.append(clf)
    print("xvalues after", xvalues)
    estimators[clf].fit(trnX_smoted, trnY_smoted)
    print(estimators[clf].fit(trnX_smoted, trnY_smoted))
    prdY = estimators[clf].predict(tstX_normalized)
    print("prdY", prdY)
    yvalues.append(metrics.accuracy_score(tstY, prdY))
    print("yvalues accuracy score", yvalues)
    print("\n")

plt.figure()
func.bar_chart(plt.gca(), xvalues, yvalues, 'Comparison of Naive Bayes Models', '', 'accuracy', percentage=True)
plt.show()
"""