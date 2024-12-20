#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import functions as func
import seaborn as sns
import scipy.stats as _stats
import numpy as np
 
register_matplotlib_converters()
data = pd.read_csv('data/pd.csv',  index_col = 'id', header = 1)

# print(data)

shape = data.shape

print("Number of Variables/Attributes: " + str(shape[1]))
print("Number of Records/Instances: " + str(shape[0]))

#print(data.dtypes)
#print(3 * "\n")
#print(data.dtypes.value_counts())
#new_data = data.iloc[:,1:22]

# print(new_data)
#print(new_data.shape)
#print(new_data.dtypes)

print(3 * "\n")

"""
# Boxplot
columns = new_data.select_dtypes(include='number').columns
rows, cols = func.choose_grid(len(columns))
plt.figure()
fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4), squeeze=False)
i, j = 0, 0
for n in range(len(columns)):
    axs[i, j].set_title('Boxplot for %s'%columns[n])
    axs[i, j].boxplot(new_data[columns[n]].dropna().values)
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
fig.tight_layout()
plt.show()

# Histogram
columns = new_data.select_dtypes(include='number').columns
rows, cols = func.choose_grid(len(columns))
plt.figure()
fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4), squeeze=False)
i, j = 0, 0
for n in range(len(columns)):
    axs[i, j].set_title('Histogram for %s'%columns[n])
    axs[i, j].set_xlabel(columns[n])
    axs[i, j].set_ylabel("probability")
    axs[i, j].hist(new_data[columns[n]].dropna().values, 'auto')
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
fig.tight_layout()
plt.show()

# Histogram (granularity)
columns = new_data.select_dtypes(include='number').columns
rows = len(columns)
cols = 5
plt.figure()
fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4), squeeze=False)
bins = range(5, 100, 20)
for i in range(len(columns)):
    for j in range(len(bins)):
        axs[i, j].set_title('Histogram for %s'%columns[i])
        axs[i, j].set_xlabel(columns[i])
        axs[i, j].set_ylabel("probability")
        axs[i, j].hist(data[columns[i]].dropna().values, bins[j])
fig.tight_layout()
plt.show()


# Distribution fits
import scipy.stats as _stats 
def compute_known_distributions(x_values, n_bins) -> dict:
    distributions = dict()
    # Gaussian
    mean, sigma = _stats.norm.fit(x_values)
    distributions['Normal(%.1f,%.2f)'%(mean,sigma)] = _stats.norm.pdf(x_values, mean, sigma)
    # LogNorm
    sigma, loc, scale = _stats.lognorm.fit(x_values)
    distributions['LogNor(%.1f,%.2f)'%(np.log(scale),sigma)] = _stats.lognorm.pdf(x_values, sigma, loc, scale)
    # Exponential
    loc, scale = _stats.expon.fit(x_values)
    distributions['Exp(%.2f)'%(1/scale)] = _stats.expon.pdf(x_values, loc, scale)
    # SkewNorm
   # a, loc, scale = _stats.skewnorm.fit(x_values)
   # distributions['SkewNorm(%.2f)'%a] = _stats.skewnorm.pdf(x_values, a, loc, scale) 
    return distributions

def histogram_with_distributions(ax: plt.Axes, series: pd.Series, var: str):
    values = series.sort_values().values
    n, bins, patches = ax.hist(values, 20, density=True, edgecolor='grey')
    distributions = compute_known_distributions(values, bins)
    func.multiple_line_chart(ax, values, distributions, 'Best fit for %s'%var, var, 'probability')

columns = new_data.select_dtypes(include='number').columns
rows, cols = func.choose_grid(len(columns))
plt.figure()
fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4), squeeze=False)
i, j = 0, 0
for n in range(len(columns)):
    histogram_with_distributions(axs[i, j], new_data[columns[n]].dropna(), columns[n])
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
fig.tight_layout()
plt.show()



columns = new_data.select_dtypes(include='number').columns
rows, cols = func.choose_grid(len(columns))
plt.figure()
fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4), squeeze=False)
i, j = 0, 0
for n in range(len(columns)):
    axs[i, j].set_title('Histogram with trend for %s'%columns[n])
    axs[i, j].set_ylabel("probability")
    sns.distplot(data[columns[n]].dropna().values, norm_hist=True, ax=axs[i, j], axlabel=columns[n])
    i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
fig.tight_layout()
plt.show()


"""


group_baseline = data.iloc[:,1:22]
group_intensity = data.iloc[:,22:25]
group_formant = data.iloc[:,25:29]
group_bandwidth = data.iloc[:,29:33]
group_vocalfold = data.iloc[:,33:55]
group_mfcc = data.iloc[:,55:139]
group_wavelet = data.iloc[:,139:-1]

#new_data = {group_baseline: "Baseline", group_intensity : "group_intensity" }
#print(new_data.get(group_baseline))
#print(data[0])

#### Sparcity ###############################

def sparcity(data):
    columns = data.select_dtypes(include='number').columns
    rows, cols = len(columns)-1, len(columns)-1
    plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4), squeeze=False)
    for i in range(len(columns)):
        var1 = columns[i]
        for j in range(i+1, len(columns)):
            var2 = columns[j]
            axs[i, j-1].set_title("%s x %s"%(var1,var2))
            axs[i, j-1].set_xlabel(var1)
            axs[i, j-1].set_ylabel(var2)
            axs[i, j-1].scatter(data[var1], data[var2])
    fig.tight_layout()
    plt.show()


#### HeatMap ###############################
def heatmap(data):
    fig = plt.figure(figsize=[12, 12])
    corr_mtx = data.corr()
    sns.heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
    plt.title('Correlation analysis')
    plt.show()



all_data = [group_baseline, group_intensity, group_formant, group_bandwidth, group_vocalfold, group_mfcc, group_wavelet]
#all_data_named = ["Baseline", "Intensity", "Format", "Bandwith", "VocalFold", "MFCC", "Wavelet"]
#print(all_data)


    
#group_mfcc_sm = data.iloc[:,56:8]

#sparcity(data.iloc[:,5:10])


def filter_columns(group, coef):
    corr_mtx = group.corr()
    
    #print("Pairs of parameters with Pearson Coefficient larger than " + str(coef))
    print(60*"-")
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

new_base = filter_columns(group_baseline, 0.90)
print("Baseline")
heatmap(new_base)  

#tao pequeno vale a pena?
new_int = filter_columns(group_intensity, 0.90)
#print("Intensity")
#heatmap(new_int) 

#igual
new_form = filter_columns(group_formant, 0.90)
#print("Formant")
#heatmap(new_form) 

#igual
new_band = filter_columns(group_bandwidth, 0.90)
#print("Bandwith")
#heatmap(new_band) 

new_vocal = filter_columns(group_vocalfold, 0.90)
print("VocalFolde")
heatmap(new_vocal) 

new_mfcc = filter_columns(group_mfcc, 0.70)
print("MFCC")
heatmap(new_mfcc) 

new_wav = filter_columns(group_wavelet, 0.50)
print("Wavelet")
heatmap(new_wav) 









#for group in range(len(all_data)-2):
#    print(all_data)
#    new_data = filter_columns(group)
#    heatmap(new_data)




#for group in range(len(all_data)-2):
#    new_data = filter_columns(group, 0.90)
#    heatmap(new_data)  



# group baseline
#heatmap(group_baseline)
#filter_columns(group_baseline, 0.80)
#sparcity(group_baseline)

# group intensity
#heatmap(group_intensity)
#filter_columns(group_intensity, 0.80)
#sparcity(group_intensity)


# group formant
#heatmap(group_formant)
#filter_columns(group_formant, 0.80)
#sparcity(group_formant)

# group bandwidth
#heatmap(group_bandwidth)
#filter_columns(group_bandwidth, 0.80)
#sparcity(group_bandwidth)

# group vocalfold
#heatmap(group_vocalfold)
#filter_columns(group_vocalfold, 0.80)
#sparcity(group_vocalfold)

# group mfcc
#heatmap(group_mfcc)
#filter_columns(group_mfcc, 0.80)
#sparcity(group_mfcc)

# group wavelet
#heatmap(group_wavelet)
#filter_columns(group_wavelet, 0.80)
#sparcity(group_wavelet)

          
            


