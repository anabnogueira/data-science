#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import functions as func


register_matplotlib_converters()
data = pd.read_csv('data/pd.csv',  index_col = 'id', header = 1)

# print(data)

# print(data.shape)

# print(data.dtypes)

# print(data.dtypes.value_counts())

new_data = data.iloc[:,1:22]

# print(new_data)



#print(data)

#print(new_data.shape)

#print(new_data.dtypes)



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
#plt.show()


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
#plt.show()


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



