import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import arff
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn import neighbors


np.random.seed(0)

data = arff.loadarff('housing.arff')
df = pd.DataFrame(data[0])

#print(df)
df.pop('CHAS')
print(df)

# draw scatter

plt.figure()
axs = plt.gca()
plt.title("Scatter plot for DIS / class")
df.plot(kind='scatter', x='DIS', y='class', ax=axs)
plt.show()



y: np.ndarray = df.pop('class').values
X: np.ndarray = df.values

# Linear Regression
lm = LinearRegression()
lm.fit(X, y)

print("\n")
scores_lm = cross_val_score(lm, X, y, cv=10, scoring='neg_mean_absolute_error')
score_lm = sum(scores_lm)/len(scores_lm)
print("Linear Regression")
print("\tAveraged Mean Absolute Error:\t", score_lm, "\n")


# Decision Trees
dt = DecisionTreeRegressor()
dt = dt.fit(X, y)

scores_dt = cross_val_score(dt, X, y, cv=10, scoring='neg_mean_absolute_error')
score_dt = sum(scores_dt)/len(scores_dt)
print("Decision Trees")
print("\tAveraged Mean Absolute Error:\t", score_dt, "\n")


# KNN

n_neighbors = 5
knn_uniform = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')
knn_uniform = knn_uniform.fit(X, y)

scores_knn_uniform = cross_val_score(knn_uniform, X, y, cv=10, scoring='neg_mean_absolute_error')
score_knn_uniform = sum(scores_knn_uniform)/len(scores_knn_uniform)
print("Nearest Neighbours : Uniform")
print("\tAveraged Mean Absolute Error:\t", score_knn_uniform, "\n")
   

knn_distance = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
knn_distance = knn_distance.fit(X, y)

scores_knn_distance = cross_val_score(knn_distance, X, y, cv=10, scoring='neg_mean_absolute_error')
score_knn_distance = sum(scores_knn_distance)/len(scores_knn_distance)
print("Nearest Neighbours : Distance")
print("\tAveraged Mean Absolute Error:\t", score_knn_distance, "\n")
