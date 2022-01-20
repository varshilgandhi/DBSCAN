# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 11:19:20 2022

@author: varsh
"""

## DBSCAN clustering

##Importing Libraries
import pandas as pd
import numpy as np

## Importing the dataset
dataset = pd.read_csv("Mall_Customers.csv")

dataset.head()

## Only we interested in Annual Income and Spending score
X = dataset.iloc[:, [3,4]].values

## Import the DBSCAN Algorithm
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps = 3, min_samples=4)

## eps means Epsilon values

## Fitting the model
model = dbscan.fit(X)

## Create labels of our model
labels = model.labels_

## In labels -1 indicates that they are outliers
## Remaining points are group of clusters

from sklearn import metrics

## Identifying the points which makes up our core points
sample_cores = np.zeros_like(labels, dtype = bool)

sample_cores[dbscan.core_sample_indices_] = True


#### Meaning of above 2 cells is that if in dbscan when outliers shows indicates them as false
#### and remaining groups indicates as True.


## Calculating the number of clusters
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    

### It prints total number of clusters from the dataset

## Now print metrics score
print(metrics.silhouette_score(X, labels))







