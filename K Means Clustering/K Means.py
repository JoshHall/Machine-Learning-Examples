# Centroids = two random points in a graph
# used to help determine groupings of all other points
# after that you move the centroids to the center of their groups
# (X1(point1) + X1(point2) + X1(point3) + etc...)/ amount of points
# (X2(point1) + X2(point2) + X2(point3) + etc...)/ amount of points
# draw the line again and reassign points to the correct centroids
# repeat until no changes occur
#  number of points * number of centroids * number of iterations * number of features(X1,X2,X3,...Xn)
# euclidean distance is basically the absolute distance between two vectors/points in space

import numpy as np
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

digits = load_digits()
# scale all the data down to be between -1 and 1
data = scale(digits.data)
y = digits.target

# dynamic way
# k = len(np.unique(y))
k = 10
samples, features = data.shape

def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

clf = KMeans(n_clusters=k, init='random',n_init=10)
bench_k_means(clf, '1', data)