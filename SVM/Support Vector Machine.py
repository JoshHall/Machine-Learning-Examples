# SVM creates a hyperplane
# find the same distance from the two opposite groupings to the hyperplane
# infinite hyperplanes, find the best
# pick the one that has the largest distance from the hyperplane in order to maximize the margin
# the 2 closet points are support vectors
# margin is the gap from the 2 groupings
# use kernels to bring 2d points into 3d space
# kernel = function(x1, x2) -> x3
# can add another kernel for a 4th dimension
# typically you do not need to create kernels and instead use premade kernels
# soft margin = all for a few points to exist on either side that might not be the correct grouping, used to get a more effective hyperplane
# hard margin = Cannot have incorrect points on the wrong side

import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

# show all the different features and labels
# print(cancer.feature_names)
# print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

# print(x_train, y_train)
classes = ['malignant' 'benign']

# c= soft margin,  use  numbers closer to 0 for hard margin, must be  > 0
clf = svm.SVC(kernel='linear', C=2)
# clf = KNeighborsClassifier(n_neighbors=9)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)

print(acc)