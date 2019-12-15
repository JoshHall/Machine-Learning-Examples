######
# KNN = K-Nearest Neighbors Algo
######
# looks for groupings and groups it into that class
# K = (hyperparameter) the amount of neighbors we look for
# Make K an odd number so there is a clear choice
# make sure K is not set too high
# finds the magnitude with euclidean distance
#       _____________________________
# m =  √(x2−x1)^2+(y2−y1)^2+(z2−z1)^2
# Square root symbol above, in case it is not clear later on
# Z is only for 3 space
######
# Limitations
######
# - computational heavy
#     * have to know the distance to every point
#     * linear and not constant




import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
# print(data.head())

# take labels and encode them into integer values
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
cls = le.fit_transform(list(data['class']))

predict = 'class'

# zip() creates a bunch of tuple values with lists we give it
X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
# print(acc)

predicted = model.predict((x_test))
names = ['unacc', 'acc', 'good', 'vgood']

for x in range(len(predicted)):
    print(f'Predicted: {names[predicted[x]]}, Data: {x_test[x]}, Actual: {names[y_test[x]]}')
    n = model.kneighbors([x_test[x]], 9, True)
    print(f'N: {n}')