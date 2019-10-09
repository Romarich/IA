# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
"""from sklearn.model_selection import GridSearchCV"""
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

x_moons,y_moons = make_moons(n_samples=10000, noise=0.4, random_state=0) 

x_train, x_test, y_train, y_test = train_test_split(x_moons,y_moons,test_size=0.1,random_state=0)



modelTree = DecisionTreeClassifier(max_leaf_nodes=60)
modelTree.fit(x_train,y_train)
y_predit = modelTree.predict(x_test)
print(y_predit)

print("Accuracy:", metrics.accuracy_score(y_test,y_predit))

from sklearn.model_selection import cross_val_score
all_accuracies = cross_val_score(estimator=modelTree, X=x_train, y=y_train, cv=5)
print(all_accuracies)
print(all_accuracies.mean())





""" KIT LEGO """

import os
import pandas as pd
import matplotlib.pyplot as plt

LEGO_PATH = "U:\IA\IA\Semaine4"

def load_lego_data(lego_path,csvGived):
    csv_path = os.path.join(lego_path,csvGived)
    return pd.read_csv(csv_path)

lego_data_training_set = load_lego_data(LEGO_PATH,"train.csv")
lego_data_test_set = load_lego_data(LEGO_PATH,"test.csv")

from sklearn.ensemble import RandomForestRegressor
"""from sklearn.model_selection import GridSearchCV"""



modelRandomForest = RandomForestRegressor()
modelRandomForest.fit(lego_data_training_set, lego_data_test_set)

from sklearn.model_selection import cross_val_score
all_accuracies = cross_val_score(estimator=modelRandomForest, X=x_train, y=y_train, cv=5)
print(all_accuracies)
print(all_accuracies.mean())
