# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 16:38:06 2019

@author: Gregg
"""
import pandas as pd
from MLH_Classifiers import MOCClassifier
from MLH_Classifiers import NSPLBClassifier


# Import recurring cancer dataset
cancerDF = pd.read_csv("data 2.csv")
features = cancerDF.drop(["id", "diagnosis", "Unnamed: 32"], axis=1)
labels_cat = cancerDF["diagnosis"]
# Convert categories to num labels
labels_num = [ 1 if x == 'M' else 0 for x in labels_cat]


# Evaluate MOC & NSPLB classifiers

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test \
     = train_test_split(features, labels_num, test_size=0.3, random_state=10)

moc_classifer = MOCClassifier()
moc_classifer.fit(features_train, labels_train)
labels_pred_moc = moc_classifer.predict(features_test)

from sklearn.metrics import accuracy_score
accuracy_moc = accuracy_score(labels_test, labels_pred_moc)

print ("Accuracy MOC = ", accuracy_moc)

nsplb_classifer = NSPLBClassifier()
nsplb_classifer.fit(features_train, labels_train)
labels_pred_nsplb = nsplb_classifer.predict(features_test)

accuracy_nsplb = accuracy_score(labels_test, labels_pred_nsplb)

print ("Accuracy NSPLB = ", accuracy_nsplb)

#from sklearn.metrics import confusion_matrix

#print("confusion_matrice_moc")
#confusion_matrice_moc = confusion_matrix(labels_test,labels_pred_moc)
#print(confusion_matrice_moc)'

#print("confusion_matrice_nsplb")
#confusion_matrice_nsplb = confusion_matrix(labels_test,labels_pred_nsplb)
#print(confusion_matrice_nsplb)
#
#from sklearn.metrics import precision_score, recall_score
#print("metrique de précision moc")
#print(precision_score(labels_test,labels_pred_moc))
#print("metrique de précision nsplb")
#print(precision_score(labels_test,labels_pred_nsplb))
#
#print("metrique de recall moc")
#print(recall_score(labels_test,labels_pred_moc))
#print("metrique de recall nsplb")
#print(recall_score(labels_test,labels_pred_nsplb))
#
#print(features.info())

from sklearn.neighbors import NearestNeighbors
import sys
import numpy
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(features_train,labels_train)
numpy.set_printoptions(threshold=sys.maxsize)
#print(nbrs.kneighbors(features_train))
print(nbrs.kneighbors_graph(features_test).toarray())

