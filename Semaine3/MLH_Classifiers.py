# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:23:14 2019

@author: Gregg
"""

#
# The two classifier from MAssachusset Lambin Hospital
#

import pandas as pd

def predict_NSPLB(X):    
    # scale de data
    weigths = (2.6,3.2,6.3,35,0.5,56,4,5,8,2,45,6,5,89,45,15,25,47,56,32,45,78,95,65,32,45,12,45,78,65,35,65,78,65)
    sum = 0
    for i in range(len(X)):
        sum = sum + weigths[i]*X[i]
        
    if (sum>3):
        return 1
    else:
        return 0
    
    
    

def predict_MOC(X):    
    # scale de data
    weigths = (2.6,3.2,5,35,0.5,156,4,5,8,2,45,6,5,89,45,15,25,147,56,32,455,78,95,65,32,45,12,45,78,65,35,65,78,65)
    sum = 0.0
    for i in range(len(X)):
        sum = sum + weigths[i]*X[i]
        
    if (sum<5.6):
        return 1
    else:
        return 0
        
    
from sklearn.base import BaseEstimator, ClassifierMixin

class MOCClassifier(BaseEstimator, ClassifierMixin):  
    """An example of classifier"""

    def __init__(self, intValue=0, stringParam="defaultValue", otherParam=None):
        """
        Called when initializing the classifier
        """
        self.intValue = intValue
        self.stringParam = stringParam

        # THIS IS WRONG! Parameters should have same name as attributes
        self.differentParam = otherParam 


    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """

        assert (type(self.intValue) == int), "intValue parameter must be integer"
        assert (type(self.stringParam) == str), "stringValue parameter must be string"
        #assert (len(X) == 20), "X must be list with numerical values."

        self.treshold_ = 5.6

        return self

    def _meaning(self, x):
        # returns True/False according to fitted classifier
        # notice underscore on the beginning
        return( predict_MOC(x) )

    def predict(self, X, y=None):
        try:
            getattr(self, "treshold_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
            
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values

        return([self._meaning(x) for x in X])

class NSPLBClassifier(BaseEstimator, ClassifierMixin):  
    """An example of classifier"""

    def __init__(self, intValue=0, stringParam="defaultValue", otherParam=None):
        """
        Called when initializing the classifier
        """
        self.intValue = intValue
        self.stringParam = stringParam

        # THIS IS WRONG! Parameters should have same name as attributes
        self.differentParam = otherParam 


    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """

        assert (type(self.intValue) == int), "intValue parameter must be integer"
        assert (type(self.stringParam) == str), "stringValue parameter must be string"
        #assert (len(X) == 20), "X must be list with numerical values."

        self.treshold_ = 3

        return self

    def _meaning(self, x):
        # returns True/False according to fitted classifier
        # notice underscore on the beginning
        return( predict_NSPLB(x) )

    def predict(self, X, y=None):
        try:
            getattr(self, "treshold_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
            
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values

        return([self._meaning(x) for x in X])