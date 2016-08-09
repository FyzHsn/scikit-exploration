# -*- coding: utf-8 -*-
"""
In this script, I play around with the Perceptron algorithm and splitting data 
sets into training and test subsets with the aim of getting familiear with
scikit-learn. I am following Sebastian Raschka's Python Machine Learning book.

"""
# importing list of popular datsets from scikit
from sklearn import datasets
import numpy as np

# load iris dataset
iris = datasets.load_iris()

# I am finding that if the dataset wasn't imported using pandas, then the 
# head and tail commands don't work.
# iris.tail()
X = iris.data[:, [2, 3]]
y = iris.target

# To print a pre-stored data set from the scikit package, this is the command
print(iris.data[0:5, ])
print(X[1:5, ])

# Find out unique elements of the target variable
print(np.unique(y))

# test and train data split
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0)
                                                    
print(X_train.shape)
print(X_test.shape)                                

# Using the test_train_split function from scikit-learn's cross_validation
# module, we split the x and y arrays into 30 percent test and 70 percent (45)
# training data (105 samples).

# Since, many machine learning algorithms perform better for standardized data
# such as the adaline, we standardize the training and test data via the
# pre-processing module's StandardScaler function.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)

# extract the mean and variance from the standard scalar fit
print(sc.fit(X_train).mean_)
print(sc.fit(X_train).var_)
 
# Standardize data
X_train_std = sc.transform(X_train) 
X_test_std = sc.transform(X_test)

"""
Having loaded, split and standardized the data, now we apply the perceptron 
learning model. We import the Perceptron function from the linear_model
module in sklearn.

"""
from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=40 , eta0=0.1, random_state=0) 

# random_state sets the int seed for the random number generator. I am assuming 
# that it is used to shuffle the data after each epoch.
# The number of iterations through the data set is n_iter=40
# The learning rate is set to be 0.1
 
# fit the training data
ppn.fit 
 