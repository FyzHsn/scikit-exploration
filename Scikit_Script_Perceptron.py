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
module in sklearn. In contrast to the function we wrote previously, the built-
in python function already has OVR i.e. One-versus-Rest method implemented.

"""
from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=40 , eta0=0.1, random_state=0) 

# random_state sets the int seed for the random number generator. I am assuming 
# that it is used to shuffle the data after each epoch.
# The number of iterations through the data set is n_iter=40
# The learning rate is set to be 0.1
 
# fit the training data
ppn.fit(X_train_std, y_train)

# make predictions with the fitted data
y_pred = ppn.predict(X_test_std) 

# number of wrong predictions
wrong_pred = float((y_pred != y_test).sum())

print('Incorrect predictions: %d' % wrong_pred) 

# %d is a placeholder for a number. Meanwhile %s is a placeholder for a string
# Moreover, you put the actual values in vector format using () brackets
print('This is an %s of %s I mean. You owe me %d %s' % ('example', 'what', 5,
                                                        'dollars'))
print('Accuracy of the model: %f' % (1.0 - wrong_pred / y_test.shape[0]))

# there is a function that finds the accuracy
from sklearn.metrics import accuracy_score
print('Accuracy score using accuracy_score: %.2f' % accuracy_score(y_test, 
                                                                   y_pred))
"""
Next up, we plot the decision regions.

"""                                  
from matplotlib.colors import ListedColormap  
import matplotlib.pyplot as plt

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    
    # Set up marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface - this step is all about covering the data set
    # by a larger rectangle, and considering the grid of values seprated by
    # resolution. ravel() command flattens out the whole grid. 
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                           np.arange(x2_min, x2_max, resolution))
                           
    # We predict the class of the points on the grid via the classifier                           
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    # Contour plot for discrete results showing boundaries of class according
    # to the classifier
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot all sample points: train and test data
    for idx, cl in enumerate(np.unique(y)):
        
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
                    
        # highlight test data
        if test_idx:
            X_test = X[test_idx, :]
            plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, 
                        linewidths=1, marker='o', s=55, label='test set')
    
# plot of the decision boundary with the training and the test data sets
# np.vstack command is equivalent to the R command rbind and binds rows
# together.
X_combined_std = np.vstack((X_train_std, X_test_std))    
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, 
                      y=y_combined, 
                      classifier=ppn, 
                      test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.title('From Raschka\'s book')
plt.savefig('Perceptron_Iris.png')
plt.clf()                      
    


















                               
                                                                   
