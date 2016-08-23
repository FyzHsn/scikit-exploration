# -*- coding: utf-8 -*-
"""
In this script, we explore the Support Vector Machines (SVM) algorithm. This
algorithm is similar to the perceptron. It involves correctly indentifying
some test data set with the weights being updated accordingly in addition to
simulatneously minimizing the magnitude of the weights. More intuitively, this
procedure ends up finding a decision boundary that is maximum distance away 
from the negative and positive hyperplanes. Lastly, another improvement over
the perceptron is the introduction of the slack variable to combat overfitting.

"""
import matplotlib.pyplot as plt
import numpy as np

# function to plot decision boundaries predicted by algorithms
from matplotlib.colors import ListedColormap  

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
            X_test, y_test = X[test_idx, :], y[test_idx]
            plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, 
                        linewidths=1, marker='o', s=55, label='test set')
# end of function to plot decision boundaries predicted by algorithms

# preprocessing iris dataset
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0)

# standardize data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train) 
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))    
y_combined = np.hstack((y_train, y_test))
# end of preprocessing iris dataset

"""
Maximum margin classification with linear Support Vector Machines (SVM).
From the module svm we import the function SVC.
This is the meat of the script.

"""
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=0.015, random_state=0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=svm, 
                      test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.title('Linear SVM, C=0.015 - From Raschka\'s book \n Underfitting')
plt.savefig('SVMC0p01.png')
plt.clf()
#plt.show()

svm = SVC(kernel='linear', C=1, random_state=0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=svm, 
                      test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.title('Linear SVM, C=1 - From Raschka\'s book \n Good compromise')
#plt.savefig('SVMC1.png')
#plt.clf()
plt.show()

# SVM using a non-linear kernel
# generate random data cluster
np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

# plot clusters
plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1],
            c='b', marker='x', label='1')
plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1],
            c='r', marker='s', label='-1')
plt.ylim(-3.0)
plt.legend()
plt.show()            

# Apply SVM algorithm to the non-linearly separable data
svm = SVC(kernel='rbf', random_state=0, gamma=0.1, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.xlabel('X_xor[:, 0]')
plt.ylabel('X_xor[:, 1]')
plt.legend(loc='upper left')
plt.title('Non-linear (RBF) SVM, gamma=0.1, C=10 \n From Raschka\'s book')
plt.show()

# SVM using a nonlinear RBF (radial basis function) kernel
svm = SVC(kernel='rbf', random_state=0, gamma=100.0, C=1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=svm, 
                      test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.title('Non-linear (RBF) SVM, gamma=100, C=1 \n From Raschka\'s book')
plt.show()

# Play with gamma and C