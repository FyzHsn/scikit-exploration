# -*- coding: utf-8 -*-
"""
In this script, I explore the modelling of discrete classes and their correspo-
nding probabilities using the Logistic Regression  algorithm. Note that this 
algortihm is used for classification and not regression fitting.

So far, we have seen the Heaviside (constant with jump discontinuity) and the 
linear activation function. Logistic regression or logit uses a sigmoid functi-
on which looks linear close to the center and become more constant as we appro-
ach positive/negative infinities. 

Moreover, mathematically, we see that maximizing the likelihood function leads
to minimizing the cost function from the adaline algorithm.

"""

# import libraries
import matplotlib.pyplot as plt
import numpy as np

# plot of the sigmoid function
# create sigmoid method
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
    
# points from range -7 to 7
z = np.arange(-7, 7, 0.1)

# sigmoid activation function
phi_z = sigmoid(z)

# plot of function
plt.plot(z, phi_z)    
plt.axvline(0.0, color = 'k')
plt.axhspan(0.0, 1.0, facecolor='1.0', alpha=1.0, ls='dotted')
plt.axhline(y=0.5, ls='dotted', color='k')
plt.yticks([0.0, 0.5, 1.0])
plt.ylim([-0.1, 1.1])
plt.xlabel('z')
plt.ylabel('$\phi (z)$')
plt.title('Sigmoid Activation Function gives \n probability of classification')
#plt.savefig('SigmoidActivation.png')
#plt.clf()
plt.show()

# to implement logistic regression, we could either change the cost function
# in the AdalineGD code or use scikit-learns built-in function which is more
# sophisticated.
from sklearn.linear_model import LogisticRegression

# Processed Iris dataset. Processed Iris dataset. Processed Iris dataset.
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train) 
X_test_std = sc.transform(X_test)

# decision regions code
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

# call the logistic regression function. C is related to regularization
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)

# plot decision regions
X_combined_std = np.vstack((X_train_std, X_test_std))    
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, 
                      y=y_combined, 
                      classifier=lr, 
                      test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.title('Logistic Regression - From Raschka\'s book')
plt.savefig('LogisticRegression.png')
plt.clf()                      
#plt.show()



