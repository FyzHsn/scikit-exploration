# -*- coding: utf-8 -*-
"""
In this script, I work through Raschka's section on Decision Tree Learning. 

"""
# Comparison of the three measures of impurity given by entropy, gini and 
# classification error.
import matplotlib.pyplot as plt
import numpy as np

# define gini impurity
def gini(p):
    return (p)*(1-p) + (1-p)*(1-(1-p))
    
# define entropy impurity
def entropy(p):
    return -p*np.log2(p) - (1-p)*np.log2(1-p)
    
# define classification error
def error(p):
    return 1 - np.max([p, 1-p])
    
# break up the interval 0 to 1 into partitions
x = np.arange(0.0, 1.0, 0.01)

# some fancy coding: function * decision * loop
ent = [entropy(p) if p!=0 else None for p in x]
err = [error(i) for i in x]

# plot figure
fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c, in zip([ent, gini(x), err],
                          ['Entropy', 'Gini Impurity', 'Misclassification'],
                          ['-', '--', '-.'],
                          ['black', 'lightgray', 'red']):
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=3, fancybox=True, shadow=False)    
plt.ylim([0, 1.1])
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')          
plt.savefig('ImpurityMeasures.png')
plt.clf()
#plt.show()                              

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
            X_test, y_test = X[test_idx, :], y[test_idx]
            plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, 
                        linewidths=1, marker='o', s=55, label='test set')

# Built in Decision Tree Learning algorithms from scikit-learn
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=0)
tree.fit(X_train, y_train)

# combined data set
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

# decision region boundaries
plot_decision_regions(X_combined, y_combined,
                      classifier=tree, test_idx=range(105, 150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.savefig('DecisionTreeLearningIris.png')
plt.clf()
#plt.show()                

"""
Random forests are ensemble of decision trees combining a robust wek learner
to a strong learner that is less susceptible to overfitting. The steps are
as follows for the random forest algorithm:
1. Draw a random bootstrap sample size n.
2. Grow a decision tree from each sample. At each node:
   * Randomly select d features without replacement.
   * Split the node using the feature that provides the best split via the IG
     function.
3. Repeat steps 1 to 2 k times.
4. Aggregate the prediction by each tree to assign the class label by majority 
   vote. 
   
"""
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=10,
                                random_state=1, 
                                n_jobs=2)
forest.fit(X_train, y_train)
plot_decision_regions(X_combined, y_combined,
                      classifier=forest, test_idx=range(105, 150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.savefig('RandomForestIris.png')
plt.clf()
#plt.show()                                

"""
k-nearest neighbors - lazy learning algorithm
Parametric models such as linear SVM, Perceptron estimate a fixed number of 
parameters to describe the data.
Non-parametric models such as decision tree learning and kernel SVM are descri-
bed by parameters that grow with the data. Lazy learning is a special case with
zero learning cost.

"""
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train, y_train)
plot_decision_regions(X_combined, y_combined,
                      classifier=knn, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')                      
plt.ylabel('petal width [standardized]')
plt.title('k-nearest neighbors on Iris dataset \n From Raschka\'s book')
plt.savefig('knnIris.png')
plt.clf()
#plt.show()                      

