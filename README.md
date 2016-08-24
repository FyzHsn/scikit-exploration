scikit-learn
============

This repository contains my explorations in the scikit-learn package.

Scikit-learn package
--------------------

1. Classification  
2. Regression  
3. Clustering     
4. Dimensionality reduction    
5. Model selection    
6. Pre-processing   

More information can be found on the sklearn [website](http://scikit-learn.org/stable/).  
  
In particular, we explore:  
* [branching data sets into test and train subsets](https://github.com/FyzHsn/scikit-exploration/blob/master/Scikit_Script_Perceptron.py)  
* [perceptron](https://github.com/FyzHsn/scikit-exploration/blob/master/Scikit_Script_Perceptron.py)      
![](https://github.com/FyzHsn/scikit-exploration/blob/master/Perceptron_Iris.png?raw=true)  
* modeling class [logistic regression](https://github.com/FyzHsn/scikit-exploration/blob/master/Sckikit_Script_LogisticRegression.py) allowing us to attach probability/confidence to our predictions. This is achieved by replacing the linear activation function in the Adaline Gradient Descent algorithm by the sigmoid function shown below.        
![](https://github.com/FyzHsn/scikit-exploration/blob/master/SigmoidActivation.png?raw=true)  
Moreover, if one attaches a likelihood function that is to be maximized for the model to be probabilistically most correct, it is exactly equivalent to minimizing the cost function associated with the sigmoid activation function. The decision boundary from logistic regression is given below.    
![](https://github.com/FyzHsn/scikit-exploration/blob/master/LogisticRegression.png?raw=true)  
* regularization to prevent overfitting of data -  One can have a model that is not complex enough to fit the data and leads to poor classification. This is known as underfitting. On the other hand, it is also possible to have a model that is too complex for the underlying data. In this case, it won't work well for unseen data. Overfitting is easier to see in non-linearly separable data sets. Regularization (in our case L2) leads to a good compromise between the two extremes by tuning the complexity (C) parameter in the algorithms. Another way to think of this is to note that for higher regularization, extreme weights are rejected (Of course it's a balancing act between that and correctly classifying the samples). Dependence of the weight coefficient on the parameter C can be seen in the following plot from Raschka's book.  
![](https://github.com/FyzHsn/scikit-exploration/blob/master/weightcoeffcomplexity.png?raw=true)  
* maximum margin classification with support vector machines (SVM) is an off shoot of the perceptron. In addition to correctly identifying the test set, there is the additional condition of maximizing the distance of the decision boundary to that of the negative and positve hyperplane boundary which is achieved by minimzing the weight coefficient while correctly identifying the test set. Moreover, as this algorithm can also end up overfitting the data, the notion of slack variables was introduced by Vladimir Vapnik. The slack variables work by allowing us to tune the penalties for misclassification. The variable C in the SVC function controls the relative importance of the slack variables. Here are two plots to show the effects of underfitting and good compromise.  
![](https://github.com/FyzHsn/scikit-exploration/blob/master/SVMC0p01.png?raw=true)  
![](https://github.com/FyzHsn/scikit-exploration/blob/master/SVMC1.png?raw=true)  
* dealing with nonlinearly separable case using slack variables (Shown above) and kernel SVM. Non-linear cases can be dealth with using either slack variables in the SVM algorithm. This leads to a good compromise from overfitting. We tune the parameter C to get an appropriate fit for the Iris dataset. Next, consider the non-linearly separable data set and its corresponding decision boundary using a radial basis function (RBF) kernel in the SVM algorithm. scikit-learn, conveniently, has built in functions for this.    
![](https://github.com/FyzHsn/scikit-exploration/blob/master/SVMNonlinearlySeparableData.png?raw=true)    
Lastly, the RBF kernel uses another parameter gamma, which can be tuned in a way to lead to various degrees of fitting. For example,  
![](https://github.com/FyzHsn/scikit-exploration/blob/master/NonlinearSVMIrisData.png?raw=true)    
* decision tree learning and maximizing  information gain via measures of impurity. Decision tree learning is based on splitting up a parent dataset into various nodes and based on some measure of information gained (i.e. purity of class labels in the child datasets). Though this procedure can be continued until all the leaves are pure that might lead to overfitting. THerefore, a maximum depth of the decision tree is set. It is called pruning the tree. The information gain functions are based on measures of impurity or how mixed the class labels of the nodes are. The three major measures are entropy, Gini and classification error impurity. A comparison of the three measures is shown below for binary node splits.  
![](https://github.com/FyzHsn/scikit-exploration/blob/master/ImpurityMeasures.png?raw=true)  
Furthermore, the decision boundary regions using the built in scikit decision tree learning algorithms is shown below (maximum depth of 4):  
![](https://github.com/FyzHsn/scikit-exploration/blob/master/DecisionTreeLearningIris.png?raw=true)  

* combining weak to strong learners via random forests. Ensembles of decision trees can be considered by combining weak learners (robust to parameter tuning) to strong learners (avoids overfitting) for n_estimators=10, n_jobs=2:  
![](https://github.com/FyzHsn/scikit-exploration/blob/master/RandomForestIris.png?raw=true)  
n_estimators are the number of decision trees used.  
* K-nearest neighbors  
