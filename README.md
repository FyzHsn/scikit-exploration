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
Moreover, if one attaches a likelihood function that is to be maximized to be probabilistically more correct, it is exactly equivalent to minimizing the cost function associated with the sigmoid activation function.  
* regularization to prevent overfitting of data  
* maximum margin classification with support vector machines (SVM)    
* dealing with nonlinearly separable case using slack variables and kernel SVM  
* decision tree learning and maximizing  information gain  
* combining weak to strong learners via random forests  
* K-nearest neighbors  
