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
* maximum margin classification with support vector machines (SVM) is an off shoot of the perceptron. In addition to correctly identifying the test set, there is the additional condition of maximizing the distance of te decision boundary to that of the negative and positve hyperplane boundary which is achieved by minimzing the weight coefficient while correctly identifying the test set. Moreover, as this algorithm can also end up overfitting the data, the notion of slack variables was introduced by Vladimir Vapnik. The slack variables work by allowing us to tune the penalties for misclassification.  
* dealing with nonlinearly separable case using slack variables and kernel SVM  
* decision tree learning and maximizing  information gain  
* combining weak to strong learners via random forests  
* K-nearest neighbors  
