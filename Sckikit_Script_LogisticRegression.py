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
plt.savefig('SigmoidActivation.png')
plt.clf()