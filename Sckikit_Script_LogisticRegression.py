# -*- coding: utf-8 -*-
"""
In this script, I explore the modelling of discrete classes and their correspo-
nding probabilities using the Logistic Regression  algorithm. Note that regres-
sion in this case has nothing to do with regression analysis from statistics.

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
