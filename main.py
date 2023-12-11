import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return max(0, x)

def myactivation(x):
    return pow(x, 2)/2