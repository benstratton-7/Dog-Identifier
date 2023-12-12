import numpy as np
import matplotlib.pyplot as plt
import process_data
# from sys import maxsize
# np.set_printoptions(threshold=maxsize)

image_channels = 3
input_size = process_data.size_of_processed_image * process_data.size_of_processed_image * image_channels
hidden_size = 224
output_size = 2

weights_hidden = np.random.randn(input_size, hidden_size)

print(weights_hidden)

#activation functions
def sigmoid(x):
    return 1/(1+np.exp(-x))
def relu(x):
    return max(0, x)
def myactivation(x):
    return pow(x, 2)/2

#loss functions
def cross_entropy(Y, P):
    epsilon = 1e-10
    loss = -np.mean(np.sum(Y*np.log(P + epsilon), axis=1))
    return loss