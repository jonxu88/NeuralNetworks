#Goal: To code the backpropogation for the following neural network:
#One input layer with two neurons, one output layer with one neuron, and no biases
#Tested git/github on this file on 9/5/21



#Should use numpy since we are using matrices
#import
import math
import numpy as np

#Forget about asking for user inputs, let's just have constants in the beginning of the code
x_train = np.array([1.5, 0.5])
weights = np.array([0.8, 0.7])
y_train = 0.5
epochs = 10
learning_rate= 0.1

for i in range(epochs):
    C = (x_train @ np.transpose(weights) - y_train)**2 #This is the cost function. The symbol @ is shorthand for matrix multiplication
    dC_dw = np.array(2*x_train[0]*(weights[0]*x_train[0]+weights[1]*x_train[1]-y_train), 
    2*x_train[1]*(weights[0]*x_train[0]+weights[1]*x_train[1]-y_train) )
    weights = weights - learning_rate * dC_dw
    print('Epoch: ' + str(i+1) + ', Weights:' + str(weights) + ', Loss:' + str(C))

#Note this example there are many weights along a line for which the loss is minimised


    

