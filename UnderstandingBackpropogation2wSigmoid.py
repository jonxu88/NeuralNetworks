#Following https://www.youtube.com/watch?v=8d6jf7s6_Qs
# I coded the backpropogation for the simplest neural network.
#There is only one input layer (with one neuron),
# one output layer with one neuron, with sigmoid activation function.

#import math and define the sigmoid function
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))



print("-----")
print('Enter the input sample (set to 1.5 in the video):')
inputlayer = input()
inputlayer = float(inputlayer)

print("-----")
print('Enter the desired output (set to 0.5 in the video):')
desired_output = input()
desired_output = float(desired_output)

print("------")
print('Enter the starting weight:') #usually this is randomized?
weight = input()
weight = float(weight) #in the video this is set to be 0.8 I believe

print("------")
print('Enter the starting bias:') #usually this is randomized
bias = input()
bias = float(bias) 

print("-----")
print('Enter the number of iterations:') #what is the machine learning term for 'number of iterations?'
iterations = input()
iterations = int(iterations)

print("-----")
print('Enter the learning rate:')
learning_rate = input()
learning_rate = float(learning_rate)


for i in range(iterations):
    z = weight * inputlayer + bias
    dC_dw = 2 * inputlayer * (sigmoid(z) - desired_output) * sigmoid(z) * (1-sigmoid(z))
    dC_db = 2 * (sigmoid(z) - desired_output)* sigmoid(z) * (1-sigmoid(z))
    weight = weight - learning_rate * dC_dw
    bias = bias - learning_rate * dC_db 
    cost = (sigmoid(z)- desired_output )**2
    print('Iteration Number: ' + str(i+1) + ', Weight:' + str(weight) + ', Bias:' + str(bias) + ', Cost:' + str(cost))
    

