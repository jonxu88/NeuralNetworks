#Following https://www.youtube.com/watch?v=8d6jf7s6_Qs
# I coded the backpropogation for the simplest neural network.
#There is only one input layer (with one neuron),
# one output layer with one neuron, and no bias or activation function.

print("-----")
print('Enter the number of iterations:') #what is the machine learning term for 'number of iterations?'
iterations = input()
iterations = int(iterations)

print("-----")
print('Enter the learning rate:')
learning_rate = input()
learning_rate = float(learning_rate)

print("------")
print('Enter the starting weight:') #usually this is randomized?
weight = input()
weight = float(weight) #in the video this is set to be 0.8 I believe

inputlayer = 1.5 #Can also edit to change these if desired
desired_output = 0.5



for i in range(iterations):
    dC_dw = inputlayer * 2 * (inputlayer * weight  - desired_output)
    weight = weight - learning_rate * dC_dw
    cost = (inputlayer * weight - desired_output)**2
    print('Iteration Number: ' + str(i+1) + ', Weight:' + str(weight) + ', Cost:' + str(cost))
    

