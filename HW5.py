# Authored by: Ryan Rubadue, Peter Kroeger, Ryan Kunkel, Yahya Emara, Griffin Ramsey
import random
import math
import numpy as np

LEARNINGRATE = 0.01

def GenerateDataSets():
    data_set = []
    with open('data_banknote_authentication.txt') as f:
        file_line = f.readline()
        
        data_set.append(file_line.split(','))
        while file_line:
            file_line = f.readline()
            data_set.append(file_line.split(','))
            data_set[-1][-1] = data_set[-1][-1].rstrip('\n')
        data_set.pop(-1)
        np.random.shuffle(data_set)
        chunk = int(len(data_set) / 3)
    for i in range(len(data_set)):
        for j in range(len(data_set[i])):
            try:
                data_set[i][j] = float(data_set[i][j])
            except ValueError:
                print("Failed to convert array entries from string to float")
                exit(1)
    return data_set[0: chunk], data_set[chunk: 2 * chunk], data_set[2*chunk: len(data_set)]

def Sigmoid(zValue):
    return (1/(1+math.e**(-zValue)))

def DSignmoid(zVal):
    return zVal * (1-zVal)

def Hyperbolic(zValue):
    return ((math.e**(zValue) - math.e**(-zValue))/(math.e**(zValue) + math.e**(-zValue)))

def DHyperbolic(zVal):
    return 1 - (Hyperbolic(zVal))**2

def GenerateRandomWeights(num_rows, num_cols):
    weights = []
    for i in range(num_rows):
        weight_row = []
        for j in range(num_cols):
            weight_row.append(np.random.random())
        weights.append(weight_row)
    return weights

def InitializeNetwork(num_hidden, num_outputs): # Only need one output layer node for binary classification
    # The number of inputs is always 5
    num_inputs = 5
    network = {}

    hidden = GenerateRandomWeights(num_hidden, num_inputs)
    output = GenerateRandomWeights(num_outputs, num_hidden)

    network['hidden'] = hidden
    network['output'] = output

    # for row in network['hidden']:
    #     print("hidden ", row)
    # for row in network['output']:
    #     print("output ", row)

    # print(network['hidden'][1])

    return network

def ForwardProp(network, row, SoT):
    val = row
    for nnLayer in network.keys():
        new = []
        for node in network[nnLayer]:
            newVal = 0
            # zVal = ZVector(node['weights'], val)
            zVal = ZVector(node, val)
            #print(nnLayer)
            if SoT == "S":
                newVal = Sigmoid(zVal)
            else:
                newVal = Hyperbolic(zVal)
            new.append(newVal)
        val = new
    return new

def ZVector(weights, vals):
    activation = 0   
    for i in range(len(vals)):
        activation += weights[i]*vals[i] 
    return activation

def BackwardProp(nn, yVals, SoT, forwarPropVals):

    for i in range(len(nn) - 1, 0, -1): #Start from output layer and work backwards
        curLayer = nn[i]
        errors = list()
        if i == len(nn) - 1:    #If this is the hidden layer, calculate the initial error
            for j in range(len(curLayer)):
                neuron = curLayer[j]
                errors.append(forwarPropVals[j] - yVals[j]) #Error list will be from go output layer -> hidden layer (normal order)
                if SoT == "S":
                    weightChange = errors[j] * DSignmoid(neuron) #There will only be one instance where the loop reaches this part 
                else:                                            #because there is only one neuron in the output layer (i = length of network -1)
                    weightChange = errors[j] * DHyperbolic(neuron)
        else:
            for j in range(len(curLayer)):
                error = 0
                for neuron in nn[i+1]:
                    error += (neuron[j] * weightChange)
                errors.append(error)
        for k in range(len(curLayer)):
            neuron = curLayer[k]
            if SoT == "S":
                weightChange = errors[k] * DSignmoid(neuron)
            else:
                weightChange = errors[k] * DHyperbolic(neuron)
    pass

def GetAccuracy():
    # use MSE on neurons to determine total error for accuracy
    pass

if __name__ == "__main__":
    training, validation, testing = GenerateDataSets()
    # print(training[0])
    # print(len(training))
    numHiddenNodes = len(training[-1]) # Number of input nodes
    optimumHiddenNodes, maxAccuracy = -1, -1
    for k in range(2):
        SoT = "S"
        if k == 1:
            SoT = "T"
        for i in range(numHiddenNodes, 0, -1):
            nn = {}
            nn = InitializeNetwork(i, 1)
            forwardPropOutput = list()
            for j in range(len(training) - 1):
                forwardPropOutput.append(ForwardProp(nn, training[j], SoT))
            #print(forwardPropOutput) # Will get the new list of values to use for the next layer of NN
            # accuracy = GetAccuracy()
            # if accuracy > maxAccuracy:
            #     maxAccuracy = accuracy
            #     optimumHiddenNodes = i
        
    print(f"The optimum number of hidden nodes for sigmoid function is {optimumHiddenNodes}")
        #outputs
    exit(0)
    pass
