# Authored by: Ryan Rubadue, Peter Kroeger, Ryan Kunkel, Yahya Emara, Griffin Ramsey
import random
import math
import numpy as np

def GenerateDataSets():
    data_set = []
    with open('data_banknote_authentication.txt') as f:
        file_line = f.readline()
        
        data_set.append(file_line.split(','))
        while file_line:
            file_line = f.readline()
            data_set.append(file_line.split(','))
            data_set[-1][-1] = data_set[-1][-1].rstrip('\n')
        np.random.shuffle(data_set)
        chunk = int(len(data_set) / 3)
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
            try:
                zVal = ZVector(node, val)
            except:
                print(nnLayer)
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
        activation += weights[i]*float(vals[i])  
    return activation

def BackwardProp(nn, yVals, SoT):
    for i in range(len(nn) - 1, 0, -1):
        curLayer = nn[i]
        errors = list()
        if i == len(nn) - 1:
            for j in range(len(curLayer)):
                neuron = curLayer[j]
                errors.append(nn['output'] - yVals[j])
        else:
            for j in range(len(curLayer)):
                error = 0
                for neuron in nn[i+1]:
                    error += (nn['weights'][j] * nn['delta'])
                errors.append(error)
        for k in range(len(curLayer)):
            neuron = curLayer[k]
            if SoT == "S":
                neuron['deltaW'] = errors[k] * DSignmoid(neuron['output'])
            else:
                neuron['deltaW'] = errors[k] * DSignmoid(neuron['output'])
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
            print(forwardPropOutput)
            # accuracy = GetAccuracy()
            # if accuracy > maxAccuracy:
            #     maxAccuracy = accuracy
            #     optimumHiddenNodes = i
        
    print(f"The optimum number of hidden nodes for sigmoid function is {optimumHiddenNodes}")
        #outputs
    exit(0)
    pass
