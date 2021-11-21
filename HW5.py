# Authored by: Ryan Rubadue, Peter Kroeger, Ryan Kunkel, Yahya Emara, Griffin Ramsey
import random
import math
import numpy as np

LEARNING_RATE = 0.01

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

def GetExpectedResults(training, validation, testing): #Getting expected results for error calcs
    trainingExp = list()
    validationExp = list()
    testingExp = list()
    for i in range(len(training)):
        trainingExp.append(training[i][-1])
    for i in range(len(validation)):
        validationExp.append(validation[i][-1])
    for i in range(len(training)):
        testingExp.append(testing[i][-1])
    return trainingExp, validationExp, testingExp

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

    return network

def ForwardProp(network, row, SoT):
    val = row
    new = []
    for nnLayer in network.keys(): #Hidden and Output layers
        for node in network[nnLayer]:
            newVal = 0
            zVal = ZVector(node, val)
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

def BackwardProp(nn, yVal, SoT, forwarPropVals): #Should we only perform backward prop for a single instance? I think so.

    curLayer = nn['output'] #Only 2 layers, accessing the second layer (output) first
    outputError = 0     #Weight for every neuron in hidden layer?
                        #Error list will be from go output layer -> hidden layer (normal order)
    if SoT == "S":
        outputError = (forwarPropVals[-1] - yVal) * DSignmoid(forwarPropVals[-1]) #Calculate error on output layer to propagate to hidden layer
    else:                                            
        outputError = (forwarPropVals[-1] - yVal) * DHyperbolic(forwarPropVals[-1]) #Error = (a-y)*da, da is the derivative of the squish function

    curLayer = nn['hidden']
    hiddenErrors = list()
    for j in range(len(curLayer)): #Iterate through each neuron in hidden layer
        curNeuron = curLayer[j]
        error = 0
        for i in range(len(curNeuron)): #Iterate through each weight in neuron
            error += ((curNeuron[i] * outputError)) #multiplying each weight by the error rate from previous layer to get error
        if SoT == "S":
            error = error * DSignmoid(forwarPropVals[j])
        else:                                            
            error = error * DHyperbolic(forwarPropVals[j])
        
        hiddenErrors.append(error)

    return outputError, hiddenErrors

def UpdateWeights(network, outputError, hiddenErrors, data):
    curLayer = network['hidden']
    for i in range(len(curLayer)):
        curNeuron = curLayer[i]
        for j in range(len(curNeuron)-1):
            curNeuron[j] -= LEARNING_RATE * hiddenErrors[i] * data[j]
    curNeuron[-1] = LEARNING_RATE * hiddenErrors[-1] #Bias weight updated here
    
    curLayer = network['output']
    for i in range(len(curLayer)):
        curNeuron = curLayer[i]
        for j in range(len(curNeuron)):
            curNeuron[j] -= LEARNING_RATE * outputError

    return


def ValidateNetwork(nn, validation, SoT):
    output = 0
    accuracy = 0
    if SoT == "S": #Sigmoid function utilizes a 0 to 1 scale, while hyperbolic uses a -1 to 1 scale
        bound = 0.5
    else:
        bound = 0
    for i in range(len(validation)):
        fullOutput = ForwardProp(nn, validation[i], SoT)
        output = fullOutput[-1]
        if output > bound:
            output = 1
        else:
            output = 0
        if(output == validation[i][-1]):
            accuracy += 1
    accuracy = accuracy / len(validation)
    return accuracy

def TestNetwork(nn, testing, SoT):
    output = 0
    accuracy = 0
    if SoT == "S":
        bound = 0.5
    else:
        bound = 0
    for i in range(len(testing)):
        fullOutput = ForwardProp(nn, testing[i], SoT)
        output = fullOutput[-1]
        if output > bound:
            output = 1
        else:
            output = 0
        if(output == testing[i][-1]):
            accuracy += 1
    accuracy = accuracy / len(testing)
    return accuracy

if __name__ == "__main__":
    training, validation, testing = GenerateDataSets()
    trainingExpected, validationExpected, testingExpected = GetExpectedResults(training, validation, testing)
    numHiddenNodes = len(training[-1]) - 1 # Number of input nodes
    optimumHiddenNodes, maxAccuracy = -1, -1
    oldAccuracySig, oldAccuracyHyper = -1, -1
    for k in range(2):
        if k == 1:
            SoT = "T"
            print(f"Hyperbolic Squash Function: ")
        else:
            SoT = "S"
            print(f"Sigmoid Squash Function: ")
        for i in range(numHiddenNodes, 0, -1):
            forwardPropOutput = list()
            nn = {}
            nn = InitializeNetwork(i, 1)

            for k in range(20): #Performing multiple epochs on the same nn to generate the most accurate results
                for j in range(len(training)):
                    forwardPropOutput.append(ForwardProp(nn, training[j], SoT))
                    outputError, hiddenErrors = BackwardProp(nn, trainingExpected[j], SoT, forwardPropOutput[j])
                    UpdateWeights(nn, outputError, hiddenErrors, training[j])

            curAccuracy = ValidateNetwork(nn, validation, SoT)

            print(f"Accuracy for {i} hidden nodes: {curAccuracy}")

            # hiddenLayer = nn['hidden']
            # for w in range(len(hiddenLayer)):
            #     print(hiddenLayer[w])
            # print(nn['output'][-1])
            
            if curAccuracy >= oldAccuracySig and SoT == "S": #Save the best nn size
                oldAccuracySig = curAccuracy
                bestHiddenSig = i
                bestNNSig = nn
            if curAccuracy >= oldAccuracyHyper and SoT == "T": #Save the best nn size
                oldAccuracyHyper = curAccuracy
                bestHiddenHyper = i
                bestNNHyper = nn

    testAccuracySig = TestNetwork(bestNNSig, testing, "S")
    print(f"The optimum number of hidden nodes for sigmoid function is {bestHiddenSig}\n")
    print(f"Test accuracy for Sigmoid function: {testAccuracySig}")
    testAccuracyHyper = TestNetwork(bestNNHyper, testing, "T")
    print(f"The optimum number of hidden nodes for hyperbolic function is {bestHiddenHyper}")
    print(f"Test accuracy for Hyperbolic function: {testAccuracyHyper}")