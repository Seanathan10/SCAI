import torch
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
from torch.utils.data import random_split, TensorDataset, DataLoader
import math

globTrainLoss = []

class generalModel(torch.nn.Module):
    # Initialize model
    def __init__(self, inputSize, outputSize):
        super(generalModel, self).__init__()

        # Sets up model layers : 

        # Each layeri is just a linear layer that has weights and actually does the weight calculations and changes over time
        # The activation is a equation layer between the linear ones to make it actually change between layers
        # Softmax takes the best of the given results (Used at the end to determine which option is the best)
        # (Not actually used in this order, this just sets up what they are
        self.linear1 = torch.nn.Linear(inputSize, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 200)
        self.softmax = torch.nn.Softmax()
        self.linear3 = torch.nn.Linear(200, outputSize)

    # Send a tensor through the model
    def forward(self, x):
        # This is accurate for the most part, except for some reason I didn't do softmax at the end, and I'm not sure why.
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        # Usually you'd do a softmax but right now it's forcing all guesses to 1. I think this is overfitting, but I can explian more in depth in person
        return x
    
    #Following 2 functions are used in conjunction with each other

    # Saves model to file
    def saveModel(self, name):
        path = "./" + name 
        torch.save(self.state_dict(), path)

    # Loads model from file
    def loadModel(inputSize, outputSize, path):
        model = generalModel(inputSize, outputSize)
        model.load_state_dict(torch.load("./" + path))
        model.eval()
        return model

    # The actual main part

    # Function for training the whole model
    def trainn(self, numEpochs, trainLoader, validateLoader):
        # Sets loss fn, this one is specifically for classification (We need to change this for ours)
        lossFn = torch.nn.CrossEntropyLoss()
        # Sets the optimizer, pretty standard, using ADAM is pretty much the best in almost all situations
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.0001)
        # Accuracy gets set and determines which model is the best
        bestAccuracy = 0.0
        
        print("Training with", numEpochs, "epochs...")
        # Training epoch loop begins here
        for epoch in range(1, numEpochs + 1):
            # For each epoch resets epoch vars
            runningTrainingLoss = 0.0
            runningAccuracy = 0.0
            runningValLoss = 0.0
            total = 0

            # Actually trains
            for data in trainLoader:
                # Compares actual output to predicted output
                inputs, outputs = data
                outputs = outputs.long()
                # Zero param gradients
                optimizer.zero_grad()
                predictedOutputs = self.forward(inputs)
                # Sets up and uses backpropogation to optimize
                trainLoss = lossFn(predictedOutputs, outputs[:, 0])
                trainLoss.backward()
                optimizer.step()
                #Gets loss of all of this
                runningTrainingLoss += trainLoss.item()

            trainLossValue = runningTrainingLoss/len(trainLoader)

            # Validation (AKA Figure out which model change was the best)
            with torch.no_grad():
                self.eval()
                for data in validateLoader:
                    # Get the outputs and inputs, toss inputs through model, and compare the outputs from model (predicted outputs) with the actual outputs
                    inputs, outputs = data
                    outputs = outputs.long()
                    # Gets values for loss
                    predictedOutputs = self(inputs)

                    # Toss them into the loss function
                    valLoss = lossFn(predictedOutputs, outputs[:, 0])
                    # Highest value will be our prediction
                    _, predicted = torch.max(predictedOutputs, 1)

                    # Adding up all the loss and all the accuracy over time
                    runningValLoss += valLoss.item()
                    total += outputs.size(0)
                    print(outputs.size(0))
                    runningAccuracy += (predicted == outputs).sum().item()
            
            # Calculate Validation Loss Val
            valLossValue = runningValLoss/len(validateLoader)
            # Accuracy = num of correct predictions in validation batch / total predictions done
            accuracy = (100 * runningAccuracy / total)

            # Save model if accuracy is better than current best
            if accuracy > bestAccuracy:
                self.saveModel("waveModel.pth")
                bestAccuracy = accuracy

            # Print current Epoch stats
            globTrainLoss.append(trainLossValue)
            print("Completed training for epoch :", epoch, 'Training Loss is %.4f' %trainLossValue, 'Validation Loss is: %.4f' %valLossValue, 'Accuracy is %d %%' % (accuracy))

    # Tests the model accuracy over a number of samples
    def test(self, testLoader, testSplit, solovs):
        runningAccuracy = 0
        total = 0
        checkingArray = [[0 for i in range(len(solovs))] for j in range(len(solovs))]
        print(solovs)
        print(type(solovs[0]))

        with torch.no_grad():
            for data in testLoader:
                inputs, outputs = data
                outputs = outputs.to(torch.float32) 
                predictedOutputs = self(inputs)
                _, predicted = torch.max(predictedOutputs, 1)
                print(predicted.item())
                print(outputs.item())
                predIndex = solovs.index(float(predicted.item()))
                outIndex = solovs.index(int(outputs.item()))
                checkingArray[predIndex][outIndex] += 1

                total += outputs.size(0)
                runningAccuracy += (predicted == outputs).sum().item()

            checkDf = pd.DataFrame(checkingArray, columns= solovs)
            checkDf.index = solovs
            print('Accuracy of the model based on the test set of', testSplit ,'inputs is: %d %%' % (100 * runningAccuracy / total))
            print('           Actual values')
            print(checkDf)

# Grabs data and turns it into usable form:

# Grabs raw list of dictionaries
everythingDict=[*csv.DictReader(open('oldTestData.csv'))]
actualDict = {}
actualDict['solov'] = []
actualDict['height'] = []
actualDict['inundate'] = []

# Turns list of dictionaries into dictionary of lists
for i in range(0, len(everythingDict)):
    if float(everythingDict[i]['solovievIdentity']) - math.floor(float(everythingDict[i]['solovievIdentity'])) > .5:
        tempSolov = math.ceil(float(everythingDict[i]['solovievIdentity']))
    else:
        tempSolov = math.floor(float(everythingDict[i]['solovievIdentity']))
    actualDict['solov'].append(tempSolov)
    actualDict['height'].append(float(everythingDict[i]['waveHeight']))
    actualDict['inundate'].append(float(everythingDict[i]['horizontalInundation']))

# Creates Pandas dataframe for input and output
df = pd.DataFrame(actualDict)
input = df.loc[:, ['height', 'inundate']]
output = df.loc[:, ['solov']]

# Turns pandas dataframes into tensors and Tensor Dataset
input = torch.Tensor(input.to_numpy())
print(input)
output = torch.Tensor(output.to_numpy())
data = TensorDataset(input, output)

# Split into a training, validation and testing set
trainBatchSize = 10
testSplit = int(len(input)*0.25)
print(testSplit)
trainSplit = int(len(input)*0.6)
print(trainSplit)
validateSplit = len(input) - trainSplit - testSplit
print(validateSplit)
print(len(input))
trainSet, validateSet, testSet = random_split(data, [trainSplit, validateSplit, testSplit])

# Get data in loadable form to go into model
trainLoader = DataLoader(trainSet, batch_size=trainBatchSize, shuffle=True)
validateLoader = DataLoader(validateSet, batch_size=1)
testLoader = DataLoader(testSet, batch_size=1)

# Sets input and output size for future models
inputSize = list(input.shape)[1]
solovs = []
for i in actualDict['solov']:
    if not i in solovs:
        solovs.append(i)
print(solovs)
outputSize = len(solovs)


# TRAINING AND TESTING MODEL!!!

# Actually put it into the model

# For loading current one
# waveModel = generalModel.loadModel(inputSize, outputSize, "waveModel.pth")
# For creating new one
waveModel = generalModel(inputSize, outputSize)

# Train model
print("input size :", inputSize)
print("output size :", outputSize)
waveModel.trainn(150, trainLoader, validateLoader)

waveModel.test(testLoader, testSplit, solovs)

# Analyze Training success w/ matplotlib
epochs = [i for i in range(1, len(globTrainLoss) + 1)]
fig = plt.figure(tight_layout=True)
ax = fig.add_subplot(2, 2, 2)
ax.plot(epochs, globTrainLoss, linewidth=1.5, markersize=0, color='purple')
ax.set_title("Training Loss")
ax.set_xlabel('Training Epoch')
ax.set_ylabel('Loss')
plt.show()

# Add in global variable so that every test you can plot it in matplotlib