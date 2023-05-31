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
        self.linear1 = torch.nn.Linear(inputSize, 100)
        self.linear2 = torch.nn.Linear(100, 50)
        self.linear3 = torch.nn.Linear(50, outputSize)

        self.activation = torch.nn.LeakyReLU()

    # Send a tensor through the model
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        return x
    
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

    # Function for training the model
    def trainn(self, numEpochs, trainLoader, validateLoader):
        lossFn = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=0.0001)
        bestAccuracy = 0.0
        
        print("Training with", numEpochs, "epochs...")
        for epoch in range(1, numEpochs + 1):
            # For each epoch resets epoch vars
            runningTrainingLoss = 0.0
            runningAccuracy = 0.0
            runningValLoss = 0.0
            total = 0

            # Actually trains
            for data in trainLoader:
                inputs, outputs = data
                # Zero param gradients
                optimizer.zero_grad()
                predictedOutputs = self.forward(inputs)
                # Sets up and uses backpropogation to optimize
                trainLoss = lossFn(predictedOutputs, outputs)
                trainLoss.backward()
                optimizer.step()
                runningTrainingLoss += trainLoss.item()

            trainLossValue = runningTrainingLoss/len(trainLoader)

            # Validation (AKA Figure out which model change was the best)
            with torch.no_grad():
                self.eval()
                for data in validateLoader:
                    inputs, outputs = data
                    outputs = outputs
                    # Gets values for loss
                    predictedOutputs = self(inputs)
                    valLoss = lossFn(predictedOutputs, outputs)
                    # Highest value will be our prediction
                    _, predicted = torch.max(predictedOutputs, 1)
                    runningValLoss += valLoss.item()
                    for i in range(0, len(outputs[0])):
                        if (abs(outputs[0][i] - predictedOutputs[0][i])/outputs[0][i] < .1):
                            runningAccuracy += 1
                        total += 1
            
            # Calculate Validation Loss Val
            valLossValue = runningValLoss/len(validateLoader)
            # Accuracy = num of correct predictions in validation batch / total predictions done
            accuracy = (100 * runningAccuracy / total)

            # Save model if accuracy is best
            if accuracy > bestAccuracy:
                self.saveModel("waveModel.pth")
                bestAccuracy = accuracy

            # Print current Epoch stats
            globTrainLoss.append(trainLossValue)
            print("Completed training for epoch :", epoch, 'Training Loss is %.4f' %trainLossValue, 'Validation Loss is: %.4f' %valLossValue, 'Accuracy is %d %%' % (accuracy))

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
input = df.loc[:, ['height']]
output = df.loc[:, ['inundate', 'solov']]

# Turns pandas dataframes into tensors and Tensor Dataset
input = torch.Tensor(input.to_numpy())
# print("input : ", input)
output = torch.Tensor(output.to_numpy())
# print("output : ", output)
outputSize = torch.Tensor.dim(output)
data = TensorDataset(input, output)

# Split into a training, validation and testing set
trainBatchSize = 10
testSplit = int(len(input)*0.25)
# print(testSplit)
trainSplit = int(len(input)*0.6)
# print(trainSplit)
validateSplit = len(input) - trainSplit - testSplit
# print(validateSplit)
# print(len(input))
trainSet, validateSet, testSet = random_split(data, [trainSplit, validateSplit, testSplit])

# Get data in loadable form to go into model
trainLoader = DataLoader(trainSet, batch_size=trainBatchSize, shuffle=True)
validateLoader = DataLoader(validateSet, batch_size=1)
testLoader = DataLoader(testSet, batch_size=1)

# Sets input and output size for future models
print(input.shape)
inputSize = list(input.shape)[1]
solovs = []
for i in actualDict['solov']:
    if not i in solovs:
        solovs.append(i)


# TRAINING AND TESTING MODEL!!!

# Actually put it into the model

# For loading current one
# waveModel = generalModel.loadModel(inputSize, outputSize, "waveModel.pth")
# For creating new one
print("input size :", inputSize)
print("Output size :",outputSize)
waveModel = generalModel(inputSize, outputSize)

# Train model
waveModel.trainn(400, trainLoader, validateLoader)


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
