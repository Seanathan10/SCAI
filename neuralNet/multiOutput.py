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

    def test(self, testLoader, testSplit, outLength):
        runningAccuracy = 0
        total = 0
        checkingArray = [0 for i in range(outLength)]

        with torch.no_grad():
            for data in testLoader:
                inputs, outputs = data
                print("test inputs :", inputs)
                print("test outputs :", outputs)
                predictedOutputs = self(inputs)
                print("predOuts : ", predictedOutputs)


                for i in range(0, len(outputs[0])):
                        if (abs(outputs[0][i] - predictedOutputs[0][i])/outputs[0][i] < .1):
                            runningAccuracy += 1
                            print("i :", i)
                            checkingArray[i] += len(outputs[0])
                        total += 1

            print('Accuracy of the model based on the test set of', testSplit ,'inputs is: %d %%' % (100 * runningAccuracy / total))
            print('           Actual values')
            for i in range(0, len(checkingArray)) : 
                print(str(i+1) + "th output's accuracy : " + str(100 * checkingArray[i]/total) + "%")

# Grabs data and turns it into usable form:
df = pd.read_csv("../audioRead/SpotifyFeatures.csv")

df.drop(index=df.loc[df["data"] == 0].index, inplace=True)
df.drop(index=df.loc[df["data"] == '0'].index, inplace=True)

df.reset_index(inplace=True)

output = df.loc[:, ["popularity","acousticness", "danceability", "energy","instrumentalness","liveness", "loudness", "speechiness","tempo", "valence"]]
input = df.loc[:, ["data"]]

# print("\n\n\ntypes :")
# print(df["popularity"][0].dtype)
# print(df["acousticness"][0].dtype)
# print(df["danceability"][0].dtype)
# print(df["popularity"][0].dtype)
# print(df["energy"][0].dtype)
# print(df["instrumentalness"][0].dtype)
# print(df["liveness"][0].dtype)
# print(df["loudness"][0].dtype)
# print(df["speechiness"][0].dtype)
# print(df["tempo"][0].dtype)
# print(df["valence"][0].dtype)
# print("\n\n\n")

print(input.dtypes)

for i in range(0, len(input["data"])):
    input["data"][i] = eval(input["data"][i])

print(df)

print("input : ", len(input["data"][2]))

inputArr = np.zeros((len(input["data"]), len(input["data"][0])))
for i in range(0, len(input["data"])):
    for j in range(0, len(input["data"][0])):
        inputArr[i][j] = input["data"][i][j]

# Turns pandas dataframes into tensors and Tensor Dataset
input = torch.Tensor(inputArr)
print("input : ", input)
output = torch.Tensor(output.to_numpy())
print("output : ", output)

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


# TRAINING AND TESTING MODEL!!!

# Actually put it into the model

# For loading current one
# waveModel = generalModel.loadModel(inputSize, outputSize, "waveModel.pth")
# For creating new one
print("input size :", inputSize)
print("Output size :",outputSize)
waveModel = generalModel(inputSize, outputSize)

# Train model
waveModel.trainn(40, trainLoader, validateLoader)


waveModel.test(testLoader, testSplit, outputSize)

# To actually send something through, just call modelName.forward(input array)
# If any of the values you want are not floats, you need to convert that, it will return all floats (or doubles? Not quite sure cuz python is silly)

# # Analyze Training success w/ matplotlib
# epochs = [i for i in range(1, len(globTrainLoss) + 1)]
# fig = plt.figure(tight_layout=True)
# ax = fig.add_subplot(2, 2, 2)
# ax.plot(epochs, globTrainLoss, linewidth=1.5, markersize=0, color='purple')
# ax.set_title("Training Loss")
# ax.set_xlabel('Training Epoch')
# ax.set_ylabel('Loss')
# plt.show()

# Add in global variable so that every test you can plot it in matplotlib
