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

        self.linear1 = torch.nn.Linear(inputSize, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 200)
        self.softmax = torch.nn.Softmax()
        self.linear3 = torch.nn.Linear(200, outputSize)

    # Send a tensor through the model
    def forward(self, y):
        for x in y:
            x = self.linear1(x)
            x = self.activation(x)
            x = self.linear2(x)
            x = self.activation(x)
            x = self.linear3(x)
        return y
    
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
        lossFn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.0001)
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
                outputs = outputs.long()
                # Zero param gradients
                optimizer.zero_grad()
                predictedOutputs = self.forward(inputs)
                # Sets up and uses backpropogation to optimize
                trainLoss = lossFn(predictedOutputs, outputs[:, 0])
                trainLoss.backward()
                optimizer.step()
                runningTrainingLoss += trainLoss.item()

            trainLossValue = runningTrainingLoss/len(trainLoader)

            # Validation (AKA Figure out which model change was the best)
            with torch.no_grad():
                self.eval()
                for data in validateLoader:
                    inputs, outputs = data
                    outputs = outputs.long()
                    # Gets values for loss
                    predictedOutputs = self(inputs)
                    valLoss = lossFn(predictedOutputs, outputs[:, 0])
                    # Highest value will be our prediction
                    _, predicted = torch.max(predictedOutputs, 1)
                    runningValLoss += valLoss.item()
                    total += outputs.size(0)
                    runningAccuracy += (predicted == outputs).sum().item()
            
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

