import sys 
import pandas as pd
import numpy as np
import torch
import os 

sys.path.insert(1, "./audioRead/")
sys.path.insert(2, "./neuralNet/")

from audRead import audioMod as ar
from multiOutput import generalModel as gm

x = ar.model_record()
input = torch.Tensor(x)
inputSize = 256
outputSize = 10
audioModel = gm(inputSize, outputSize)
audioModel = audioModel.loadModel(inputSize, outputSize, "./neuralNet/bestInTrain.pth")
output = audioModel.forward(input)
print(output)

