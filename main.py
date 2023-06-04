import sys 
import pandas as pd
import numpy as np
import torch
import os 

sys.path.insert(1, "./audioRead/")
sys.path.insert(2, "./neuralNet/")

from audRead import audioMod as ar
from multiOutput import generalModel as gm
import nn_output_similarity as sim

# runs model on song dictionary of fourier tensors 
def run_model(song_dict):
    for song in song_dict:
        song_dict[song] = audioModel.forward(song_dict[song])


# loads songs into fourier transform
def load_songs(directory):
    song_fourier_dict = ar.dir_record(directory)
    convert_to_tensor(song_fourier_dict)

    return song_fourier_dict

def convert_to_tensor(fourier_dict):
    for song in fourier_dict:
        fourier_dict[song] = torch.Tensor(fourier_dict[song])


inputSize = 256
outputSize = 10
audioModel = gm(inputSize, outputSize)
audioModel = audioModel.loadModel(inputSize, outputSize, "./neuralNet/bestInTrain.pth")

directory = input("Please enter a directory: ")

song_dict = load_songs(directory)
print("Done Loading")

run_model(song_dict)
print("Model finished")


# list of strings
play_len = int(input("Please enter how many songs you want in the playlist: "))
playlist = sim.build_playlist(song_dict, play_len)

print("\n".join(playlist))