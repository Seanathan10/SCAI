import numpy as np
import audiosegment
from os import system
import random

class audioMod():
    def toArr(audioFile):
        # converts an audio file to a numpy array
        song = (audiosegment.from_file(audioFile).resample(sample_rate_Hz=4000, sample_width=2, channels=1)).to_numpy_array()
        return song

    def convertWav(audioFile):
        x = "ffmpeg -i " + audioFile + " -v quiet -codec:a libmp3lame -b:a 32k -y " + audioFile[:-3] + "wav"
        system(x)
        
# 4 samples, 100 size each. one at 0:10, 10 seconds before end, and 2 samples in the middle. Each 0.025 of a second