import numpy as np
import audiosegment
from os import system
import matplotlib.pyplot as plt
import os

class audioMod():
    def songStats(audioFile):
        # converts an audio file to a numpy array
        fig = plt.figure(tight_layout=True)
        song1 = (audiosegment.from_file(audioFile).resample(sample_rate_Hz=32000, channels=1, sample_width=2)).to_numpy_array()
        # return song
        song2 = song1[0::4000]
        song3 = np.fft.rfft(song1)
        song4 = np.fft.rfft(song2)
        # print(song)
        
        x = np.arange(0,len(song1))
        y = np.arange(0,len(song2))
        w = np.arange(0, len(song3))
        z = np.arange(0, len(song4))
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        ax1.plot(x, song1)
        ax1.set_title("No fft full song")
        ax2.plot(y, song2)
        ax2.set_title("No fft 1 second")
        ax3.plot(w, song3)
        ax3.set_title("Yes fft full song")
        ax4.plot(z, song4)
        ax4.set_title("Yes fft 1 second")
        a = [0]*100
        for i in range(0, len(song3)):
            if (song3[i] > song3[a[0]]):
                a[0] = i
                a = np.sort(a)
                
        print(a)                
            
        plt.show()
        
    def toArr(audioFile, clean=False):
        song = np.fft.rfft((audiosegment.from_file(audioFile).resample(sample_rate_Hz=32000, channels=1, sample_width=2)).to_numpy_array())
        a = [0]*256
        for i in range(0, len(song)):
            if (song[i] > song[a[0]]):
                a[0] = i
                a = np.sort(a)
        if(clean):
            os.remove(audioFile)
        return a

    def convertWav(audioFile):
        x = "ffmpeg -i " + audioFile + " -v quiet -codec:a libmp3lame -b:a 32k -n " + audioFile[:-3] + "wav"
        system(x)
        os.remove(audioFile)
        
# 4 samples, 100 size each. one at 0:10, 10 seconds before end, and 2 samples in the middle. Each 0.025 of a second