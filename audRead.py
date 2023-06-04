import numpy as np
import audiosegment
from os import system
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob2 as g

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
        
        x = np.arange(0, len(song1))
        y = np.arange(0, len(song2))
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
        new_file = audioFile.replace(" ", "_")
        os.rename(audioFile, audioFile.replace(" ", "_"))
        x = "ffmpeg -i " + new_file + " -v quiet -codec:a libmp3lame -b:a 32k -n " + new_file[:-3] + "wav"
        system(x)
        os.remove(new_file)
    
    def updateCSV():
        df = pd.read_csv("./SpotifyFeatures.csv/", dtype={"genre" : "string", "artist_name" : "string", 
                                                       "track_name" : "string", "track_id" : "string",
                                                       "popularity" : float, "acousticness" : float,
                                                       "danceability" : float, "duration_ms" : int,
                                                       "energy" : float, "instrumentalness" : float,
                                                       "key" : "string", "liveness" : float,
                                                       "loudness" : float, "mode" : "string",
                                                       "speechiness" : float, "tempo" : float,
                                                       "time_signature" : "string", "valence" : float,
                                                       "data" : "string"}, encoding="utf-8")
        np.set_printoptions(linewidth=np.inf)
        
        if("data" not in df.columns):
            df["data"] = "0"

        for file in g.glob("*.wav"):
            wav_data = audioMod.toArr(file, clean=False)
            converted_arr = np.array2string(wav_data, precision=0, separator=',')
            
            file = (file.replace("_", " ")[:-4])#.lower()
            # print(file)
            if(file in (df["track_name"].values)):
                # row_num = df[df["track_name"].apply(lambda x: x.lower()) == file].index[0]
                row_num = df[df["track_name"] == file].index[0]
                # print(row_num)
                df.at[row_num, "data"] = converted_arr
        df.to_csv("./SpotifyFeatures.csv/", index=False, encoding="utf-8")
        
    def convert_mp3_to_wav():
        for file in g.glob("*.mp3"):
            audioMod.convertWav(file)
            
    def batch_convert():
        audioMod.convert_mp3_to_wav()
        audioMod.updateCSV()
    
    def model_record():
        fileName = input("Enter file name or directory of the mp3 or wav file: ")
        if(fileName[-3:0] == "mp3"):
            audioMod.convertWav(fileName)
            fileName = fileName[::-3] + "wav"
        a = audioMod.toArr(fileName)

        return a