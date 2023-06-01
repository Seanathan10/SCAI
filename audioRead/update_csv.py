import pandas as pd
from audRead import audioMod as ar
import glob2 as g
import numpy as np

df = pd.read_csv("SpotifyFeatures.csv")

df["data"] = "0"
np.set_printoptions(linewidth=np.inf)

for file in g.glob("*.wav"):
    wav_data = ar.toArr(file, clean=True)
    converted_arr = np.array2string(wav_data, precision=0, separator=',')
    
    file = (file.replace("_", " ")[:-4]).lower()
    # print(file)
    row_num = df[df['track_name'].apply(lambda x: x.lower()) == file].index[0]
    # print(row_num)-
    
    df.at[row_num, "data"] = converted_arr
df.to_csv("SpotifyFeatures.csv", index=False)