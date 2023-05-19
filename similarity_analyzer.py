import pandas as pd
import numpy as np
import random


df = pd.read_csv("SpotifyFeatures.csv", na_filter = False)

# initialize the minimum and maximum values for normalization
mins = df.min()
maxs = df.max()
extremes = {}

for i in range(len(mins)):
    column = mins.index[i]
    column_min = mins[i] 
    column_max = maxs[i] 
    extremes[column] = (column_min, column_max)

### PARAMETERS AND CONSTANTS ###

# Excluded numerical headers
blacklist = [
    "tempo"
]

# 232725 for this dataset
NUM_ENTRIES = len(df.index)

RANDOM_IND = lambda: random.randint(0,NUM_ENTRIES-1)

print("Loaded CSV")


def normalize_value(value, header_name: str):
    if header_name not in extremes:
        raise KeyError("Invalid Key")
    
    h_extremes = extremes[header_name]

    if (type(h_extremes[0]) == str):
        raise TypeError("Cannot normalize a non-numerical value")

    h_min = h_extremes[0]
    h_max = h_extremes[1]
    result = (value - h_min) / (h_max - h_min)
    assert 0 <= result <= 1, "I messed up"

    return result

# error function
def err(base, other):
    # escapes if same song
    # returns infinity since want to ignore duplicates
    if (base["track_id"] == other["track_id"]):
        return np.inf

    error = 0

    # iterates through columns
    for i in range(len(base)):
        a, b = base[i], other[i]

        # skip if  not numerical value
        if (type(a) == str):
            continue

        # normalizes the values then adds to the error
        header_name = base.index[i]

        a = normalize_value(a, header_name)
        b = normalize_value(b, header_name)
        error += (a - b) ** 2
    out = (error, other["track_id"])
    return out

def get_smallest_error(song):
    # gets error of each song and stores in new DataFrame
    # about 3 times faster than iterating
    errors = df.apply(lambda x: err(song, x), axis=1, result_type='expand')

    min_ind = errors[0].idxmin()
    min_error = errors.min(numeric_only=True)[0]

    return (min_error, min_ind)


# Misc functions

def print_song_by_track_id(track_id):
    print(df[df["track_id"]==track_id].iloc[0])

def print_song_by_index(ind: int):
    print(df.iloc[ind])


