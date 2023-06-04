import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


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

print(extremes)

### PARAMETERS AND CONSTANTS ###
WEIGHTS = {
    "genre": 0.1,           # string
    "artist_name": 0,       # string
    "track_name": 0,        # string
    "track_id": 0,          # string
    "popularity": 1,        # Z \in [0, 100]
    "acousticness": 1,      # R \in [0, 1]
    "danceability": 1,      # R \in [0, 1]
    "duration_ms": 0,       # Z \in [15387, 5552917]
    "energy": 1,            # R \in [0, 1]
    "instrumentalness": 1,  # R \in [0, 1]
    "key": 0,               # string
    "liveness": 1,          # R \in [0, 1]
    "loudness": 1,          # R \in [-52.457, 3.744]
    "mode": 0.1,            # string (either major or minor)
    "speechiness": 1,       # R \in [0, 1]
    "tempo": 0,             # R \in [30, 243] buggy, don't use
    "time_signature": 0.1,  # string \in {0/4, 1/4, 3/4, 4/4, 5/4}
    "valence": 1            # R \in [0, 1]
}

MULTIPLIER = 2

# 232725 for this dataset
NUM_ENTRIES = len(df.index)
NUM_COLUMNS = len(df.columns)

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
def error(base, other, random_nudge=0, weight_multipliers=None):
    if weight_multipliers==None:
        weight_multipliers = [1] * NUM_COLUMNS

    # escapes if same song
    # returns infinity since want to ignore duplicates
    if (base["track_id"] == other["track_id"]):
        return np.inf

    error = 0

    # iterates through columns
    for i in range(len(base)):
        a, b = base[i], other[i]

        header_name = base.index[i]

        if (type(a) == str):
            error += ((0 if a==b else 1) * WEIGHTS[header_name] * weight_multipliers[i]) ** 2
        else:
            a = normalize_value(a, header_name)
            b = normalize_value(b, header_name)
            error += ((a - b) * WEIGHTS[header_name] * weight_multipliers[i]) ** 2
    error += np.random.uniform(0, random_nudge)
    out = (error, other["track_id"])
    return out

def get_smallest_error(song):

    # can select one of these at random
    def base_error(other):
        return error(song, other)
    
    def random_multiples(other):
        weight_multiples = [1] * NUM_COLUMNS
        num_weighted = np.random.poisson(2)
        for i in range(num_weighted):
            weight_multiples[np.random.randint(0, NUM_COLUMNS)] = MULTIPLIER
        
        return error(song, other, 0, weight_multiples)

    def random_nudge(other):
        return error(song, other, 0.02)
    

    # change the application function as one sees fit
    errors = df.sample(10000).apply(base_error, axis=1, result_type='expand')

    min_ind = errors[0].idxmin()
    min_error = errors.min(numeric_only=True)[0]
    return (min_error, min_ind)


def print_song_by_track_id(track_id):
    print(df[df["track_id"]==track_id].iloc[0])

def print_song_by_index(ind: int):
    print(df.iloc[ind])

def get_random_song_and_nearest():
    song_ind = RANDOM_IND()
    song = df.iloc[song_ind]
    print(song)
    err, closest = get_smallest_error(song)
    print_song_by_index(closest)
    print(err)

def generate_random_playlist(num_songs: int, initial=-1):
    if (initial == -1):
        initial = RANDOM_IND()

    out = []
    song = df.iloc[initial]
    out.append(song)
    for i in range(num_songs):
        err, ind = get_smallest_error(song)
        song = df.iloc[ind]
        out.append(song)

    return out

# takes in list of rows
def display_playlist(playlist):
    for song in playlist:
        print("{0:50}{1:30}{2:30}".format(song["track_name"], song["artist_name"], song["genre"]))

def plot_error():
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows,cols)

    for i in range(rows):
        for j in range(cols):
            song = df.iloc[RANDOM_IND()]
            errors = df.sample(10000).apply(lambda other: error(song, other), axis=1, result_type='expand')
            errors[0].plot.hist(column=[0], bins=50, ax=axes[i][j], xlim=(0,5))
            genre = song["genre"]
            axes[i][j].set_title(f"{genre}")
    
    plt.show()