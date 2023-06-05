import pandas as pd
import numpy as np



WEIGHTS = [
0.5,                #popularity
1,                  #acousticness
1,                  #danceability
1,                  #energy
1,                  #instrumentalness
1,                  #liveness
1,                  #loudness
1,                  #speechiness
0,                  #tempo
1                   #valence
]


# base and other are both 10-element float arrays in the range [0,10]
def error(base, other, weight_multipliers=None):
    if weight_multipliers==None:
        weight_multipliers = [1] * 10

    error = 0
    for i in range(10):
        error += ((base[i] - other[i]) * weight_multipliers[i]) ** 2

    return error


# pass in characteristics of base song and the song dictionary
# returns list of tuples in (song, error) form
def get_n_smallest_errors(base_song, song_dict, n):
    assert (n < len(song_dict))

    out = []

    for song in song_dict:
        out.append((song, error(base_song, song_dict[song])))

    out.sort(key= lambda x: x[1], reverse=True)

    # skips the base song (which will have zero error)
    return out[1:n+1]

# given list of tuples of (song, error), get cumulative error
def get_cumulative_error(song_list):
    return sum(j for _, j in song_list)

# given list of tuples of (song, error), isolate songs
def get_playlist(song_list):
    return [song for song, err in song_list]


# input: dict of {song name: song characteristics array}
def build_playlist(song_dict, playlist_length):
    
    # iterates through songs to find one with min error
    best_song = None
    best_error = np.inf
    for song in song_dict:
        song_data = song_dict[song]
        error_list = get_n_smallest_errors(song_data, song_dict, playlist_length)
        cumulative_error = get_cumulative_error(error_list)
        if (cumulative_error < best_error):
            best_error = cumulative_error
            best_song = song
    
    smallest_error_list = get_n_smallest_errors(song_dict[best_song], song_dict, playlist_length)
    best_playlist = get_playlist(smallest_error_list)

    return best_playlist
