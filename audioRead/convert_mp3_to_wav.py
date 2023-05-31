from audRead import audioMod
import glob2 as g

for file in g.glob("*.mp3"):
    audioMod.convertWav(file)