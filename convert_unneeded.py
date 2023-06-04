import os
import re

for filename in os.listdir():
    if filename == "converter.py":
        continue
    if re.search("[^a-zA-Z0-9_.]+", filename):
        newfile = re.sub("[^a-zA-Z0-9_.]+", "_", filename)
        os.rename(filename, newfile)
