import ast
import pandas as pd

df = pd.read_csv("288_data.csv")

# Convert the string representation of a list into a list
df['data'] = df['data'].apply(ast.literal_eval)

for i in range(0, len(df['data'])):
    print( df['data'][i] )
    print( i )


