import pandas as pd

df = pd.read_csv("pred.csv")

print(df["classification"].value_counts())
