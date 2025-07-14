import pandas as pd

sums_file_path = "AML-Cytomorphology.sums"

df = pd.read_csv(sums_file_path, sep=" ", header=None, names=["checksum", "filepath"])

print("Head:")
print(df.head())

print("\nInfo:")
print(df.info())

print("\nDescribe:")
print(df.describe(include='all'))  


