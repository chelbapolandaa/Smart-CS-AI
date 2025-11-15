import pandas as pd

# Load dataset dari CSV
df = pd.read_csv('smart_cs_dataset.csv')

print(f"Total data: {len(df)}")
print("\nDistribusi intent:")
print(df['intent'].value_counts())

# Tampilkan sample data
print("\nSample data:")
print(df.head(10))