import pandas as pd

df = pd.read_csv("D:/AGE_EMOTION_WITH_VOICE/cv-corpus-22.0-delta-2025-06-20/en/validated.tsv", sep="\t")
print("Columns:", df.columns)
print(df['gender'].value_counts(dropna=False))
print(df[['gender', 'path']].dropna().head(10))
