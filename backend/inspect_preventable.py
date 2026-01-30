import pandas as pd

df = pd.read_excel(
    "RCM_Analytics_10k_Sample_Data.xlsx",
    sheet_name="fact_denials"
)

print("Column names:")
print(df.columns.tolist())

print("\nUnique values in preventable_flag:")
print(df["preventable_flag"].dropna().unique())

print("\nValue counts:")
print(df["preventable_flag"].value_counts())
