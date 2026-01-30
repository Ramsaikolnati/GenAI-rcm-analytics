import pandas as pd

DATASET_PATH = "RCM_Analytics_10k_Sample_Data.xlsx"

def load_dataset_summary():
    """
    Load dataset and return:
    - table names
    - column schemas
    - high-level stats (counts)
    """

    xl = pd.ExcelFile(DATASET_PATH)
    summary = {}

    for sheet in xl.sheet_names:
        df = xl.parse(sheet)

        summary[sheet] = {
            "rows": len(df),
            "columns": list(df.columns),
            "sample_rows": df.head(2).to_dict(orient="records")
        }

    return summary
