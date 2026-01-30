import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(
    BASE_DIR,
    "RCM_Analytics_10k_Sample_Data.xlsx"
)



# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------
def get_dataset_overview():
    xl = pd.ExcelFile(DATASET_PATH)
    overview = {}

    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        overview[sheet] = {
            "rows": len(df),
            "columns": list(df.columns),
        }

    return overview


# --------------------------------------------------
# JOIN DENIALS + CLAIMS
# --------------------------------------------------
def _joined_denials_claims():
    denials = pd.read_excel(DATASET_PATH, sheet_name="fact_denials")
    claims = pd.read_excel(DATASET_PATH, sheet_name="fact_claim_financials")

    return denials.merge(
        claims,
        on="claim_id",
        how="left",
    )


# --------------------------------------------------
# TOP DENIAL CATEGORIES
# --------------------------------------------------
def get_top_denial_categories(payer: str | None = None):
    df = _joined_denials_claims()

    if payer:
        df = df[
            df["payer_name"]
            .astype(str)
            .str.upper()
            .str.contains(payer.upper(), na=False)
        ]

    grouped = (
        df.groupby("denial_category")["denial_id"]
        .nunique()
        .sort_values(ascending=False)
    )

    return grouped.to_dict()


# --------------------------------------------------
# DENIAL SUMMARY
# --------------------------------------------------
def get_denials_summary():
    df = pd.read_excel(DATASET_PATH, sheet_name="fact_denials")

    return {
        "total_denials": df["denial_id"].nunique(),
        "date_range": {
            "start": str(df["denial_date"].min()),
            "end": str(df["denial_date"].max()),
        },
    }


# --------------------------------------------------
# AUTHORIZATION DENIAL %
# --------------------------------------------------
def get_authorization_denial_percentage():
    df = pd.read_excel(DATASET_PATH, sheet_name="fact_denials")

    total = df["denial_id"].nunique()
    auth = df[df["denial_category"] == "Authorization"]["denial_id"].nunique()

    return {
        "authorization_denials": auth,
        "total_denials": total,
        "authorization_percentage": round(auth / total * 100, 2),
    }


# --------------------------------------------------
# TOTAL DENIED AMOUNT (ROBUST)
# --------------------------------------------------
def get_total_denied_amount():
    df = pd.read_excel(DATASET_PATH, sheet_name="fact_denials")

    df["expected_recovery"] = (
        df["expected_recovery"]
        .fillna(0)
        .astype(float)
    )

    return {
        "total_denied_amount": round(df["expected_recovery"].sum(), 2)
    }


# --------------------------------------------------
# FINANCIAL IMPACT BY DENIAL CATEGORY
# --------------------------------------------------
def get_financial_impact_by_denial_category():
    df = pd.read_excel(DATASET_PATH, sheet_name="fact_denials")

    df["expected_recovery"] = (
        df["expected_recovery"]
        .fillna(0)
        .astype(float)
    )

    grouped = (
        df.groupby("denial_category")["expected_recovery"]
        .sum()
        .sort_values(ascending=False)
    )

    return grouped.to_dict()


# --------------------------------------------------
# PREVENTABLE DENIAL % (CORRECT SEMANTICS)
# --------------------------------------------------
def get_preventable_denial_percentage():
    df = pd.read_excel(DATASET_PATH, sheet_name="fact_denials")

    total = df["denial_id"].nunique()

    preventable = df[
        df["preventable_flag"]
        .astype(str)
        .str.strip()
        .str.upper()
        == "Y"
    ]["denial_id"].nunique()

    return {
        "preventable_denials": preventable,
        "total_denials": total,
        "preventable_percentage": round((preventable / total) * 100, 2),
    }


# --------------------------------------------------
# AR BALANCE BY PAYER
# --------------------------------------------------
def get_ar_balance_by_payer():
    df = pd.read_excel(DATASET_PATH, sheet_name="fact_claim_financials")

    df["ar_balance"] = df["ar_balance"].fillna(0)

    grouped = (
        df.groupby("payer_name")["ar_balance"]
        .sum()
        .sort_values(ascending=False)
    )

    return grouped.to_dict()



# --------------------------------------------------
# APPEALED DENIAL PERCENTAGE
# --------------------------------------------------
def get_appealed_denial_percentage():
    df = pd.read_excel(DATASET_PATH, sheet_name="fact_denials")

    total = df["denial_id"].nunique()

    appealed = df[
        df["appeal_filed"]
        .astype(str)
        .str.strip()
        .str.upper()
        == "Y"
    ]["denial_id"].nunique()

    return {
        "appealed_denials": appealed,
        "total_denials": total,
        "appealed_percentage": round((appealed / total) * 100, 2),
    }


# --------------------------------------------------
# TOTAL RECOVERED AMOUNT FROM APPEALS
# --------------------------------------------------
def get_total_recovered_amount():
    df = pd.read_excel(DATASET_PATH, sheet_name="fact_denials")

    df["recovered_amount"] = (
        df["recovered_amount"]
        .fillna(0)
        .astype(float)
    )

    return {
        "total_recovered_amount": round(df["recovered_amount"].sum(), 2)
    }


# --------------------------------------------------
# TOP DEPARTMENT FOR PREVENTABLE DENIALS
# --------------------------------------------------
def get_top_preventable_department():
    df = pd.read_excel(DATASET_PATH, sheet_name="fact_denials")

    df = df[
        df["preventable_flag"]
        .astype(str)
        .str.strip()
        .str.upper()
        == "Y"
    ]

    grouped = (
        df.groupby("department_at_fault")["denial_id"]
        .nunique()
        .sort_values(ascending=False)
    )

    return grouped.to_dict()


# --------------------------------------------------
# AVERAGE DENIAL AGE (DAYS)
# --------------------------------------------------
def get_average_denial_age():
    df = pd.read_excel(DATASET_PATH, sheet_name="fact_denials")

    df["denial_age_days"] = (
        df["denial_age_days"]
        .fillna(0)
        .astype(float)
    )

    return {
        "average_denial_age_days": round(df["denial_age_days"].mean(), 2)
    }
