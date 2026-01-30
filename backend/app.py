import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

from backend.analytics_engine import (
    get_dataset_overview,
    get_top_denial_categories,
    get_denials_summary,
    get_authorization_denial_percentage,
    get_total_denied_amount,
    get_financial_impact_by_denial_category,
    get_preventable_denial_percentage,
    get_ar_balance_by_payer,
)

from backend.prompt_builder import build_prompt


# ---------------------------------------
# SETUP
# ---------------------------------------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="GenAI RCM Analytics Demo")

STATIC_DIR = "static"

# ---------------------------------------
# REQUEST MODEL
# ---------------------------------------
class QueryRequest(BaseModel):
    question: str


# ---------------------------------------
# API ENDPOINT
# ---------------------------------------
@app.post("/genai/analytics/query")
def query_analytics(req: QueryRequest):
    q = req.question.lower()

    # -------------------------------
    # DATA ROUTING
    # -------------------------------
    if "ar balance" in q:
        result = get_ar_balance_by_payer()

    elif "denied amount" in q or "denial amount" in q:
        result = get_total_denied_amount()

    elif "financial impact" in q and "denial" in q:
        result = get_financial_impact_by_denial_category()

    elif "preventable" in q and "denial" in q:
        result = get_preventable_denial_percentage()

    elif "percentage" in q and "authorization" in q:
        result = get_authorization_denial_percentage()

    elif "top" in q and "denial" in q:
        payer = "BCBS" if "bcbs" in q else None
        result = get_top_denial_categories(payer)

    elif "how many" in q and "denial" in q:
        result = get_denials_summary()

    elif "dataset" in q:
        result = get_dataset_overview()

    # -------------------------------
    # GENERAL (NON-DATA) QUESTION
    # -------------------------------
    else:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "Answer briefly and clearly. Do not mention datasets.",
                },
                {
                    "role": "user",
                    "content": req.question,
                },
            ],
            temperature=0,
        )

        return {
            "answer": response.choices[0].message.content.strip()
        }

    # -------------------------------
    # DATA + LLM EXPLANATION
    # -------------------------------
    messages = build_prompt(req.question, result)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0,
    )

    return {
        "answer": response.choices[0].message.content.strip(),
        "data": result,
    }


# ---------------------------------------
# STATIC FILES (SAFE)
# ---------------------------------------
app.mount("/assets", StaticFiles(directory="static/assets"), name="assets")

@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)