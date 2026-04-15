import httpx
import json
import re
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq

load_dotenv()
# --- Setup ---
http_client = httpx.Client(verify=False)
client = Groq(
    api_key="os.getenv('GROQ_API_KEY')",
    http_client=http_client
)
app = FastAPI()

# --- Request model ---
class DisputeRequest(BaseModel):
    buyer_claim: str
    seller_claim: str
    amount: float
    job_description: str

# --- Parsing helper ---
def parse_verdict(raw: str) -> dict:
    result = {"verdict": None, "split_percentage": None, "reasoning": None, "raw": raw}
    
    verdict_match = re.search(r"VERDICT:\s*(RELEASE_TO_SELLER|REFUND_BUYER|PARTIAL_SPLIT)", raw)
    if verdict_match:
        result["verdict"] = verdict_match.group(1)
    
    split_match = re.search(r"SPLIT_PERCENTAGE:\s*(.+)", raw)
    if split_match:
        result["split_percentage"] = split_match.group(1).strip()
    
    reasoning_match = re.search(r"REASONING:\s*(.+)", raw, re.DOTALL)
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()
    
    return result

# --- Endpoint ---
@app.post("/resolve-dispute")
def resolve_dispute(request: DisputeRequest):
    prompt = f"""
You are an AI dispute resolution engine for EscrowPay, a Nigerian freelance escrow platform.

A payment of ₦{request.amount} is held in escrow for the following job:
Job Description: {request.job_description}

The buyer claims: {request.buyer_claim}
The seller claims: {request.seller_claim}

Analyze both sides fairly and return your verdict as follows:
- VERDICT: one of [RELEASE_TO_SELLER | REFUND_BUYER | PARTIAL_SPLIT]
- SPLIT_PERCENTAGE: (only if PARTIAL_SPLIT) e.g. 60% seller, 40% buyer
- REASONING: 2-3 sentences explaining your decision

Be concise and impartial.
"""
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    return parse_verdict(response.choices[0].message.content)


#---Trsutscore Feature ---
class TrustScoreRequest(BaseModel):
    total_transactions: int
    completed_successfully: int
    disputes_raised_against_user: int
    disputes_lost_by_user: int

@app.post("/calculate-trust-score")
def calculate_trust_score(request: TrustScoreRequest):
    if request.total_transactions == 0:
        return {"trust_score": 0, "reasoning": "No transactions yet."}
    
    success_rate = request.completed_successfully / request.total_transactions
    dispute_rate = request.disputes_raised_against_user / request.total_transactions
    loss_rate = request.disputes_lost_by_user / request.total_transactions
    
    trust_score = (success_rate * 100) - (dispute_rate * 50) - (loss_rate * 50)
    trust_score = max(0, min(100, trust_score))  # Clamp between 0 and 100
    
    reasoning = f"Success Rate: {success_rate:.2%}, Dispute Rate: {dispute_rate:.2%}, Loss Rate: {loss_rate:.2%}."
    
    if trust_score > 80:
        reasoning += " User is highly trustworthy."
    elif trust_score > 50:
        reasoning += " User is moderately trustworthy."
    elif trust_score > 20:
        reasoning += " User is somewhat trustworthy."
    else:        
        reasoning += " User is not trustworthy."
    return {"trust_score": round(trust_score, 2), "reasoning": reasoning}

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "EscrowPay ML API"}