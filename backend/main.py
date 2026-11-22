import io
import json
import os
import re
import time
from datetime import datetime
from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    Groq = None
    GROQ_AVAILABLE = False

try:
    from jose import jwt
    JOSE_AVAILABLE = True
except ImportError:
    jwt = None
    JOSE_AVAILABLE = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ── Config (Render > Environment me set karo) ────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_ENV = os.getenv("GROQ_MODEL", "")  # optional override
# Comma-separated allowed origins, e.g. "https://myapp.vercel.app,http://localhost:3000"
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

AI_READY = bool(GROQ_API_KEY and GROQ_AVAILABLE)

if AI_READY:
    print("✅ Groq ready")
else:
    print("⚠️  GROQ_API_KEY not set ya groq package missing")
if not SECRET_KEY:
    print("⚠️  SECRET_KEY not set — login disabled. Render me SECRET_KEY add karo.")

app = FastAPI(title="FinAdvisor API", version="5.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)


# ── Groq helpers ─────────────────────────────────────────────
CANDIDATE_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
]


@lru_cache(maxsize=1)
def get_groq_model() -> str:
    """Ek baar model detect karke cache kar leta hai (har request pe API call nahi)."""
    if GROQ_MODEL_ENV:
        return GROQ_MODEL_ENV
    if not AI_READY:
        return CANDIDATE_MODELS[0]
    try:
        client = Groq(api_key=GROQ_API_KEY)
        available = {m.id for m in client.models.list().data}
        for model in CANDIDATE_MODELS:
            if model in available:
                print(f"✅ Using Groq model: {model}")
                return model
    except Exception as e:
        print(f"⚠️  Could not list Groq models: {e}")
    return CANDIDATE_MODELS[0]


def get_client() -> "Groq":
    return Groq(api_key=GROQ_API_KEY)


# ── Pydantic models ──────────────────────────────────────────
class LoginRequest(BaseModel):
    email: str
    password: str


class ChatRequest(BaseModel):
    message: str
    history: list = Field(default_factory=list)
    user_data: dict = Field(default_factory=dict)


class ReportRequest(BaseModel):
    net_worth: float = 4250000
    gross_salary: float = 1200000
    tax_saved: float = 46500
    monthly_savings: float = 38500


# ── Routes ───────────────────────────────────────────────────
@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {
        "status": "ok",
        "message": "FinAdvisor API v5.1 — Powered by Groq (LLaMA)",
        "ai_ready": AI_READY,
    }


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "groq_configured": AI_READY,
        "pymupdf_available": fitz is not None,
        "reportlab_available": REPORTLAB_AVAILABLE,
        "auth_configured": bool(SECRET_KEY and JOSE_AVAILABLE),
    }


@app.post("/auth/login")
def login(req: LoginRequest):
    # NOTE: Abhi password verify nahi hota (demo login). Real users ke liye
    # database + password hashing (bcrypt/passlib) add karna zaroori hai.
    if not req.email.strip() or not req.password:
        raise HTTPException(status_code=400, detail="Email and password required")
    if not SECRET_KEY or not JOSE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Auth not configured (SECRET_KEY / python-jose)")
    now = int(time.time())
    token = jwt.encode(
        {"sub": req.email, "iat": now, "exp": now + 86400},
        SECRET_KEY,
        algorithm="HS256",
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"email": req.email, "name": req.email.split("@")[0].title()},
    }


FORM16_KEYS = ["gross_salary", "tax_deducted", "investments_80c", "hra_exemption", "net_taxable_income"]

FORM16_FALLBACK = {
    "gross_salary": 1200000, "tax_deducted": 120000,
    "investments_80c": 150000, "hra_exemption": 60000,
    "net_taxable_income": 990000,
}


def _to_number(v) -> float:
    try:
        if isinstance(v, str):
            v = re.sub(r"[^\d.\-]", "", v)
        return float(v) if v not in ("", None) else 0
    except (ValueError, TypeError):
        return 0


@app.post("/parse-form16")
def parse_form16(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    contents = file.file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File must be under 10MB")
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    def fallback(note: str):
        return {"status": "success", "estimated": True, "data": FORM16_FALLBACK, "note": note}

    if fitz is None:
        return fallback("PyMuPDF unavailable")

    extracted_text = ""
    try:
        doc = fitz.open(stream=contents, filetype="pdf")
        for page in doc:
            extracted_text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"PDF error: {e}")
        raise HTTPException(status_code=400, detail="Could not read this PDF")

    if not extracted_text.strip():
        return fallback("No text found in PDF (scanned image?) — using estimates")
    if not AI_READY:
        return fallback("GROQ_API_KEY not set")

    prompt = f"""Extract financial data from this Indian Form 16 document.
Return ONLY a valid JSON object with these exact keys (numeric values only, no commas, no currency symbols):
{{
  "gross_salary": <number>,
  "tax_deducted": <number>,
  "investments_80c": <number>,
  "hra_exemption": <number>,
  "net_taxable_income": <number>
}}
If a value is not found, use 0.

Document text:
{extracted_text[:12000]}"""

    try:
        response = get_client().chat.completions.create(
            model=get_groq_model(),
            messages=[
                {"role": "system", "content": "You are a financial data extractor. Return only valid JSON, nothing else."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=300,
        )
        raw = (response.choices[0].message.content or "").strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)  # markdown/extra text hata deta hai
        if not match:
            return fallback("AI response not JSON, using estimates")
        parsed = json.loads(match.group(0))
        data = {k: _to_number(parsed.get(k, 0)) for k in FORM16_KEYS}
        return {"status": "success", "estimated": False, "data": data}
    except json.JSONDecodeError:
        return fallback("JSON parse failed, using estimates")
    except Exception as e:
        print(f"Groq error in parse-form16: {e}")
        return fallback("AI service error, using estimates")


SYSTEM_PROMPT = """You are FinBot, an expert Indian financial advisor.
Give clear, concise, actionable advice on:
- Indian Income Tax (IT Act 1961), TDS, ITR filing
- Section 80C, 80D, 80CCD, HRA, LTA deductions
- Mutual funds, SIP, ELSS, PPF, NPS, EPF
- Real estate, REIT, wealth management, retirement planning
Keep responses under 250 words. Use bullet points. Use Rs. for amounts.
Respond in the same language the user writes in (Hindi or English)."""


@app.post("/chat")
def chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if not AI_READY:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY not set in Render Environment")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in req.history[-10:]:
        if not isinstance(h, dict):
            continue
        role, content = h.get("role", ""), h.get("content", "")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content[:4000]})

    user_msg = req.message[:4000]
    if req.user_data:
        user_msg = f"My financial profile: {json.dumps(req.user_data)[:2000]}\n\nQuestion: {user_msg}"
    messages.append({"role": "user", "content": user_msg})

    try:
        response = get_client().chat.completions.create(
            model=get_groq_model(),
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        return {"response": response.choices[0].message.content, "status": "success"}
    except Exception as e:
        err = str(e).lower()
        print(f"Chat error: {e}")
        if "rate_limit" in err or "429" in err:
            raise HTTPException(status_code=429, detail="Rate limit hit. Wait a moment and try again.")
        if "invalid api key" in err or "401" in err or "authentication" in err:
            raise HTTPException(status_code=401, detail="Invalid GROQ_API_KEY — check Render environment variables")
        raise HTTPException(status_code=500, detail="Chat service error. Please try again.")


@app.post("/generate-report")
def generate_report(data: ReportRequest):
    if not REPORTLAB_AVAILABLE:
        raise HTTPException(status_code=503, detail="ReportLab not installed")
    try:
        buffer = io.BytesIO()  # disk pe file nahi banti, cleanup ki zaroorat nahi
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                                rightMargin=60, leftMargin=60,
                                topMargin=60, bottomMargin=60)
        styles = getSampleStyleSheet()
        story = [
            Paragraph("FinAdvisor AI — Financial Report", styles["Title"]),
            Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", styles["Normal"]),
            Spacer(1, 24),
            Paragraph("Financial Summary", styles["Heading2"]),
            Spacer(1, 8),
        ]

        savings_rate = (
            f"{(data.monthly_savings / (data.gross_salary / 12) * 100):.1f}%"
            if data.gross_salary > 0 else "N/A"
        )
        table_data = [
            ["Metric", "Value"],
            ["Net Worth", f"Rs.{data.net_worth:,.0f}"],
            ["Annual Income", f"Rs.{data.gross_salary:,.0f}"],
            ["Tax Saved (80C+HRA)", f"Rs.{data.tax_saved:,.0f}"],
            ["Monthly Savings", f"Rs.{data.monthly_savings:,.0f}"],
            ["Savings Rate", savings_rate],
        ]
        t = Table(table_data, colWidths=[260, 180])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F62FE")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 11),
            ("ALIGN", (1, 0), (1, -1), "RIGHT"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F4FF")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ]))
        story += [t, Spacer(1, 24), Paragraph("AI Recommendations", styles["Heading2"]), Spacer(1, 8)]

        recs = [
            ("Tax Optimization", "Maximize Section 80C up to Rs.1.5L via ELSS mutual funds."),
            ("NPS Boost", "Invest Rs.50,000 in NPS for extra deduction under 80CCD(1B)."),
            ("Emergency Fund", "Keep 6 months expenses in liquid fund or sweep-in FD."),
            ("Portfolio Mix", "Follow 60:30:10 — Equity, Debt, Gold allocation."),
            ("Health Insurance", "Ensure Rs.10L+ cover. Premiums deductible u/s 80D."),
        ]
        for title, text in recs:
            story += [Paragraph(f"<b>{title}:</b> {text}", styles["Normal"]), Spacer(1, 8)]

        story += [
            Spacer(1, 16),
            Paragraph("<i>Disclaimer: For informational purposes only. "
                      "Consult a SEBI-registered advisor.</i>", styles["Normal"]),
        ]
        doc.build(story)

        return Response(
            content=buffer.getvalue(),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=FinAdvisor_Report.pdf"},
        )
    except Exception as e:
        print(f"Report error: {e}")
        raise HTTPException(status_code=500, detail="Could not generate report")


if __name__ == "__main__":
    # Sirf local run ke liye. Render pe start command use hoga (neeche dekho).
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
