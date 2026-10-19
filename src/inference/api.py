import os
import tempfile
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from src.inference.predict import MODEL, predict, ready, model_path
from src.preprocessing.data import TEXT_FIELDS
from src.inference.documents import (
    MAX_UPLOAD_BYTES,
    DocumentUploadError,
    OCRUnavailableError,
    extract_posting_text,
)

# Keep FastAPI's temporary multipart spool inside the user's requested C:\Codex root.
UPLOAD_TEMP_DIR = Path(os.getenv("BRS_TMP_DIR", r"C:\Codex\tmp"))
UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(UPLOAD_TEMP_DIR)

app = FastAPI(title="Bharat Rozgaar Sentinel API", version="1.0.0",
              description="AI-based employment posting risk assessment. Results are not proof of fraud.")
origins = os.getenv("BRS_ALLOWED_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(CORSMiddleware, allow_origins=[origin.strip() for origin in origins],
                   allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])

FRONTEND_ROOT = Path(__file__).resolve().parents[2] / "frontend"
FRONTEND_DIST = FRONTEND_ROOT / "dist"
if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="website-assets")


class JobPosting(BaseModel):
    # User-friendly prototype inputs. These are free text, not dataset columns.
    role_and_location: str = Field(default="", description="Job title, role, and location from the posting")
    company_and_recruiter: str = Field(default="", description="Company background and recruiter/contact details")
    work_and_qualifications: str = Field(default="", description="Responsibilities, work arrangement, and qualifications")
    pay_and_application: str = Field(default="", description="Pay, benefits, and application or interview instructions")

    # Original dataset-shaped fields remain accepted for backwards compatibility.
    title: str = ""
    location: str = ""
    department: str = ""
    salary_range: str = ""
    company_profile: str = ""
    description: str = Field(default="", description="Job description")
    requirements: str = ""
    benefits: str = ""
    telecommuting: str = ""
    has_company_logo: str = ""
    has_questions: str = ""
    employment_type: str = ""
    required_experience: str = ""
    required_education: str = ""
    industry: str = ""
    function: str = ""


def to_model_posting(posting: JobPosting) -> dict:
    """Convert grouped free-text inputs to the text fields used during training.

    Keep the API compatible with existing clients that still submit the original
    dataset-shaped fields. Grouped form text is placed into the corresponding
    trained text fields so the selected NLP model processes its wording.
    """
    payload = posting.model_dump()
    grouped = {
        "role_and_location": payload.pop("role_and_location"),
        "company_and_recruiter": payload.pop("company_and_recruiter"),
        "work_and_qualifications": payload.pop("work_and_qualifications"),
        "pay_and_application": payload.pop("pay_and_application"),
    }

    if not any(value.strip() for value in grouped.values()):
        return payload

    if grouped["role_and_location"].strip():
        payload["title"] = grouped["role_and_location"].strip()
    if grouped["company_and_recruiter"].strip():
        payload["company_profile"] = grouped["company_and_recruiter"].strip()

    description_parts = []
    if payload.get("description", "").strip():
        description_parts.append(payload["description"].strip())
    if grouped["work_and_qualifications"].strip():
        description_parts.append(
            "Work and qualifications: " + grouped["work_and_qualifications"].strip()
        )
    if grouped["pay_and_application"].strip():
        description_parts.append(
            "Pay, benefits, and application details: " + grouped["pay_and_application"].strip()
        )
    if description_parts:
        payload["description"] = "\n\n".join(description_parts)

    return payload


@app.get("/", include_in_schema=False)
def website():
    built_page = FRONTEND_DIST / "index.html"
    page = built_page if built_page.is_file() else FRONTEND_ROOT / "prototype.html"
    return FileResponse(page)


@app.get("/health")
def health():
    return {"status": "ready" if ready() else "model_not_trained",
            "model": MODEL, "artifact_path": str(model_path())}


@app.post("/predict")
def score_posting(posting: JobPosting):
    if not ready():
        raise HTTPException(status_code=503,
                            detail=f"No trained '{MODEL}' model artifact found at {model_path()}. Train it first.")
    payload = to_model_posting(posting)
    if not any(str(payload.get(field) or "").strip() for field in TEXT_FIELDS):
        raise HTTPException(
            status_code=422,
            detail="Enter or upload job-posting text before requesting an assessment.",
        )
    try:
        return predict(payload)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/extract")
async def extract_document(file: UploadFile = File(...)):
    """Extract text from a PDF or screenshot for user review before prediction."""
    filename = Path(file.filename or "upload").name
    try:
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The file is over the 15 MB upload limit.")
    try:
        return await run_in_threadpool(extract_posting_text, filename, contents)
    except OCRUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except DocumentUploadError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

