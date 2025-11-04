from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from pathlib import Path
from typing import Optional

# Text extraction
import docx2txt
from pdfminer.high_level import extract_text as pdf_extract_text

# Our universal engine
from universal_authenticity_engine import UniversalAuthenticityEngine

app = FastAPI(
    title="ResumeGuard V2 CALIBRATED",
    version="2.0-calibrated-universal",
    description="Universal cross-industry resume authenticity scoring"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key (set via environment variable)
API_KEY = os.getenv("RG_API_KEY", "changeme123!!")


def extract_text_from_file(path: Path) -> str:
    """Extract text from PDF, DOCX, or TXT files"""
    try:
        if path.suffix.lower() == ".pdf":
            return pdf_extract_text(str(path))
        elif path.suffix.lower() == ".docx":
            return docx2txt.process(str(path))
        else:
            return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        raise ValueError(f"Could not extract text from {path.name}: {str(e)}")


def convert_to_legacy_format(result) -> dict:
    """
    Convert new Universal format to legacy V2 format
    This ensures frontend compatibility with no changes needed
    """
    
    # Map star rating to trust_score (0-100)
    # Use raw_score directly for trust_score
    trust_score = result.raw_score
    
    # Map tier to screening_tier
    tier_mapping = {
        "Elite": "Elite Tier",
        "Excellent": "Fast-Track Tier",
        "Strong": "Standard Tier",
        "Fair": "Validation Tier",
        "Weak": "High Risk Tier",
        "High Risk": "Rejection Tier"
    }
    
    # Determine recommended_action based on score
    if trust_score >= 95:
        recommended_action = "Skip to final interview / Direct to hiring manager"
    elif trust_score >= 85:
        recommended_action = "Skip phone screen → Technical interview"
    elif trust_score >= 75:
        recommended_action = "Normal interview process"
    elif trust_score >= 60:
        recommended_action = "Technical assessment before phone screen"
    elif trust_score >= 45:
        recommended_action = "Only proceed if desperate + rigorous vetting"
    else:
        recommended_action = "Reject unless critical need + extreme vetting"
    
    # Build legacy response
    legacy_response = {
        "name": result.candidate,
        "trust_score": trust_score,
        "verdict": result.verdict,
        "recommended_action": recommended_action,
        "red_flags": result.red_flags,
        "authenticity_markers": result.authenticity_markers,
        "has_problem_specificity": result.has_real_problems,
        "does_unsexy_work": len(result.authenticity_markers) > 0,  # Approximation
        "screening_tier": tier_mapping.get(result.tier, "Standard Tier"),
        "confidence": result.industry_confidence if result.industry_detected else 0.70,
        
        # Add new fields (backwards compatible - frontend can ignore if not used)
        "star_rating": result.star_rating,
        "star_display": result.star_display,
        "industry_detected": result.industry_detected,
        "ai_probability": result.ai_probability,
        "universal_questions": result.universal_questions,
        "industry_specific_question": result.industry_specific_question
    }
    
    return legacy_response


@app.get("/")
def root():
    return {
        "service": "ResumeGuard V2 CALIBRATED - Universal Edition",
        "version": "2.0-calibrated-universal",
        "description": "Cross-industry resume authenticity scoring (backwards compatible)",
        "calibration": "Starts at 30, stars remapped for recruiter psychology",
        "industries_supported": [
            "Technology/DevOps",
            "Sales",
            "Marketing",
            "Finance",
            "Healthcare",
            "Operations",
            "HR"
        ]
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score_auto")
async def score_auto(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None)
):
    """
    Upload a resume and get V2 CALIBRATED authenticity analysis
    NOW WITH UNIVERSAL INDUSTRY SUPPORT (backwards compatible response)
    """
    
    # API key validation
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Forbidden: Invalid or missing API key"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)
        
        # Extract text
        text = extract_text_from_file(tmp_path)
        
        # Clean up temp file
        os.remove(tmp_path)
        
        # Score with universal engine
        engine = UniversalAuthenticityEngine()
        result = engine.score_resume(
            resume_text=text,
            candidate_name=file.filename
        )
        
        # Convert to legacy format for frontend compatibility
        legacy_response = convert_to_legacy_format(result)
        
        # Return as JSON
        return JSONResponse(legacy_response)
    
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
