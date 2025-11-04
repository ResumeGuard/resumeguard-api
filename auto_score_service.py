from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import Optional, List
import tempfile
import os
import io
import csv
from pathlib import Path
from datetime import datetime

# Text extraction
import docx2txt
from pdfminer.high_level import extract_text as pdf_extract_text

# Our modules
from universal_authenticity_engine import UniversalAuthenticityEngine
from database import Analysis, get_db, init_db

app = FastAPI(
    title="ResumeGuard V3 - Production",
    version="3.0-with-storage",
    description="Universal resume authenticity scoring with bulk upload and reporting"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()

# API Key
API_KEY = os.getenv("RG_API_KEY", "changeme123!!")
DEFAULT_USER_ID = "default_user"  # For MVP - replace with real auth later


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


def format_ai_probability(ai_prob: float) -> str:
    """Convert numeric AI probability to frontend-friendly string"""
    if ai_prob >= 0.50:
        return "Very High (50%+)"
    elif ai_prob >= 0.30:
        return "High (30-50%)"
    elif ai_prob >= 0.15:
        return "Moderate (15-30%)"
    elif ai_prob >= 0.05:
        return "Low (5-15%)"
    else:
        return "Very Low (<5%)"


def convert_to_legacy_format(result) -> dict:
    """Convert engine result to API response format"""
    
    trust_score = result.raw_score
    
    tier_mapping = {
        "Elite": "Elite Tier",
        "Excellent": "Fast-Track Tier",
        "Strong": "Standard Tier",
        "Fair": "Validation Tier",
        "Weak": "High Risk Tier",
        "High Risk": "Rejection Tier"
    }
    
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
    
    return {
        "name": result.candidate,
        "trust_score": trust_score,
        "verdict": result.verdict,
        "recommended_action": recommended_action,
        "red_flags": result.red_flags,
        "authenticity_markers": result.authenticity_markers,
        "has_problem_specificity": result.has_real_problems,
        "does_unsexy_work": len(result.authenticity_markers) > 0,
        "screening_tier": tier_mapping.get(result.tier, "Standard Tier"),
        "confidence": result.industry_confidence if result.industry_detected else 0.70,
        "star_rating": result.star_rating,
        "star_display": result.star_display,
        "industry_detected": result.industry_detected,
        "ai_probability": format_ai_probability(result.ai_probability),
        "ai_probability_raw": result.ai_probability,
        "universal_questions": result.universal_questions,
        "industry_specific_question": result.industry_specific_question
    }


def save_analysis(
    db: Session,
    result,
    resume_text: str,
    filename: str,
    user_id: str = DEFAULT_USER_ID
) -> Analysis:
    """Save analysis to database"""
    
    analysis = Analysis(
        user_id=user_id,
        candidate_name=result.candidate,
        filename=filename,
        resume_text=resume_text,
        trust_score=result.raw_score,
        star_rating=result.star_rating,
        tier=result.tier,
        verdict=result.verdict,
        recommended_action=convert_to_legacy_format(result)["recommended_action"],
        industry=result.industry_detected,
        industry_confidence=result.industry_confidence,
        ai_probability=result.ai_probability,
        ai_probability_display=format_ai_probability(result.ai_probability),
        problems_count=len([m for m in result.authenticity_markers if m]),
        authenticity_count=len(result.authenticity_markers),
        career_growth=0,
        has_long_tenure=0,
        word_count=len(resume_text.split()),
        red_flags=result.red_flags,
        authenticity_markers=result.authenticity_markers,
        universal_questions=result.universal_questions
    )
    
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    
    return analysis


@app.get("/")
def root():
    return {
        "service": "ResumeGuard V3 - Production",
        "version": "3.0-with-storage",
        "description": "Resume authenticity scoring with bulk upload and reporting",
        "features": [
            "Single resume analysis",
            "Bulk upload (up to 50 resumes)",
            "Analysis history & reporting",
            "Export to CSV/JSON",
            "Cross-industry support"
        ]
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": "3.0"}


@app.post("/score_auto")
async def score_auto(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """Upload single resume - NOW WITH DATABASE STORAGE"""
    
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)
        
        text = extract_text_from_file(tmp_path)
        os.remove(tmp_path)
        
        engine = UniversalAuthenticityEngine()
        result = engine.score_resume(resume_text=text, candidate_name=file.filename)
        
        analysis = save_analysis(db, result, text, file.filename)
        
        legacy_response = convert_to_legacy_format(result)
        legacy_response["analysis_id"] = analysis.id
        legacy_response["analyzed_at"] = analysis.analyzed_at.isoformat()
        
        return JSONResponse(legacy_response)
    
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/score_bulk")
async def score_bulk(
    files: List[UploadFile] = File(...),
    x_api_key: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """Upload up to 50 resumes and get batch analysis"""
    
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 files per batch")
    
    results = []
    errors = []
    engine = UniversalAuthenticityEngine()
    
    for file in files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = Path(tmp.name)
            
            text = extract_text_from_file(tmp_path)
            os.remove(tmp_path)
            
            result = engine.score_resume(resume_text=text, candidate_name=file.filename)
            analysis = save_analysis(db, result, text, file.filename)
            
            results.append({
                "filename": file.filename,
                "analysis_id": analysis.id,
                "trust_score": result.raw_score,
                "star_rating": result.star_rating,
                "star_display": result.star_display,
                "tier": result.tier,
                "verdict": result.verdict,
                "status": "success"
            })
            
        except Exception as e:
            errors.append({
                "filename": file.filename,
                "error": str(e),
                "status": "failed"
            })
    
    return JSONResponse({
        "batch_id": f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        "total_files": len(files),
        "successful": len(results),
        "failed": len(errors),
        "results": results,
        "errors": errors
    })


@app.get("/analyses/history")
async def get_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    min_score: Optional[float] = Query(None, ge=0, le=100),
    max_score: Optional[float] = Query(None, ge=0, le=100),
    tier: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    x_api_key: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """Get paginated history with filtering"""
    
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    
    query = db.query(Analysis).filter(Analysis.user_id == DEFAULT_USER_ID)
    
    if min_score is not None:
        query = query.filter(Analysis.trust_score >= min_score)
    if max_score is not None:
        query = query.filter(Analysis.trust_score <= max_score)
    if tier:
        query = query.filter(Analysis.tier == tier)
    if start_date:
        query = query.filter(Analysis.analyzed_at >= datetime.fromisoformat(start_date))
    if end_date:
        query = query.filter(Analysis.analyzed_at <= datetime.fromisoformat(end_date))
    
    total = query.count()
    analyses = query.order_by(Analysis.analyzed_at.desc()).offset(offset).limit(limit).all()
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "results": [analysis.to_dict() for analysis in analyses]
    }


@app.get("/analyses/{analysis_id}")
async def get_analysis(
    analysis_id: str,
    x_api_key: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """Retrieve specific analysis by ID"""
    
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis.to_dict()


@app.post("/analyses/export")
async def export_analyses(
    format: str = Query("csv", regex="^(csv|json)$"),
    analysis_ids: Optional[List[str]] = None,
    x_api_key: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """Export analyses to CSV or JSON"""
    
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    
    if analysis_ids:
        analyses = db.query(Analysis).filter(Analysis.id.in_(analysis_ids)).all()
    else:
        analyses = db.query(Analysis).filter(Analysis.user_id == DEFAULT_USER_ID).order_by(Analysis.analyzed_at.desc()).all()
    
    if format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        
        writer.writerow([
            "Candidate", "Score", "Stars", "Tier", "Verdict", 
            "Industry", "AI Probability", "Problems", "Authenticity Markers",
            "Red Flags", "Analyzed At"
        ])
        
        for analysis in analyses:
            writer.writerow([
                analysis.candidate_name,
                analysis.trust_score,
                analysis.star_rating,
                analysis.tier,
                analysis.verdict,
                analysis.industry or "Unknown",
                analysis.ai_probability_display,
                analysis.problems_count,
                len(analysis.authenticity_markers) if analysis.authenticity_markers else 0,
                len(analysis.red_flags) if analysis.red_flags else 0,
                analysis.analyzed_at.isoformat()
            ])
        
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=resumeguard_export_{datetime.utcnow().strftime('%Y%m%d')}.csv"}
        )
    
    else:
        return {
            "exported_at": datetime.utcnow().isoformat(),
            "total_analyses": len(analyses),
            "analyses": [analysis.to_dict() for analysis in analyses]
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
