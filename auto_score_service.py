from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
from statistics import mean

# --- Optional parsers ---
import docx2txt
from pdfminer.high_level import extract_text as pdf_extract_text

app = FastAPI(title="ResumeGuard Auto-Scorer API", version="1.0")

# ======================================================================
# CORE SCORER (UnifiedResumeScorerV2)
# ======================================================================

@dataclass
class EvaluationResult:
    candidate: str
    legitimacy_score: int
    authenticity: int
    technical_fit: int
    context_alignment: int
    noise_control: int
    timeline_realism: int
    strengths: List[str]
    red_flags: List[str]
    verdict: str


class UnifiedResumeScorerV2:
    def __init__(self, candidate_name: str):
        self.name = candidate_name
        self.category_scores: Dict[str, int] = {}
        self.strengths: List[str] = []
        self.red_flags: List[str] = []

    def score_authenticity(self, cadence_realistic: bool, project_detail: bool, tool_scope_realistic: bool):
        score = 0
        if cadence_realistic:
            score += 12
            self.strengths.append("Natural cadence and human sentence rhythm.")
        else:
            self.red_flags.append("Monotone or AI-templated writing cadence.")
        if project_detail:
            score += 8
            self.strengths.append("Includes concrete project-level detail.")
        else:
            self.red_flags.append("Project narratives lack clarity or specificity.")
        if tool_scope_realistic:
            score += 5
        else:
            self.red_flags.append("Tool list appears inflated or implausible.")
        self.category_scores["Authenticity"] = min(score, 25)

    def score_technical_fit(self, stack_alignment: bool, certs_relevant: bool, domain_experience: bool):
        score = 0
        if stack_alignment:
            score += 10
            self.strengths.append("Tech stack aligns with role responsibilities.")
        else:
            self.red_flags.append("Stack misalignment between role and tools.")
        if certs_relevant:
            score += 8
            self.strengths.append("Certifications reinforce technical credibility.")
        if domain_experience:
            score += 7
            self.strengths.append("Domain experience supports claimed expertise.")
        self.category_scores["Technical Fit"] = min(score, 25)

    def score_context_alignment(self, industry_relevant: bool, project_scale_clear: bool):
        score = 0
        if industry_relevant:
            score += 10
        else:
            self.red_flags.append("Industry context unclear or inconsistent.")
        if project_scale_clear:
            score += 5
            self.strengths.append("Project scale and environment clearly described.")
        self.category_scores["Context Alignment"] = min(score, 25)

    def score_noise_control(self, buzzword_stuffing: bool, duplicate_verbs: bool, tool_dump: bool):
        score = 25
        if buzzword_stuffing:
            score -= 5
            self.red_flags.append("Buzzword stuffing detected.")
        if duplicate_verbs:
            score -= 5
            self.red_flags.append("Repetitive phrasing or redundant bullets.")
        if tool_dump:
            score -= 5
            self.red_flags.append("Tool list without context or application.")
        self.category_scores["Noise Control"] = max(0, score)

    def score_timeline_realism(self, tenure_years: int, logical_progression: bool, unexplained_gaps: bool):
        score = 0
        if tenure_years >= 10:
            score += 10
        elif tenure_years >= 5:
            score += 8
        elif tenure_years >= 3:
            score += 6
        else:
            score += 4

        if logical_progression:
            score += 5
            self.strengths.append("Career shows logical progression across roles.")
        else:
            self.red_flags.append("Career trajectory unclear or erratic.")

        if unexplained_gaps:
            score -= 5
            self.red_flags.append("Unexplained gaps detected in timeline.")

        self.category_scores["Timeline Realism"] = max(0, min(score, 15))

    def finalize(self) -> EvaluationResult:
        total = sum(self.category_scores.values())
        if total >= 90:
            verdict = "✅ Highly Credible Resume"
        elif total >= 80:
            verdict = "✅ Credible With Minor Flags"
        elif total >= 70:
            verdict = "⚠️ Questionable – Verify in Screening"
        else:
            verdict = "❌ High Risk – Possibly Fabricated"
        return EvaluationResult(
            candidate=self.name,
            legitimacy_score=total,
            authenticity=self.category_scores.get("Authenticity", 0),
            technical_fit=self.category_scores.get("Technical Fit", 0),
            context_alignment=self.category_scores.get("Context Alignment", 0),
            noise_control=self.category_scores.get("Noise Control", 0),
            timeline_realism=self.category_scores.get("Timeline Realism", 0),
            strengths=self.strengths,
            red_flags=self.red_flags,
            verdict=verdict
        )


def evaluate_resume(candidate_name: str, signals: Dict[str, bool], tenure_years: int,
                    logical_progression=True, unexplained_gaps=False) -> EvaluationResult:
    s = UnifiedResumeScorerV2(candidate_name)
    s.score_authenticity(
        cadence_realistic=signals.get("cadence_realistic", True),
        project_detail=signals.get("project_detail", True),
        tool_scope_realistic=signals.get("tool_scope_realistic", True)
    )
    s.score_technical_fit(
        stack_alignment=signals.get("stack_alignment", True),
        certs_relevant=signals.get("certs_relevant", True),
        domain_experience=signals.get("domain_experience", True)
    )
    s.score_context_alignment(
        industry_relevant=signals.get("industry_relevant", True),
        project_scale_clear=signals.get("project_scale_clear", True)
    )
    s.score_noise_control(
        buzzword_stuffing=signals.get("buzzword_stuffing", False),
        duplicate_verbs=signals.get("duplicate_verbs", False),
        tool_dump=signals.get("tool_dump", False)
    )
    s.score_timeline_realism(tenure_years, logical_progression, unexplained_gaps)
    return s.finalize()


# ======================================================================
# TEXT PARSER + SIGNAL DETECTOR
# ======================================================================

BUZZWORDS = {
    "synergy", "innovative", "results-oriented", "dynamic", "strategic",
    "cutting-edge", "visionary", "transformative", "impactful"
}
TOOL_KEYWORDS = {
    "aws", "azure", "gcp", "terraform", "kubernetes", "docker", "python",
    "ansible", "git", "jenkins", "splunk", "prometheus", "grafana",
    "linux", "windows", "devops", "security", "network"
}


def extract_text_from_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return pdf_extract_text(str(path))
    elif path.suffix.lower() == ".docx":
        return docx2txt.process(str(path))
    else:
        return path.read_text(encoding="utf-8", errors="ignore")


def extract_text_metrics(text: str) -> dict:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    sentences = re.split(r"[.!?]", text)
    words = re.findall(r"\b\w+\b", text.lower())
    unique_words = len(set(words))
    avg_sentence_length = mean([len(s.split()) for s in sentences if len(s.split()) > 2]) if sentences else 0

    cadence_realistic = 8 < avg_sentence_length < 28
    duplicate_verbs = len(re.findall(r"^\s*(led|managed|implemented|designed|developed|built)\b", text, re.M | re.I)) > 8
    buzzword_stuffing = sum(w in BUZZWORDS for w in words) > 8
    tool_dump = len([l for l in lines if len(l.split(",")) > 6]) > 0

    tech_mentions = sum(1 for w in words if w in TOOL_KEYWORDS)
    tool_scope_realistic = tech_mentions / (unique_words + 1) < 0.12
    uses_metrics = bool(re.search(r"\d+%|\$\d+|[0-9]+ (users|systems|servers|projects|apps)", text))
    project_detail = any(k in text.lower() for k in ["project", "migration", "deployment", "architecture", "implementation"])
    industry_relevant = any(k in text.lower() for k in ["finance", "government", "health", "retail", "manufacturing", "public sector"])
    certs_relevant = bool(re.search(r"certified|certificate|ck[a|s]|aws|azure|gcp", text, re.I))
    domain_experience = bool(re.search(r"cloud|devops|security|data|network", text, re.I))

    return {
        "cadence_realistic": cadence_realistic,
        "project_detail": project_detail,
        "tool_scope_realistic": tool_scope_realistic,
        "stack_alignment": domain_experience,
        "certs_relevant": certs_relevant,
        "domain_experience": domain_experience,
        "industry_relevant": industry_relevant,
        "project_scale_clear": project_detail and uses_metrics,
        "buzzword_stuffing": buzzword_stuffing,
        "duplicate_verbs": duplicate_verbs,
        "tool_dump": tool_dump
    }


# ======================================================================
# FASTAPI ENDPOINT
# ======================================================================

from fastapi import Header, HTTPException
import os
import tempfile
import re
from pathlib import Path
from fastapi.responses import JSONResponse
from fastapi import UploadFile, File

# ✅ Load the correct environment variable from Railway
API_KEY = os.getenv("RG_API_KEY", "resume-guard-demo-key")

@app.post("/score_auto")
async def score_auto(file: UploadFile = File(...), x_api_key: str = Header(None)):
    """Upload a resume (PDF, DOCX, or TXT) and get authenticity scores."""
    
    # ✅ Simple API key validation
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid or missing API key")

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        # Extract text and signals
        text = extract_text_from_file(tmp_path)
        signals = extract_text_metrics(text)

        # Estimate career timeline duration
        years = re.findall(r"20\d{2}", text)
        tenure_est = max(1, min((max(map(int, years)) - min(map(int, years))) if len(years) >= 2 else 5, 25))

        # Evaluate using your scoring engine
        result = evaluate_resume(file.filename, signals, tenure_years=tenure_est, logical_progression=True)

        os.remove(tmp_path)
        return JSONResponse(result.__dict__)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


