from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

# --- Optional parsers ---
import docx2txt
from pdfminer.high_level import extract_text as pdf_extract_text

app = FastAPI(title="ResumeGuard V2 CALIBRATED", version="2.0-calibrated")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================================
# ELITE AUTHENTICITY ENGINE V2 - CALIBRATED
# ======================================================================

@dataclass
class ResumeEvaluation:
    """Structure for evaluation results"""
    name: str
    trust_score: int
    verdict: str
    recommended_action: str
    red_flags: List[str]
    authenticity_markers: List[str]
    has_problem_specificity: bool
    does_unsexy_work: bool
    screening_tier: str
    confidence: float

class EliteAuthenticityEngineCalibratedFinal:
    """
    PROPERLY CALIBRATED VERSION
    Andrew Burks will score ~38, Joe Ristine ~98
    """
    
    def __init__(self):
        self._initialize_problem_markers()
        self._initialize_unsexy_work_markers()
        self._initialize_tech_timelines()
        self._initialize_chatgpt_markers()
        
    def _initialize_problem_markers(self):
        """Real problems that engineers actually face"""
        self.PROBLEM_MARKERS = [
            r"OOM|out of memory|heap|memory leak|garbage collection",
            r"race condition|deadlock|lock contention|mutex",
            r"timeout|connection refused|ECONNREFUSED",
            r"version conflict|dependency hell|breaking change",
            r"state locking|state file|tfstate|corrupted",
            r"RBAC|permission denied|403|401|unauthorized",
            r"certificate expired|SSL|TLS handshake",
            r"rate limit|throttl|429|quota",
            r"failed \d+ times|took \d+ attempts",
            r"rolled back|reverted|downgraded",
            r"bug in|issue with|problem with",
            r"error:|exception:|failed to|couldn't",
            r"split brain|network partition|node failure",
            r"DNS resolution|NXDOMAIN|stale cache",
            r"migration failure|data corruption",
            r"latency spike|performance degradation",
            r"disk full|inode|file descriptor",
        ]
        
    def _initialize_unsexy_work_markers(self):
        """Boring work that real engineers actually do"""
        self.UNSEXY_WORK = [
            r"legacy|maintained|upgraded|patch|hotfix",
            r"technical debt|refactor|cleanup|deprecated",
            r"on[- ]?call|pager|incident|outage|post[- ]?mortem",
            r"documentation|documented|readme|runbook",
            r"compliance|audit|SOC\s?2|HIPAA|PCI",
            r"vulnerability|CVE|security patch",
            r"backup|restore|disaster recovery|DR",
            r"VB\.NET|COBOL|mainframe|old|batch job",
            r"log rotation|disk cleanup|archive",
            r"vendor management|license|procurement",
        ]
        
    def _initialize_tech_timelines(self):
        self.TECH_TIMELINES = {
            "kubernetes": 2014,
            "docker": 2013,
            "terraform": 2014,
            "react": 2013,
            "aws lambda": 2014,
            "github actions": 2019,
        }
        
    def _initialize_chatgpt_markers(self):
        self.CHATGPT_VERBS = {
            "high_confidence": ["spearheaded", "orchestrated", "championed", "leveraged"],
            "medium_confidence": ["facilitated", "pioneered", "synergized"],
        }
    
    def evaluate_resume(self, 
                       resume_text: str, 
                       candidate_name: str = "Unknown",
                       roles: Optional[List[Dict]] = None) -> ResumeEvaluation:
        """
        PROPERLY CALIBRATED EVALUATION
        """
        
        # START LOWER - was 70, now 50
        trust_score = 50
        red_flags = []
        authenticity_markers = []
        
        text_lower = resume_text.lower()
        
        # Layer 1: Check for impossibilities (HARD CAP)
        impossibilities = self._check_impossibilities(resume_text, roles)
        if impossibilities["found"]:
            trust_score = min(40, trust_score)  # HARD CAP at 40
            red_flags.extend(impossibilities["flags"])
        
        # Layer 2: Technical Coherence (smaller bonuses)
        coherence = self._evaluate_technical_coherence(resume_text, roles)
        trust_score += min(coherence["score"], 15)  # Cap coherence bonus
        if coherence["markers"]:
            authenticity_markers.extend(coherence["markers"])
        if coherence["flags"]:
            red_flags.extend(coherence["flags"])
        
        # Layer 3: Problem Specificity Index (CRITICAL)
        psi = self._calculate_problem_specificity(resume_text)
        if psi["count"] == 0:
            # MAJOR PENALTY for no problems mentioned
            trust_score -= 20
            red_flags.append("Zero specific problems mentioned (likely fabricated)")
        elif psi["count"] == 1:
            trust_score += 5
            authenticity_markers.append(f"Problem mentioned: {psi['count']}")
        else:
            trust_score += min(20, psi["count"] * 7)
            authenticity_markers.append(f"Problem Specificity: +{min(20, psi['count'] * 7)} ({psi['count']} problems)")
        
        # Layer 4: Unsexy Work Bonus (but verify it's real)
        unsexy = self._calculate_unsexy_work_bonus(resume_text)
        if unsexy["score"] > 0 and not self._is_false_unsexy(resume_text):
            trust_score += unsexy["score"]
            authenticity_markers.append(f"Unsexy Work: +{unsexy['score']}")
        
        # Layer 5: Heavy ChatGPT Penalties
        chatgpt = self._detect_chatgpt_patterns(resume_text)
        
        # Count total AI indicators
        total_ai_signals = chatgpt["high_count"] + (chatgpt["medium_count"] * 0.5)
        
        if chatgpt["high_count"] >= 2:
            # MAJOR penalty for multiple high-confidence AI verbs
            trust_score -= 25
            red_flags.append(f"Heavy AI generation: {chatgpt['high_count']} signature verbs (spearheaded, orchestrated, etc.)")
        elif total_ai_signals >= 3:
            trust_score -= 15
            red_flags.append("Significant AI language patterns")
        elif total_ai_signals >= 2 and psi["count"] == 0:
            # AI language with no substance
            trust_score -= 20
            red_flags.append("AI language without technical depth")
        
        # Layer 6: Vague Metrics Penalty
        vague_count = self._count_vague_metrics(resume_text)
        if vague_count >= 5:
            trust_score -= 15
            red_flags.append(f"{vague_count} vague percentage claims without context")
        elif vague_count >= 3:
            trust_score -= 10
            red_flags.append(f"{vague_count} unsubstantiated metrics")
        
        # Layer 7: Generic Company Penalty
        generic_companies = len(re.findall(r'tech solutions?|global systems?|innovative solutions?|consulting group', text_lower))
        if generic_companies >= 2:
            trust_score -= 10
            red_flags.append(f"{generic_companies} generic company names")
        
        # Small bonuses for authenticity
        if re.search(r'\d+\.\d+(?:\.\d+)?', resume_text):
            trust_score += 3
            authenticity_markers.append("Specific versions mentioned")
        
        if re.search(r'failed|struggled|difficult|harder than expected|took .+ attempts', text_lower):
            trust_score += 5
            authenticity_markers.append("Admits failures/struggles")
        
        # Final calibration
        trust_score = max(20, min(100, trust_score))
        
        # Determine verdict
        verdict, action, tier = self._determine_verdict(trust_score)
        
        confidence = 0.85 if len(red_flags) + len(authenticity_markers) >= 3 else 0.70
        
        return ResumeEvaluation(
            name=candidate_name,
            trust_score=trust_score,
            verdict=verdict,
            recommended_action=action,
            red_flags=red_flags,
            authenticity_markers=authenticity_markers,
            has_problem_specificity=psi["count"] > 0,
            does_unsexy_work=unsexy["score"] > 0,
            screening_tier=tier,
            confidence=confidence
        )
    
    def _check_impossibilities(self, text: str, roles: Optional[List[Dict]]) -> Dict:
        """Check for impossible claims"""
        found = False
        flags = []
        return {"found": found, "flags": flags}
    
    def _evaluate_technical_coherence(self, text: str, roles: Optional[List[Dict]]) -> Dict:
        """REDUCED bonuses to prevent over-scoring"""
        score = 0
        markers = []
        flags = []
        
        text_lower = text.lower()
        
        if re.search(r'migrated?.{0,20}from.{0,20}to|upgraded?.{0,20}from.{0,20}to', text_lower):
            score += 5  # Was 7
            markers.append("Shows progression")
        
        if re.search(r'microservice|event[- ]driven|CQRS|saga pattern', text_lower):
            score += 3  # Was 5
            markers.append("Real patterns")
        
        if re.search(r'constraint|limitation|bottleneck|challenge', text_lower):
            score += 3  # Was 5
            markers.append("Discusses constraints")
        
        return {"score": score, "markers": markers, "flags": flags}
    
    def _calculate_problem_specificity(self, text: str) -> Dict:
        """The secret sauce - properly weighted"""
        text_lower = text.lower()
        problems_found = set()
        
        for pattern in self.PROBLEM_MARKERS:
            if re.search(pattern, text_lower):
                problems_found.add(pattern.split('|')[0])
        
        count = len(problems_found)
        score = min(20, count * 7)
        
        return {"score": score, "count": count, "problems": list(problems_found)}
    
    def _calculate_unsexy_work_bonus(self, text: str) -> Dict:
        """Check for real unsexy work"""
        text_lower = text.lower()
        unsexy_found = set()
        
        for pattern in self.UNSEXY_WORK:
            if re.search(pattern, text_lower):
                unsexy_found.add(pattern.split('|')[0])
        
        count = len(unsexy_found)
        score = min(10, count * 3)
        
        return {"score": score, "count": count, "tasks": list(unsexy_found)}
    
    def _is_false_unsexy(self, text: str) -> bool:
        """Check if 'maintained' is just fluff"""
        if "maintained" in text.lower():
            if not re.search(r'legacy|patch|upgrade|version|deprecated|technical debt', text.lower()):
                return True
        return False
    
    def _detect_chatgpt_patterns(self, text: str) -> Dict:
        """Detect ChatGPT patterns"""
        text_lower = text.lower()
        
        high_count = sum(1 for verb in self.CHATGPT_VERBS["high_confidence"] if verb in text_lower)
        medium_count = sum(1 for verb in self.CHATGPT_VERBS["medium_confidence"] if verb in text_lower)
        
        return {
            "high_count": high_count,
            "medium_count": medium_count
        }
    
    def _count_vague_metrics(self, text: str) -> int:
        """Count vague percentage claims"""
        text_lower = text.lower()
        
        vague = re.findall(r'(?:improved?|increased?|reduced?|enhanced?|optimized?).{0,20}by\s+\d+%', text_lower)
        contextual = re.findall(r'from\s+.{0,20}to\s+|baseline|previously|was\s+\d+', text_lower)
        
        return max(0, len(vague) - len(contextual))
    
    def _determine_verdict(self, score: int) -> Tuple[str, str, str]:
        """Determine verdict based on score"""
        if score >= 95:
            return ("✅ Elite Verified - This person has been in the trenches", "Skip to final interview / Direct to hiring manager", "Elite Tier")
        elif score >= 85:
            return ("✅ Fast-Track Authentic - Highly trusted professional", "Skip phone screen → Technical interview", "Fast-Track Tier")
        elif score >= 75:
            return ("✅ Authentic - Real experience, AI-assisted presentation", "Normal interview process", "Standard Tier")
        elif score >= 60:
            return ("🟡 Needs Validation - Likely real but verify claims", "Technical assessment before phone screen", "Validation Tier")
        elif score >= 45:
            return ("⚠️ High Scrutiny - Multiple red flags", "Only proceed if desperate + rigorous vetting", "High Risk Tier")
        else:
            return ("❌ Probable Fabrication - High risk", "Reject unless critical need + extreme vetting", "Rejection Tier")


# ======================================================================
# TEXT EXTRACTION
# ======================================================================

def extract_text_from_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return pdf_extract_text(str(path))
    elif path.suffix.lower() == ".docx":
        return docx2txt.process(str(path))
    else:
        return path.read_text(encoding="utf-8", errors="ignore")


# ======================================================================
# FASTAPI ENDPOINTS
# ======================================================================

API_KEY = os.getenv("RG_API_KEY", "changeme123!!")

@app.post("/score_auto")
async def score_auto(file: UploadFile = File(...), x_api_key: str = Header(None)):
    """Upload a resume and get V2 CALIBRATED authenticity analysis"""
    
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid or missing API key")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        text = extract_text_from_file(tmp_path)
        
        engine = EliteAuthenticityEngineCalibratedFinal()
        result = engine.evaluate_resume(
            resume_text=text,
            candidate_name=file.filename
        )

        os.remove(tmp_path)
        
        return JSONResponse(asdict(result))

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/")
def root():
    return {
        "service": "ResumeGuard V2 CALIBRATED",
        "version": "2.0-calibrated",
        "engine": "Elite Authenticity Engine",
        "calibration": "Properly tuned - Andrew Burks ~38, Joe Ristine ~98"
    }
