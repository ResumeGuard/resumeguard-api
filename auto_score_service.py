from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

# --- Optional parsers ---
import docx2txt
from pdfminer.high_level import extract_text as pdf_extract_text

app = FastAPI(title="ResumeGuard V2 API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================================
# ELITE AUTHENTICITY ENGINE V2
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

class EliteAuthenticityEngine:
    """
    Resume Authenticity Engine v2.0
    Built to recognize that elite engineers use AI for polish,
    but fabricators can't fake actual experience.
    
    Trust Score Tiers:
    95-100: Elite Verified - Direct to hiring manager
    85-94:  Fast-Track - Skip phone screen
    75-84:  Authentic - Normal process
    60-74:  Needs Validation - Tech test first
    45-59:  High Scrutiny - Only if desperate
    <45:    Probable Fabrication - Reject
    """
    
    def __init__(self):
        # Initialize pattern databases
        self._initialize_problem_markers()
        self._initialize_unsexy_work_markers()
        self._initialize_tech_timelines()
        self._initialize_chatgpt_markers()
        
    def _initialize_problem_markers(self):
        """Real problems that engineers actually face"""
        self.PROBLEM_MARKERS = [
            # Memory/Resource issues
            r"OOM|out of memory|heap|memory leak|garbage collection",
            r"CPU spike|high load|resource exhaustion",
            r"disk full|inode|file descriptor|ulimit",
            
            # Concurrency/Timing
            r"race condition|deadlock|lock contention|mutex",
            r"timeout|connection refused|ECONNREFUSED|socket hang",
            r"cold start|warm[- ]up|latency spike|performance degradation",
            
            # Version/Dependency
            r"version conflict|dependency hell|incompatible|breaking change",
            r"deprecated|end[- ]of[- ]life|EOL|unsupported",
            
            # State/Data issues
            r"state locking|state file|tfstate|lock file|corrupted state",
            r"data corruption|data loss|backup restore|recovery",
            r"migration failure|schema mismatch|database lock",
            
            # Security/Auth
            r"RBAC|permission denied|403|401|unauthorized|forbidden",
            r"certificate expired|SSL|TLS handshake|cert renewal",
            r"token expired|refresh token|CORS|CSP violation",
            
            # Network/Distributed
            r"split brain|network partition|node failure|quorum",
            r"DNS resolution|NXDOMAIN|stale cache|TTL",
            r"rate limit|throttl|429|quota exceeded",
            
            # Development/Deployment
            r"failed \d+ times|took \d+ attempts|after several",
            r"rolled back|reverted|downgraded|rollback",
            r"bug in|issue with|problem with|quirk in",
            r"error:|exception:|failed to|couldn't|wouldn't",
            r"broke|broken|breaking|failure",
            
            # Specific error messages
            r"segfault|core dump|kernel panic|blue screen",
            r"null pointer|undefined|NaN|type error",
            r"connection pool|thread pool|exhausted",
        ]
        
    def _initialize_unsexy_work_markers(self):
        """Boring work that real engineers actually do"""
        self.UNSEXY_WORK = [
            # Maintenance
            r"legacy|maintained?|upgraded?|patch|hotfix|bugfix",
            r"technical debt|refactor|cleanup|deprecated|EOL",
            r"backwards? compatibility|migration|upgrade path",
            
            # Operations
            r"on[- ]?call|pager|incident|outage|post[- ]?mortem",
            r"runbook|playbook|SOP|standard operating",
            r"monitoring|alerting|observability|logging",
            r"backup|restore|disaster recovery|DR test|BCP",
            
            # Documentation
            r"documentation|documented|readme|wiki|confluence",
            r"onboarding guide|knowledge transfer|handoff",
            
            # Compliance/Security
            r"compliance|audit|SOC\s?2|HIPAA|PCI|GDPR",
            r"vulnerability|CVE|security patch|penetration test",
            r"TLS\s?1\.[0-2]|SSL upgrade|cipher suite",
            
            # Old/Boring Tech
            r"VB\.NET|COBOL|mainframe|AS/?400|fortran",
            r"batch job|cron|scheduled task|ETL",
            r"SOAP|XML|XSLT|WCF|web service",
            r"stored procedure|trigger|cursor|SQL Server \d{4}",
            
            # Unglamorous tasks
            r"log rotation|disk cleanup|archive|purge",
            r"user support|help desk|ticket|escalation",
            r"vendor management|license|procurement",
        ]
        
    def _initialize_tech_timelines(self):
        """Technology release dates for impossibility checking"""
        self.TECH_TIMELINES = {
            "kubernetes": 2014,
            "docker": 2013,
            "terraform": 2014,
            "react": 2013,
            "angular": 2010,
            "vue": 2014,
            "golang": 2009,
            "rust": 2010,
            "aws lambda": 2014,
            "azure functions": 2016,
            "github actions": 2019,
            "chatgpt": 2022,
            "graphql": 2015,
            "kafka": 2011,
            "redis": 2009,
            "mongodb": 2009,
            "elasticsearch": 2010,
            "prometheus": 2012,
            "grafana": 2014,
            "istio": 2017,
            "helm": 2016,
            "argo cd": 2018,
            "github copilot": 2021,
        }
        
    def _initialize_chatgpt_markers(self):
        """ChatGPT signature patterns (not necessarily bad)"""
        self.CHATGPT_VERBS = {
            "high_confidence": ["spearheaded", "orchestrated", "championed"],
            "medium_confidence": ["leveraged", "facilitated", "pioneered", "synergized"],
            "low_confidence": ["implemented", "developed", "managed", "led"]
        }
        
        self.BUZZWORD_PATTERNS = [
            r"cutting[- ]edge",
            r"best[- ]in[- ]class",
            r"world[- ]class",
            r"revolutionary",
            r"transform(?:ative|ational)",
            r"seamless(?:ly)?",
            r"robust solution",
            r"innovative approach",
            r"paradigm shift",
        ]
    
    def evaluate_resume(self, 
                       resume_text: str, 
                       candidate_name: str = "Unknown",
                       roles: Optional[List[Dict]] = None) -> ResumeEvaluation:
        """
        Main evaluation function.
        Returns a ResumeEvaluation object with trust score and recommendations.
        """
        
        # Start with base score
        trust_score = 70
        red_flags = []
        authenticity_markers = []
        
        # Clean text
        text_lower = resume_text.lower()
        
        # Layer 1: Check for impossibilities (hard cap at 45)
        impossibilities = self._check_impossibilities(resume_text, roles)
        if impossibilities["found"]:
            trust_score = min(45, trust_score)
            red_flags.extend(impossibilities["flags"])
        
        # Layer 2: Technical Coherence
        coherence = self._evaluate_technical_coherence(resume_text, roles)
        trust_score += coherence["score"]
        if coherence["markers"]:
            authenticity_markers.extend(coherence["markers"])
        if coherence["flags"]:
            red_flags.extend(coherence["flags"])
        
        # Layer 3: Problem Specificity Index (GAME CHANGER)
        psi = self._calculate_problem_specificity(resume_text)
        if psi["score"] > 0:
            trust_score += psi["score"]
            authenticity_markers.append(
                f"Problem Specificity: +{psi['score']} ({psi['count']} real problems mentioned)"
            )
        
        # Layer 4: Unsexy Work Bonus
        unsexy = self._calculate_unsexy_work_bonus(resume_text)
        if unsexy["score"] > 0:
            trust_score += unsexy["score"]
            authenticity_markers.append(
                f"Unsexy Work: +{unsexy['score']} (does real maintenance/ops work)"
            )
        
        # Layer 5: ChatGPT Detection (but don't over-penalize)
        chatgpt = self._detect_chatgpt_patterns(resume_text)
        if chatgpt["high_count"] >= 2:
            if coherence["score"] > 10:
                authenticity_markers.append("AI-polished but technically sound (no penalty)")
            else:
                trust_score -= 10
                red_flags.append(f"Heavy AI generation without depth ({chatgpt['high_count']} high-confidence AI verbs)")
        
        # Bonus: Specific version numbers
        if re.search(r'\d+\.\d+(?:\.\d+)?', resume_text):
            trust_score += 3
            authenticity_markers.append("Mentions specific versions")
        
        # Bonus: Admits failures
        if re.search(r'failed|struggled|difficult|took .+ attempts|harder than expected', text_lower):
            trust_score += 5
            authenticity_markers.append("Admits struggles/failures")
        
        # Final calibration
        trust_score = max(20, min(100, trust_score))
        
        # Determine verdict and action
        verdict, action, tier = self._determine_verdict(trust_score)
        
        # Calculate confidence
        confidence = self._calculate_confidence(trust_score, len(red_flags), len(authenticity_markers))
        
        return ResumeEvaluation(
            name=candidate_name,
            trust_score=trust_score,
            verdict=verdict,
            recommended_action=action,
            red_flags=red_flags,
            authenticity_markers=authenticity_markers,
            has_problem_specificity=psi["score"] > 0,
            does_unsexy_work=unsexy["score"] > 0,
            screening_tier=tier,
            confidence=confidence
        )
    
    def _check_impossibilities(self, text: str, roles: Optional[List[Dict]]) -> Dict:
        """Check for impossible claims like using tools before they existed"""
        found = False
        flags = []
        
        if not roles:
            return {"found": found, "flags": flags}
        
        text_lower = text.lower()
        
        for role in roles:
            year = None
            if "start" in role:
                year_match = re.search(r'(\d{4})', str(role["start"]))
                if year_match:
                    year = int(year_match.group(1))
            
            if not year:
                continue
            
            role_text = role.get("description", "").lower()
            for tech, release_year in self.TECH_TIMELINES.items():
                if tech in role_text and year < release_year:
                    found = True
                    flags.append(f"{tech.title()} claimed in {year} (released {release_year})")
        
        return {"found": found, "flags": flags}
    
    def _evaluate_technical_coherence(self, text: str, roles: Optional[List[Dict]]) -> Dict:
        """Evaluate if the technical content makes sense"""
        score = 0
        markers = []
        flags = []
        
        text_lower = text.lower()
        
        # Positive signals
        if re.search(r'migrated?.{0,20}from.{0,20}to|upgraded?.{0,20}from.{0,20}to|replaced.{0,20}with', text_lower):
            score += 7
            markers.append("Shows technical progression")
        
        if re.search(r'microservice|event[- ]driven|CQRS|saga pattern|circuit breaker', text_lower, re.I):
            score += 5
            markers.append("Mentions real architectural patterns")
        
        if re.search(r'CI/?CD|continuous integration|continuous deployment|pipeline', text_lower):
            score += 3
            markers.append("DevOps practices mentioned")
        
        if re.search(r'constraint|limitation|bottleneck|challenge|trade[- ]off', text_lower):
            score += 5
            markers.append("Discusses real constraints")
        
        # Negative signals
        generic_companies = len(re.findall(r'tech solutions?|global systems?|innovative solutions?', text_lower))
        if generic_companies >= 2:
            score -= 5
            flags.append(f"{generic_companies} generic company names")
        
        vague_improvements = len(re.findall(r'improved?.{0,10}by \d+%|increased?.{0,10}by \d+%|reduced?.{0,10}by \d+%', text_lower))
        contextual_improvements = len(re.findall(r'from \d+.{0,10}to \d+|from \$.{0,10}to \$', text_lower))
        
        if vague_improvements > 3 and contextual_improvements == 0:
            score -= 5
            flags.append(f"{vague_improvements} vague percentage improvements without baselines")
        
        return {"score": score, "markers": markers, "flags": flags}
    
    def _calculate_problem_specificity(self, text: str) -> Dict:
        """Real engineers remember what broke"""
        text_lower = text.lower()
        problems_found = set()
        
        for pattern in self.PROBLEM_MARKERS:
            if re.search(pattern, text_lower):
                problems_found.add(pattern.split('|')[0])
        
        count = len(problems_found)
        score = min(20, count * 5)
        
        return {"score": score, "count": count, "problems": list(problems_found)}
    
    def _calculate_unsexy_work_bonus(self, text: str) -> Dict:
        """Real engineers do boring work"""
        text_lower = text.lower()
        unsexy_found = set()
        
        for pattern in self.UNSEXY_WORK:
            if re.search(pattern, text_lower):
                unsexy_found.add(pattern.split('|')[0])
        
        count = len(unsexy_found)
        score = min(10, count * 3)
        
        return {"score": score, "count": count, "tasks": list(unsexy_found)}
    
    def _detect_chatgpt_patterns(self, text: str) -> Dict:
        """Detect ChatGPT patterns (not always bad)"""
        text_lower = text.lower()
        
        high_count = sum(1 for verb in self.CHATGPT_VERBS["high_confidence"] if verb in text_lower)
        medium_count = sum(1 for verb in self.CHATGPT_VERBS["medium_confidence"] if verb in text_lower)
        buzzword_count = sum(1 for pattern in self.BUZZWORD_PATTERNS if re.search(pattern, text_lower))
        
        return {
            "high_count": high_count,
            "medium_count": medium_count,
            "buzzword_count": buzzword_count
        }
    
    def _determine_verdict(self, score: int) -> Tuple[str, str, str]:
        """Determine verdict, action, and tier based on trust score"""
        
        if score >= 95:
            return (
                "✅ Elite Verified - This person has been in the trenches",
                "Skip to final interview / Direct to hiring manager",
                "Elite Tier"
            )
        elif score >= 85:
            return (
                "✅ Fast-Track Authentic - Highly trusted professional",
                "Skip phone screen → Technical interview",
                "Fast-Track Tier"
            )
        elif score >= 75:
            return (
                "✅ Authentic - Real experience, AI-assisted presentation",
                "Normal interview process",
                "Standard Tier"
            )
        elif score >= 60:
            return (
                "🟡 Needs Validation - Likely real but verify claims",
                "Technical assessment before phone screen",
                "Validation Tier"
            )
        elif score >= 45:
            return (
                "⚠️ High Scrutiny - Multiple red flags",
                "Only proceed if desperate + rigorous vetting",
                "High Risk Tier"
            )
        else:
            return (
                "❌ Probable Fabrication - High risk",
                "Reject unless critical need + extreme vetting",
                "Rejection Tier"
            )
    
    def _calculate_confidence(self, score: int, red_flags: int, auth_markers: int) -> float:
        """Calculate confidence in the assessment"""
        base_confidence = 0.7
        
        if score >= 85:
            base_confidence += 0.15
        elif score >= 70:
            base_confidence += 0.10
        elif score <= 45:
            base_confidence += 0.15
        
        evidence_factor = min(0.1, (auth_markers + red_flags) * 0.02)
        
        return min(0.95, base_confidence + evidence_factor)


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
    """Upload a resume and get V2 authenticity analysis"""
    
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid or missing API key")

    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        # Extract text
        text = extract_text_from_file(tmp_path)
        
        # Evaluate with V2 engine
        engine = EliteAuthenticityEngine()
        result = engine.evaluate_resume(
            resume_text=text,
            candidate_name=file.filename
        )

        os.remove(tmp_path)
        
        # Convert dataclass to dict for JSON response
        return JSONResponse(asdict(result))

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/")
def root():
    return {
        "service": "ResumeGuard V2 API",
        "version": "2.0",
        "engine": "Elite Authenticity Engine",
        "features": [
            "Problem Specificity Index",
            "Unsexy Work Detection",
            "Technical Coherence Analysis",
            "AI-Polish Recognition",
            "Trust Score Tiers"
        ]
    }
