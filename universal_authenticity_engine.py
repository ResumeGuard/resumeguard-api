import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

# Domain question modules (for tech roles)
try:
    from domains.devops import q5_devops
    from domains.backend_api import q5_backend
    from domains.iam import q5_iam
    from domains.data_eng import q5_data
    DOMAIN_MODULES_AVAILABLE = True
except ImportError:
    DOMAIN_MODULES_AVAILABLE = False


@dataclass
class AuthenticityResult:
    """Universal result structure for any industry"""
    candidate: str
    star_rating: float
    star_display: str
    tier: str
    verdict: str
    raw_score: int  # Internal only - not shown to users
    red_flags: List[str]
    authenticity_markers: List[str]
    industry_detected: Optional[str]
    industry_confidence: float
    has_real_problems: bool
    ai_probability: str
    universal_questions: List[str]
    industry_specific_question: Optional[str]


class UniversalAuthenticityEngine:
    """
    Cross-industry authenticity scoring engine
    Works for Tech, Sales, Marketing, Finance, Healthcare, Operations, HR, etc.
    
    Philosophy: "Guilty until proven innocent" - start at 30, earn your way up
    Output: 5-star rating system (no half-stars, clean buckets)
    """
    
    def __init__(self):
        self._initialize_universal_patterns()
        self._initialize_industry_patterns()
        self._initialize_universal_questions()
        
    def _initialize_universal_patterns(self):
        """Patterns that work across ALL industries"""
        
        # AI/ChatGPT verbs (universal red flags)
        self.AI_VERBS = {
            "high_confidence": [
                "spearheaded", "orchestrated", "championed", "leveraged",
                "pioneered", "revolutionized", "transformed", "synergized"
            ],
            "medium_confidence": [
                "facilitated", "optimized", "enhanced", "streamlined",
                "modernized", "catalyzed"
            ]
        }
        
        # Vague metrics without context (universal)
        self.VAGUE_METRICS_PATTERN = r'(?:improved?|increased?|reduced?|enhanced?|optimized?|boosted?|drove).{0,25}by\s+\d+%'
        
        # Generic company names (universal)
        self.GENERIC_COMPANIES = [
            "tech solutions", "global systems", "innovative solutions",
            "consulting group", "technology partners", "digital solutions",
            "strategic consulting", "enterprise solutions"
        ]
        
        # Authenticity markers (universal)
        self.AUTHENTICITY_PATTERNS = {
            "version_numbers": r'\d+\.\d+(?:\.\d+)?',  # "v3.1.4"
            "admits_struggles": r'failed|struggled|difficult|challenge|took .+ attempts|harder than expected',
            "contextual_metrics": r'from\s+[\d.]+.{0,20}to\s+[\d.]+|baseline|previously|was\s+\d+',
            "mixed_sentence_length": True,  # Check variance in bullet lengths
        }
        
    def _initialize_industry_patterns(self):
        """Industry-specific problem markers and unsexy work"""
        
        self.INDUSTRY_PATTERNS = {
            "tech": {
                "keywords": ["kubernetes", "terraform", "aws", "docker", "api", "microservice", "cicd", "devops"],
                "problems": [
                    r"OOM|out of memory|heap|memory leak|garbage collection",
                    r"race condition|deadlock|lock contention|mutex",
                    r"timeout|connection refused|ECONNREFUSED|ETIMEDOUT",
                    r"version conflict|dependency hell|breaking change",
                    r"rolled back|reverted|downgraded|rollback",
                    r"bug in|issue with|debugged|troubleshoot",
                    r"latency spike|performance degradation|slow query",
                    r"production incident|outage|downtime|post-?mortem",
                    r"state locking|tfstate|corrupted state",
                    r"certificate expired|SSL|TLS handshake"
                ],
                "unsexy_work": [
                    r"legacy|maintained|upgraded|patch|hotfix",
                    r"technical debt|refactor|cleanup|deprecated",
                    r"on[- ]?call|pager|incident response",
                    r"documentation|documented|runbook|wiki",
                    r"compliance|audit|SOC\s?2|HIPAA",
                    r"vulnerability|CVE|security patch",
                    r"backup|restore|disaster recovery|DR"
                ]
            },
            
            "sales": {
                "keywords": ["quota", "pipeline", "crm", "salesforce", "outreach", "closed", "deals", "revenue"],
                "problems": [
                    r"missed quota|below target|lost deal",
                    r"churn|cancellation|downgrade",
                    r"pipeline dried up|low activity",
                    r"competitive loss|lost to \w+",
                    r"pricing objection|budget concerns",
                    r"ghosted|unresponsive|went dark",
                    r"deal fell through|stalled|stuck",
                    r"gatekeeper|couldn't reach|blocked"
                ],
                "unsexy_work": [
                    r"cold call|prospecting|outbound",
                    r"data cleanup|list hygiene|duplicate",
                    r"lost accounts|territory cleanup",
                    r"admin|crm updates|logging calls",
                    r"rejection|no response|not interested"
                ]
            },
            
            "marketing": {
                "keywords": ["campaign", "seo", "sem", "content", "email", "social", "analytics", "conversion"],
                "problems": [
                    r"low CTR|click[- ]?through rate",
                    r"campaign failed|underperformed|missed target",
                    r"A/B test showed|conversion dropped",
                    r"budget cut|reduced spend",
                    r"low engagement|high bounce",
                    r"algorithm change|organic decline",
                    r"unsubscribe rate|spam complaints",
                    r"attribution issue|tracking broke"
                ],
                "unsexy_work": [
                    r"A/B testing|split test|multivariate",
                    r"list hygiene|email cleanup",
                    r"reporting|dashboard|weekly metrics",
                    r"compliance|GDPR|CAN-SPAM|opt[- ]?out",
                    r"manual posting|scheduling|calendar"
                ]
            },
            
            "finance": {
                "keywords": ["ledger", "reconciliation", "gaap", "sox", "forecast", "budget", "variance", "audit"],
                "problems": [
                    r"variance|discrepancy|mismatch",
                    r"audit finding|control deficiency",
                    r"reconciliation issue|unreconciled",
                    r"late close|missed deadline",
                    r"journal entry error|posting error",
                    r"system cutover|migration issue",
                    r"compliance gap|sox issue",
                    r"manual workaround|spreadsheet error"
                ],
                "unsexy_work": [
                    r"month[- ]?end close|quarter[- ]?end",
                    r"expense reports|T&E|travel reimbursement",
                    r"accrual|deferrals|prepaid",
                    r"audit prep|audit support|sox testing",
                    r"reconciliation|rec|bank rec",
                    r"journal entries|manual entries"
                ]
            },
            
            "healthcare": {
                "keywords": ["patient", "clinical", "ehr", "emr", "hipaa", "epic", "cerner", "billing", "icd"],
                "problems": [
                    r"patient load|high census|understaffed",
                    r"protocol violation|non[- ]?compliance",
                    r"medication error|dosage issue",
                    r"wait time|delay|backlog",
                    r"billing error|claim denied|rejected",
                    r"system downtime|ehr crash",
                    r"staffing shortage|turnover|retention"
                ],
                "unsexy_work": [
                    r"documentation|charting|notes",
                    r"compliance|hipaa|audit",
                    r"on[- ]?call|night shift|weekend",
                    r"prior auth|authorization|approval",
                    r"claims|billing|coding|icd-?\d+"
                ]
            },
            
            "operations": {
                "keywords": ["logistics", "supply chain", "inventory", "fulfillment", "warehouse", "procurement"],
                "problems": [
                    r"stockout|out of stock|inventory shortage",
                    r"delivery delay|late shipment|missed sla",
                    r"damaged goods|defect|return",
                    r"supplier issue|vendor delay",
                    r"capacity constraint|bottleneck",
                    r"forecast error|demand spike",
                    r"warehouse issue|picking error"
                ],
                "unsexy_work": [
                    r"cycle count|physical inventory",
                    r"vendor management|po|purchase order",
                    r"receiving|put[- ]?away|bin location",
                    r"reporting|kpi|metrics tracking",
                    r"safety|osha|incident report"
                ]
            },
            
            "hr": {
                "keywords": ["recruiting", "talent", "onboarding", "hris", "workday", "adp", "benefits", "compensation"],
                "problems": [
                    r"turnover|attrition|retention issue",
                    r"offer decline|candidate withdrew",
                    r"slow hire|time to fill",
                    r"compliance issue|labor law|misclassification",
                    r"benefit error|payroll error",
                    r"employee complaint|investigation",
                    r"low engagement|survey results"
                ],
                "unsexy_work": [
                    r"onboarding|new hire|paperwork",
                    r"compliance|i-?9|background check",
                    r"exit interview|offboarding|termination",
                    r"open enrollment|benefit admin",
                    r"hris data|cleanup|audit"
                ]
            }
        }
        
    def _initialize_universal_questions(self):
        """4 universal questions that work across all industries"""
        self.UNIVERSAL_QUESTIONS = [
            "What decision authority did you personally hold on this project?",
            "What did the team depend on you for that no one else could do?",
            "What changed in the system/process because of your involvement?",
            "What trade-off did you personally choose and why?"
        ]
    
    def score_resume(self, 
                    resume_text: str,
                    candidate_name: str = "Unknown") -> AuthenticityResult:
        """
        Main scoring function - works across all industries
        Philosophy: Start low (30), earn your way up
        """
        
        # START AT 30 - Guilty until proven innocent
        base_score = 30
        red_flags = []
        authenticity_markers = []
        
        text_lower = resume_text.lower()
        
        # LAYER 1: Detect Industry
        industry, confidence = self._detect_industry(text_lower)
        
        # LAYER 2: Check for Hard Fails
        hard_fail = self._check_hard_fails(text_lower, industry)
        if hard_fail:
            base_score = min(hard_fail, base_score)
            red_flags.append(f"Critical issues detected - capped at {hard_fail}")
        
        # LAYER 3: Problem Specificity (MOST IMPORTANT)
        problem_analysis = self._analyze_problems(text_lower, industry)
        if problem_analysis["count"] == 0:
            base_score -= 20  # MAJOR PENALTY
            red_flags.append("❌ Zero specific problems or challenges mentioned")
        elif problem_analysis["count"] <= 2:
            base_score += problem_analysis["count"] * 10
            authenticity_markers.append(f"✓ {problem_analysis['count']} real problem(s) discussed")
        else:
            base_score += min(35, problem_analysis["count"] * 12)
            authenticity_markers.append(f"✓ Strong problem specificity ({problem_analysis['count']} issues)")
        
        # LAYER 4: AI/ChatGPT Detection
        ai_analysis = self._detect_ai_patterns(text_lower)
        if ai_analysis["high_count"] >= 3:
            base_score = min(40, base_score)  # Hard cap
            red_flags.append(f"⚠️ Heavy AI language ({ai_analysis['high_count']} signature verbs)")
        elif ai_analysis["total_weighted"] >= 4:
            base_score -= 18
            red_flags.append("⚠️ Significant AI patterns detected")
        elif ai_analysis["total_weighted"] >= 2:
            base_score -= 10
            red_flags.append("⚠️ Some AI language patterns")
        
        # LAYER 5: Unsexy Work Bonus
        unsexy_analysis = self._analyze_unsexy_work(text_lower, industry)
        if unsexy_analysis["score"] > 0:
            base_score += unsexy_analysis["score"]
            authenticity_markers.append(f"✓ Real unsexy work: +{unsexy_analysis['score']}")
        
        # LAYER 6: Authenticity Bonuses
        authenticity_bonus = self._calculate_authenticity_bonus(resume_text, text_lower)
        base_score += authenticity_bonus["score"]
        if authenticity_bonus["markers"]:
            authenticity_markers.extend(authenticity_bonus["markers"])
        
        # LAYER 7: Vague Metrics Penalty
        vague_count = self._count_vague_metrics(text_lower)
        if vague_count >= 5:
            base_score -= 15
            red_flags.append(f"⚠️ {vague_count} unsubstantiated metrics")
        elif vague_count >= 3:
            base_score -= 8
            red_flags.append(f"⚠️ {vague_count} vague percentage claims")
        
        # LAYER 8: Generic Company Penalty
        generic_count = self._count_generic_companies(text_lower)
        if generic_count >= 2:
            base_score -= 10
            red_flags.append(f"⚠️ {generic_count} generic company names")
        
        # Calculate final score
        raw_score = max(20, min(100, base_score))
        
        # Convert to 5-star system
        star_rating, star_display, tier, verdict = self._convert_to_stars(raw_score)
        
        # Get industry-specific question
        industry_question = self._get_industry_question(industry, text_lower, confidence)
        
        # Determine AI probability
        ai_probability = self._determine_ai_probability(ai_analysis, problem_analysis)
        
        return AuthenticityResult(
            candidate=candidate_name,
            star_rating=star_rating,
            star_display=star_display,
            tier=tier,
            verdict=verdict,
            raw_score=raw_score,
            red_flags=red_flags,
            authenticity_markers=authenticity_markers,
            industry_detected=industry if confidence >= 0.3 else None,
            industry_confidence=confidence,
            has_real_problems=problem_analysis["count"] > 0,
            ai_probability=ai_probability,
            universal_questions=self.UNIVERSAL_QUESTIONS,
            industry_specific_question=industry_question
        )
    
    def _detect_industry(self, text_lower: str) -> Tuple[Optional[str], float]:
        """Auto-detect industry from resume content"""
        scores = {}
        
        for industry, patterns in self.INDUSTRY_PATTERNS.items():
            keyword_hits = sum(1 for kw in patterns["keywords"] if kw in text_lower)
            scores[industry] = keyword_hits / len(patterns["keywords"])
        
        if scores:
            best_industry = max(scores, key=scores.get)
            return (best_industry, scores[best_industry]) if scores[best_industry] > 0 else (None, 0)
        return (None, 0)
    
    def _check_hard_fails(self, text_lower: str, industry: Optional[str]) -> Optional[int]:
        """Check for immediate disqualifiers"""
        
        # Multiple "spearheaded" or "orchestrated"
        spearhead_count = text_lower.count("spearheaded") + text_lower.count("orchestrated")
        if spearhead_count >= 4:
            return 35
        
        # Tech-specific timeline checks (if tech industry detected)
        if industry == "tech":
            if re.search(r"kubernetes.*20(0[0-9]|1[0-3])", text_lower):
                return 38  # K8s before 2014
            if re.search(r"terraform.*20(0[0-9]|1[0-3])", text_lower):
                return 38  # Terraform before 2014
        
        return None
    
    def _analyze_problems(self, text_lower: str, industry: Optional[str]) -> Dict:
        """Count real problems mentioned (industry-aware)"""
        problems_found = set()
        
        # Use industry-specific patterns if detected
        if industry and industry in self.INDUSTRY_PATTERNS:
            patterns = self.INDUSTRY_PATTERNS[industry]["problems"]
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    problems_found.add(pattern.split('|')[0])
        else:
            # Fallback: check all industries
            for ind_patterns in self.INDUSTRY_PATTERNS.values():
                for pattern in ind_patterns["problems"]:
                    if re.search(pattern, text_lower):
                        problems_found.add(pattern.split('|')[0])
        
        return {
            "count": len(problems_found),
            "problems": list(problems_found)[:5]
        }
    
    def _detect_ai_patterns(self, text_lower: str) -> Dict:
        """Detect AI/ChatGPT language patterns"""
        high_count = sum(1 for verb in self.AI_VERBS["high_confidence"] 
                        if verb in text_lower)
        medium_count = sum(1 for verb in self.AI_VERBS["medium_confidence"] 
                          if verb in text_lower)
        
        total_weighted = high_count * 2 + medium_count
        
        return {
            "high_count": high_count,
            "medium_count": medium_count,
            "total_weighted": total_weighted
        }
    
    def _analyze_unsexy_work(self, text_lower: str, industry: Optional[str]) -> Dict:
        """Check for real unsexy work (industry-aware)"""
        unsexy_found = set()
        
        if industry and industry in self.INDUSTRY_PATTERNS:
            patterns = self.INDUSTRY_PATTERNS[industry]["unsexy_work"]
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    unsexy_found.add(pattern.split('|')[0])
        
        count = len(unsexy_found)
        score = min(12, count * 4)
        
        return {"score": score, "count": count}
    
    def _calculate_authenticity_bonus(self, text: str, text_lower: str) -> Dict:
        """Bonus points for universal authenticity markers"""
        score = 0
        markers = []
        
        # Version numbers
        if re.search(self.AUTHENTICITY_PATTERNS["version_numbers"], text):
            score += 5
            markers.append("✓ Specific versions mentioned")
        
        # Admits struggles/failures
        if re.search(self.AUTHENTICITY_PATTERNS["admits_struggles"], text_lower):
            score += 8
            markers.append("✓ Admits real challenges")
        
        # Contextual metrics
        if re.search(self.AUTHENTICITY_PATTERNS["contextual_metrics"], text_lower):
            score += 6
            markers.append("✓ Contextual metrics (before/after)")
        
        # Check sentence length variance (natural writing)
        bullets = re.findall(r'[•\-\*]\s*(.+)', text)
        if len(bullets) >= 3:
            lengths = [len(b) for b in bullets]
            if max(lengths) - min(lengths) > 50:  # Good variance
                score += 4
                markers.append("✓ Natural writing rhythm")
        
        return {"score": score, "markers": markers}
    
    def _count_vague_metrics(self, text_lower: str) -> int:
        """Count vague percentage claims without context"""
        vague = re.findall(self.VAGUE_METRICS_PATTERN, text_lower)
        contextual = re.findall(r'from\s+.{0,20}to\s+|baseline|previously|was\s+\d+', text_lower)
        return max(0, len(vague) - len(contextual))
    
    def _count_generic_companies(self, text_lower: str) -> int:
        """Count generic company names"""
        count = 0
        for company in self.GENERIC_COMPANIES:
            count += text_lower.count(company)
        return count
    
    def _convert_to_stars(self, raw_score: int) -> Tuple[float, str, str, str]:
        """Convert raw score to 5-star rating system"""
        if raw_score >= 90:
            return (5.0, "⭐⭐⭐⭐⭐", "Elite", 
                   "✅ Elite verified - Fast-track to final interview")
        elif raw_score >= 80:
            return (4.0, "⭐⭐⭐⭐", "Excellent", 
                   "✅ Excellent - Strong proceed with standard process")
        elif raw_score >= 70:
            return (4.0, "⭐⭐⭐⭐", "Strong", 
                   "✅ Strong - Normal screening process")
        elif raw_score >= 55:
            return (3.0, "⭐⭐⭐", "Fair", 
                   "🟡 Fair - Extra verification recommended")
        elif raw_score >= 40:
            return (2.0, "⭐⭐", "Weak", 
                   "⚠️ Weak - Heavy vetting required")
        else:
            return (1.0, "⭐", "High Risk", 
                   "❌ High risk - Multiple red flags")
    
    def _get_industry_question(self, industry: Optional[str], 
                              text: str, confidence: float) -> Optional[str]:
        """Get industry-specific question"""
        
        # Tech domain questions (if available and high confidence)
        if industry == "tech" and confidence >= 0.5 and DOMAIN_MODULES_AVAILABLE:
            if "kubernetes" in text or "terraform" in text:
                return q5_devops(text)
            elif "microservice" in text or "api" in text:
                return q5_backend(text)
            elif "saml" in text or "oauth" in text:
                return q5_iam(text)
            elif "spark" in text or "kafka" in text:
                return q5_data(text)
        
        # Industry-specific screening questions
        if confidence >= 0.4:
            industry_questions = {
                "sales": "Walk me through your most difficult lost deal and what you learned.",
                "marketing": "Describe a campaign that underperformed and how you diagnosed the issue.",
                "finance": "What was your most challenging month-end close and why?",
                "healthcare": "Describe a situation where you had competing patient priorities - how did you handle it?",
                "operations": "Tell me about a time when a supplier let you down - what was your backup plan?",
                "hr": "What's the toughest hiring mistake you made and how did you fix it?"
            }
            return industry_questions.get(industry)
        
        return None
    
    def _determine_ai_probability(self, ai_analysis: Dict, problem_analysis: Dict) -> str:
        """Determine likelihood of AI generation"""
        if ai_analysis["high_count"] >= 3 and problem_analysis["count"] == 0:
            return "Very High (90%+)"
        elif ai_analysis["total_weighted"] >= 4:
            return "High (70-90%)"
        elif ai_analysis["total_weighted"] >= 2:
            return "Moderate (40-70%)"
        else:
            return "Low (<40%)"


# Simple usage example
if __name__ == "__main__":
    engine = UniversalAuthenticityEngine()
    
    # Example tech resume
    tech_resume = """
    Senior DevOps Engineer
    • Resolved production Kubernetes OOM errors by implementing resource limits
    • Debugged Terraform state locking issues during migration
    • Maintained legacy Jenkins pipelines and technical debt cleanup
    """
    
    # Example sales resume
    sales_resume = """
    Account Executive
    • Closed $2.3M in new business, missing Q4 quota by 8%
    • Lost major deal to competitor due to pricing, adjusted strategy
    • Cold called 50+ prospects daily to rebuild pipeline after territory change
    """
    
    result_tech = engine.score_resume(tech_resume, "Tech Candidate")
    result_sales = engine.score_resume(sales_resume, "Sales Candidate")
    
    print(f"\n=== TECH CANDIDATE ===")
    print(f"Rating: {result_tech.star_display} ({result_tech.star_rating}/5)")
    print(f"Tier: {result_tech.tier}")
    print(f"Industry: {result_tech.industry_detected} ({result_tech.industry_confidence:.0%} confidence)")
    print(f"Verdict: {result_tech.verdict}")
    print(f"Red Flags: {result_tech.red_flags}")
    print(f"Authenticity: {result_tech.authenticity_markers}")
    
    print(f"\n=== SALES CANDIDATE ===")
    print(f"Rating: {result_sales.star_display} ({result_sales.star_rating}/5)")
    print(f"Tier: {result_sales.tier}")
    print(f"Industry: {result_sales.industry_detected} ({result_sales.industry_confidence:.0%} confidence)")
    print(f"Verdict: {result_sales.verdict}")
    print(f"Red Flags: {result_sales.red_flags}")
    print(f"Authenticity: {result_sales.authenticity_markers}")
