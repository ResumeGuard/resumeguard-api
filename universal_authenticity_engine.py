"""
ResumeGuard Universal Authenticity Engine - CALIBRATED
Cross-industry resume authenticity scoring with reasonable math

Supports: Tech, Sales, Marketing, Finance, Healthcare, Operations, HR
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class AuthenticityResult:
    """Scoring result with universal industry support"""
    candidate: str
    raw_score: float
    star_rating: int
    star_display: str
    tier: str
    verdict: str
    red_flags: List[str]
    authenticity_markers: List[str]
    has_real_problems: bool
    industry_detected: Optional[str]
    industry_confidence: float
    ai_probability: float
    universal_questions: List[str]
    industry_specific_question: str


class UniversalAuthenticityEngine:
    """
    Cross-industry resume authenticity scoring engine
    CALIBRATED for reasonable human-centric scoring
    """
    
    def __init__(self):
        # Industry detection patterns
        self.industry_patterns = {
            "Technology": [
                r'\b(kubernetes|docker|aws|azure|gcp|terraform|ansible|jenkins|ci/cd|devops|microservices|api|rest|graphql|sql|nosql|python|java|javascript|react|node\.?js|golang|rust|git|agile|scrum)\b',
                r'\b(cloud|infrastructure|deployment|monitoring|logging|observability|sre|site reliability|backend|frontend|full[- ]?stack)\b',
                r'\b(software engineer|devops|platform engineer|sre|backend developer|frontend developer|full stack developer)\b'
            ],
            "Sales": [
                r'\b(quota|pipeline|cold call|outreach|prospecting|closing|bdm|account executive|ae|sdr|bdr|territory|crm|salesforce|hubspot)\b',
                r'\b(deal|contract|revenue|commission|attainment|forecasting|negotiation|enterprise sales|b2b|saas sales)\b',
                r'\b(sales development|business development|account manager|sales manager|revenue|hunter)\b'
            ],
            "Marketing": [
                r'\b(campaign|seo|sem|ppc|content marketing|social media|brand|positioning|messaging|gtm|go[- ]?to[- ]?market)\b',
                r'\b(leads|mql|sql|conversion|funnel|analytics|google analytics|facebook ads|linkedin ads|email marketing|marketing automation)\b',
                r'\b(marketing manager|digital marketing|growth marketing|demand generation|product marketing|brand manager)\b'
            ],
            "Finance": [
                r'\b(financial modeling|valuation|dcf|lbo|m&a|fp&a|budgeting|forecasting|gaap|ifrs|sox|audit)\b',
                r'\b(balance sheet|income statement|cash flow|variance analysis|financial analysis|cost accounting|treasury)\b',
                r'\b(financial analyst|accountant|cpa|controller|finance manager|investment|portfolio)\b'
            ],
            "Healthcare": [
                r'\b(patient|clinical|ehr|emr|epic|cerner|hipaa|medical|nursing|physician|diagnosis|treatment|care plan)\b',
                r'\b(healthcare|hospital|clinic|medical center|pharmacy|medication|prescription|therapy|rehabilitation)\b',
                r'\b(nurse|doctor|physician|medical assistant|healthcare administrator|clinical coordinator)\b'
            ],
            "Operations": [
                r'\b(supply chain|logistics|inventory|procurement|vendor management|process improvement|lean|six sigma|warehouse)\b',
                r'\b(operations|efficiency|productivity|kpi|metrics|sop|standard operating procedure|workflow|automation)\b',
                r'\b(operations manager|project manager|program manager|logistics coordinator|supply chain analyst)\b'
            ],
            "HR": [
                r'\b(recruiting|talent acquisition|onboarding|performance review|compensation|benefits|hris|workday|adp)\b',
                r'\b(employee relations|hr policies|compliance|labor law|diversity|dei|culture|retention|turnover)\b',
                r'\b(hr manager|recruiter|talent|people operations|hr business partner|hr generalist)\b'
            ]
        }
        
        # Problem specificity patterns (UNIVERSAL across industries)
        # More flexible patterns to catch real-world metrics
        self.problem_patterns = [
            # Reductions/decreases
            r'reduc(?:ed|ing)\s+(?:\w+\s+){0,4}by\s+\d+\s*%',
            r'decreas(?:ed|ing)\s+(?:\w+\s+){0,4}by\s+\d+\s*%',
            r'cut\s+(?:\w+\s+){0,4}by\s+\d+\s*%',
            r'lowered\s+(?:\w+\s+){0,4}by\s+\d+\s*%',
            
            # Increases/improvements
            r'increas(?:ed|ing)\s+(?:\w+\s+){0,4}by\s+\d+\s*%',
            r'improv(?:ed|ing)\s+(?:\w+\s+){0,4}by\s+\d+\s*%',
            r'boost(?:ed|ing)\s+(?:\w+\s+){0,4}by\s+\d+\s*%',
            r'grew\s+(?:\w+\s+){0,4}by\s+\d+\s*%',
            r'enhanced\s+(?:\w+\s+){0,4}by\s+\d+\s*%',
            
            # Scaling/from-to metrics
            r'from\s+\d+[%\w]*\s+to\s+\d+[%\w]*',
            r'scaled?\s+\w+\s+from\s+\d+',
            r'migrated\s+\d+\+?\s*[km]?',
            
            # Time improvements
            r'from\s+\d+\s*(?:hours?|minutes?|days?|weeks?|months?)\s+to\s+\d+',
            r'under\s+\d+\s*(?:hours?|minutes?|days?)',
            r'in\s+under\s+\d+',
            
            # Absolute numbers
            r'\d+[km]?\+?\s+(?:users?|customers?|transactions?|requests?|pods?|servers?|gpus?)',
            r'supporting?\s+\d+[km]?\+?',
            r'handling?\s+\d+[km]?\+?',
            
            # SLA/uptime metrics
            r'\d+\.\d+%\s+(?:availability|uptime|sla|slo)',
            r'slo?\s+[≤<]?\s*\d+',
            r'rto\s+[≤<]?\s*\d+',
            r'rpo\s+[≤<]?\s*\d+',
            
            # Financial/revenue
            r'\$\d+[km]?\+?',
            r'€\d+[km]?\+?',
            r'revenue.*?\d+%',
            r'cost.*?\d+%'
        ]
        
        # Unsexy/authentic work patterns (UNIVERSAL)
        self.unsexy_work_patterns = [
            r'\b(debug|troubleshoot|firefight|hotfix|incident|outage|downtime|on[- ]?call|pager)\b',
            r'\b(manual|tedious|repetitive|grunt work|weekend|late night|emergency|urgent)\b',
            r'\b(legacy|technical debt|refactor|cleanup|migration|deprecated)\b',
            r'\b(broke|failed|crashed|bug|error|issue|problem|escalation)\b',
            r'\b(difficult customer|challenging stakeholder|pushback|resistance|conflict)\b',
            r'\b(tight deadline|crunch|pressure|stressful|demanding|difficult period)\b'
        ]
        
        # AI/template red flags - REFINED (only truly suspicious language)
        self.ai_red_flags = [
            # Classic AI power verbs (the really obvious ones)
            r'\b(spearheaded|orchestrated|championed|pioneered|revolutionized|transformed|leveraged|utilized)\b',
            r'\b(facilitated|synergized|catalyzed|actualized|conceptualized|operationalized)\b',
            
            # Buzzword nouns/adjectives
            r'\b(synergy|paradigm|holistic|robust|scalable|innovative|cutting[- ]?edge|state[- ]?of[- ]?the[- ]?art)\b',
            r'\b(comprehensive|strategic|dynamic|mission[- ]?critical|industry[- ]?leading|world[- ]?class)\b',
            r'\b(next[- ]?generation|future[- ]?proof|best[- ]?in[- ]?class|enterprise[- ]?grade)\b',
            
            # Generic cliché phrases
            r'drive results|exceed expectations|think outside the box|hit the ground running',
            r'passion for excellence|passion for|team player|detail[- ]?oriented|fast[- ]?paced environment',
            r'strong communication|proven track record|results[- ]?driven|goal[- ]?oriented',
            r'go[- ]?getter|self[- ]?starter|wear many hats'
        ]
    
    def detect_industry(self, text: str) -> Tuple[Optional[str], float]:
        """Detect primary industry from resume text"""
        text_lower = text.lower()
        scores = {}
        
        for industry, patterns in self.industry_patterns.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                count += len(matches)
            scores[industry] = count
        
        if not scores or max(scores.values()) == 0:
            return None, 0.0
        
        top_industry = max(scores, key=scores.get)
        top_count = scores[top_industry]
        total_count = sum(scores.values())
        
        confidence = min(top_count / max(total_count, 1), 1.0)
        
        return top_industry, confidence
    
    def count_problems(self, text: str) -> int:
        """Count specific quantified problems/achievements"""
        count = 0
        for pattern in self.problem_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            count += len(matches)
        return count
    
    def find_unsexy_work(self, text: str) -> List[str]:
        """Find evidence of authentic, difficult work"""
        markers = []
        text_lower = text.lower()
        
        for pattern in self.unsexy_work_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches[:3]:  # Limit per pattern
                if match not in markers:
                    markers.append(match)
        
        return markers[:5]  # Return top 5
    
    def detect_ai_probability(self, text: str) -> float:
        """
        Estimate AI-generation probability based on:
        1. Buzzwords
        2. Structural patterns (overly uniform formatting)
        3. Lack of authentic voice
        """
        text_lower = text.lower()
        word_count = len(text.split())
        
        if word_count == 0:
            return 0.0
        
        # 1. COUNT BUZZWORDS
        ai_count = 0
        for pattern in self.ai_red_flags:
            matches = re.findall(pattern, text_lower)
            ai_count += len(matches)
        
        buzzword_ratio = ai_count / (word_count / 100)  # Per 100 words
        
        # 2. DETECT FORMULAIC STRUCTURE
        # Count sentences that start with past-tense action verbs (AI pattern)
        action_verbs = [
            'led', 'built', 'created', 'developed', 'implemented', 'designed',
            'managed', 'established', 'delivered', 'executed', 'improved',
            'increased', 'reduced', 'optimized', 'enhanced', 'automated',
            'spearheaded', 'orchestrated', 'pioneered', 'architected',
            'engineered', 'formulated', 'directed', 'conducted', 'performed'
        ]
        
        sentences = re.split(r'[.!?]\s+', text_lower)
        action_starts = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if any(sentence.startswith(verb) for verb in action_verbs):
                action_starts += 1
        
        if len(sentences) > 10:
            action_ratio = action_starts / len(sentences)
            # If >60% of sentences start with action verbs, add to AI probability
            if action_ratio > 0.6:
                formula_penalty = (action_ratio - 0.6) * 0.5  # Up to +0.20
            else:
                formula_penalty = 0
        else:
            formula_penalty = 0
        
        # 3. CHECK FOR OVER-WORDINESS
        # Very long resumes (1200+ words) without personality are suspect
        if word_count > 1200:
            length_penalty = (word_count - 1200) / 2000 * 0.15  # Up to +0.15
        else:
            length_penalty = 0
        
        # COMBINE FACTORS
        total_probability = min(
            (buzzword_ratio * 0.15) + formula_penalty + length_penalty,
            0.95  # Cap at 95%
        )
        
        return total_probability
    
    def generate_universal_questions(self, text: str, problems: int, unsexy: List[str]) -> List[str]:
        """Generate universal interview questions"""
        questions = []
        
        if problems > 0:
            questions.append("Walk me through your biggest measurable impact - what was the before/after and how did you achieve it?")
        
        if unsexy:
            questions.append("Tell me about a time something broke in production or went wrong - what happened and how did you fix it?")
        else:
            questions.append("What's the most frustrating or tedious part of your current role? How do you handle it?")
        
        questions.append("Describe a time you had to deal with ambiguity or incomplete information. How did you move forward?")
        
        return questions
    
    def detect_career_growth(self, text: str) -> Tuple[int, bool]:
        """
        Detect career progression within same company/through acquisition
        Returns: (promotion_count, has_long_tenure)
        
        Strong authenticity signal - hard to fake, shows real value
        SELECTIVE: Only bonuses clear same-company growth
        """
        text_lower = text.lower()
        
        # Look for explicit same-company progression signals
        # Pattern: Same company name with multiple different titles
        
        # 1. EXPLICIT PROMOTION MENTIONS
        promotion_indicators = [
            r'promoted\s+to',
            r'advancement\s+to',
            r'grew\s+(?:career|role)',
            r'progressed\s+to',
            r'elevated\s+to'
        ]
        
        explicit_promotions = 0
        for pattern in promotion_indicators:
            explicit_promotions += len(re.findall(pattern, text_lower))
        
        # 2. ACQUISITION SURVIVAL (strong signal)
        acquisition_mentions = len(re.findall(r'(?:led|through|during|managed).*?acquisition|acquired\s+by', text_lower))
        
        # 3. DETECT MULTIPLE ROLES AT SAME COMPANY
        # Look for patterns like "Company — Role1" then "Company — Role2"
        # This is tricky, so we'll look for repeated company names with different role keywords
        
        # Extract lines that look like job titles
        lines = text.split('\n')
        company_roles = []
        
        for i, line in enumerate(lines):
            # Look for lines with "—" or "," separating company and role
            if '—' in line or ' - ' in line:
                company_roles.append(line.lower().strip())
        
        # Count how many times we see progression keywords WITH same company context
        same_company_progression = 0
        
        # Look for clear patterns of: "senior", "lead", "principal", "manager" appearing multiple times
        # in close proximity (same company section)
        progression_keywords = ['engineer', 'senior', 'lead', 'principal', 'manager', 'director', 'architect']
        
        # Count unique progression levels found
        levels_in_resume = [kw for kw in progression_keywords if kw in text_lower]
        
        # If we see 4+ different role levels + acquisition OR explicit promotion, it's likely progression
        if len(set(levels_in_resume)) >= 4 and (acquisition_mentions > 0 or explicit_promotions > 0):
            same_company_progression = 2
        elif len(set(levels_in_resume)) >= 4:
            same_company_progression = 1
        
        # 4. LONG TENURE DETECTION (10+ years, 15+ years, 20+ years)
        tenure_patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?(?:experience|expertise)',
            r'over\s+(\d+)\s+years?',
            r'(\d+)\+?\s*years?\s+of\s+(?:experience|expertise)'
        ]
        
        max_tenure_mentioned = 0
        for pattern in tenure_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                try:
                    years = int(match)
                    max_tenure_mentioned = max(max_tenure_mentioned, years)
                except:
                    pass
        
        # SCORING
        total_promotion_score = 0
        
        # Explicit promotions: strong signal
        if explicit_promotions > 0:
            total_promotion_score += explicit_promotions
        
        # Acquisition survival: strong signal  
        if acquisition_mentions > 0:
            total_promotion_score += 1
        
        # Same company progression: strong signal
        total_promotion_score += same_company_progression
        
        # Long tenure flag - VERY STRICT
        # Only bonus if explicitly mentioned "20+ years" OR 15+ with acquisition/progression
        has_significant_tenure = False
        
        if max_tenure_mentioned >= 20:
            # Explicitly states 20+ years
            has_significant_tenure = True
        elif max_tenure_mentioned >= 15 and total_promotion_score >= 2:
            # 15+ years AND clear progression/acquisition
            has_significant_tenure = True
        
        return (total_promotion_score, has_significant_tenure)
    
    def generate_industry_question(self, industry: Optional[str], text: str) -> str:
        """Generate industry-specific behavioral question"""
        
        industry_questions = {
            "Technology": "Walk me through a recent outage or production incident - what was your role in the resolution?",
            "Sales": "Tell me about a deal that fell apart late in the process - what happened and what did you learn?",
            "Marketing": "Describe a campaign that underperformed - how did you diagnose the issue and what changes did you make?",
            "Finance": "Tell me about a time your financial model or forecast was significantly off - what happened?",
            "Healthcare": "Describe a difficult patient situation where you had to balance multiple priorities - how did you handle it?",
            "Operations": "Tell me about a process that broke down or caused delays - how did you identify and fix it?",
            "HR": "Describe a difficult employee situation you had to manage - what was the outcome?"
        }
        
        if industry and industry in industry_questions:
            return industry_questions[industry]
        
        return "Tell me about a time you failed at something important - what happened and what did you learn?"
    
    def calculate_score(
        self,
        problems: int,
        unsexy_count: int,
        ai_probability: float,
        word_count: int,
        industry_detected: bool,
        career_growth: int,
        has_long_tenure: bool
    ) -> float:
        """
        CALIBRATED scoring algorithm
        Balanced bonuses with career growth reward
        """
        
        # BASELINE: Start at 55
        score = 55.0
        
        # PROBLEM SPECIFICITY: +3 per problem (strong signal)
        if problems > 0:
            problem_bonus = min(problems * 3.0, 24)  # Cap at +24
            score += problem_bonus
        
        # UNSEXY WORK: +2.5 per marker (authenticity signal)
        if unsexy_count > 0:
            unsexy_bonus = min(unsexy_count * 2.5, 12)  # Cap at +12
            score += unsexy_bonus
        
        # CAREER GROWTH: Bonus for internal progression (selective)
        # Growing within a company = strong authenticity signal
        if career_growth > 0:
            growth_bonus = min(career_growth * 1.0, 5)  # Up to +5
            score += growth_bonus
        
        # LONG TENURE: Bonus for significant staying power (15+ years)
        if has_long_tenure:
            score += 3  # Loyalty/expertise bonus (reduced)
        
        # AI PROBABILITY: SEVERE penalty for template resumes
        # Anyone over 40% AI probability is highly suspect
        if ai_probability > 0.15:
            # Progressive penalty that gets exponentially worse
            excess = ai_probability - 0.15
            if ai_probability > 0.40:
                # Severe penalty for obvious templates (40%+)
                ai_penalty = 20 + (excess * 60)  # Massive hit
            else:
                # Moderate penalty for 15-40%
                ai_penalty = excess * 40
            score -= ai_penalty
        
        # LENGTH PENALTIES: Smart about achievement density
        if word_count < 150:
            score -= 15  # Too brief, lacks substance
        elif word_count > 1000:
            # Long resume is OK if packed with achievements
            # Calculate achievement density
            achievement_density = problems / max(word_count, 1) * 1000  # Problems per 1000 words
            
            if achievement_density >= 8.0:
                # High density (8+ problems per 1000 words) = no penalty
                verbosity_penalty = 0
            elif achievement_density >= 6.0:
                # Medium-high density = small penalty
                verbosity_penalty = (word_count - 1000) / 100 * 2
            else:
                # Low density = harsh penalty (verbose without substance)
                verbosity_penalty = min((word_count - 1000) / 50 * 10, 40)
            
            score -= verbosity_penalty
        
        # INDUSTRY DETECTION: Small bonus for clarity
        if industry_detected:
            score += 5
        
        # FLOOR AND CEILING
        score = max(15, min(100, score))
        
        return round(score, 1)
    
    def score_to_stars(self, score: float) -> Tuple[int, str]:
        """Convert score to star rating"""
        if score >= 95:
            return 5, "★★★★★"
        elif score >= 85:
            return 4, "★★★★☆"
        elif score >= 70:
            return 3, "★★★☆☆"
        elif score >= 55:
            return 2, "★★☆☆☆"
        else:
            return 1, "★☆☆☆☆"
    
    def score_to_tier(self, score: float) -> str:
        """Map score to hiring tier"""
        if score >= 95:
            return "Elite"
        elif score >= 85:
            return "Excellent"
        elif score >= 70:
            return "Strong"
        elif score >= 55:
            return "Fair"
        elif score >= 40:
            return "Weak"
        else:
            return "High Risk"
    
    def generate_verdict(self, score: float, problems: int, unsexy_count: int) -> str:
        """Generate human-readable verdict"""
        if score >= 95:
            return f"Elite candidate with {problems} quantified achievements and clear evidence of real work."
        elif score >= 85:
            return f"Strong candidate with {problems} measurable impacts. Excellent authenticity signals."
        elif score >= 70:
            return f"Solid candidate with {problems} specific achievements. Some authenticity markers present."
        elif score >= 55:
            return f"Fair candidate with {problems} quantified problems. Limited authenticity signals."
        elif score >= 40:
            return f"Weak profile with only {problems} specific achievements. High AI/template probability."
        else:
            return f"High risk candidate. Minimal specificity ({problems} problems) and authenticity signals."
    
    def identify_red_flags(self, text: str, ai_probability: float, problems: int, unsexy_count: int) -> List[str]:
        """Identify resume red flags"""
        flags = []
        
        if ai_probability > 0.5:
            flags.append(f"High AI/template probability ({ai_probability:.0%})")
        
        if problems == 0:
            flags.append("No quantified achievements or measurable impact")
        elif problems == 1:
            flags.append("Only 1 specific, quantified achievement")
        
        if unsexy_count == 0:
            flags.append("No evidence of handling difficult/unglamorous work")
        
        if len(text.split()) < 200:
            flags.append("Resume suspiciously short")
        
        # Check for generic phrases
        generic_count = 0
        generic_phrases = ["team player", "detail oriented", "fast paced", "results driven"]
        for phrase in generic_phrases:
            if phrase in text.lower():
                generic_count += 1
        
        if generic_count >= 3:
            flags.append(f"Generic template language ({generic_count} clichés)")
        
        return flags
    
    def score_resume(self, resume_text: str, candidate_name: str = "Candidate") -> AuthenticityResult:
        """
        Score a resume with CALIBRATED universal engine
        """
        
        # Detect industry
        industry, confidence = self.detect_industry(resume_text)
        
        # Extract signals
        problems = self.count_problems(resume_text)
        unsexy_markers = self.find_unsexy_work(resume_text)
        unsexy_count = len(unsexy_markers)
        ai_probability = self.detect_ai_probability(resume_text)
        word_count = len(resume_text.split())
        
        # Detect career growth (new!)
        career_growth, has_long_tenure = self.detect_career_growth(resume_text)
        
        # Calculate calibrated score
        raw_score = self.calculate_score(
            problems=problems,
            unsexy_count=unsexy_count,
            ai_probability=ai_probability,
            word_count=word_count,
            industry_detected=(industry is not None),
            career_growth=career_growth,
            has_long_tenure=has_long_tenure
        )
        
        # Convert to stars and tier
        stars, star_display = self.score_to_stars(raw_score)
        tier = self.score_to_tier(raw_score)
        
        # Generate outputs
        verdict = self.generate_verdict(raw_score, problems, unsexy_count)
        red_flags = self.identify_red_flags(resume_text, ai_probability, problems, unsexy_count)
        universal_questions = self.generate_universal_questions(resume_text, problems, unsexy_markers)
        industry_question = self.generate_industry_question(industry, resume_text)
        
        return AuthenticityResult(
            candidate=candidate_name,
            raw_score=raw_score,
            star_rating=stars,
            star_display=star_display,
            tier=tier,
            verdict=verdict,
            red_flags=red_flags,
            authenticity_markers=unsexy_markers,
            has_real_problems=(problems > 0),
            industry_detected=industry,
            industry_confidence=confidence if industry else 0.0,
            ai_probability=ai_probability,
            universal_questions=universal_questions,
            industry_specific_question=industry_question
        )


# Quick test if run directly
if __name__ == "__main__":
    engine = UniversalAuthenticityEngine()
    
    test_resume = """
    Senior Software Engineer
    
    Led migration of monolithic app to microservices, reducing deployment time by 60%.
    Debugged critical production issue affecting 10k users during holiday weekend.
    Reduced API latency from 800ms to 120ms through caching improvements.
    
    Skills: Python, Kubernetes, AWS, PostgreSQL
    """
    
    result = engine.score_resume(test_resume, "Test Candidate")
    print(f"\n{'='*60}")
    print(f"Candidate: {result.candidate}")
    print(f"Industry: {result.industry_detected} ({result.industry_confidence:.0%} confidence)")
    print(f"Score: {result.raw_score} {result.star_display}")
    print(f"Tier: {result.tier}")
    print(f"Verdict: {result.verdict}")
    print(f"Red Flags: {', '.join(result.red_flags) if result.red_flags else 'None'}")
    print(f"Authenticity Markers: {', '.join(result.authenticity_markers) if result.authenticity_markers else 'None'}")
    print(f"{'='*60}\n")
