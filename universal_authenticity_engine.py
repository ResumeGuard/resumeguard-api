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
        PRODUCTION SCORING ALGORITHM
        
        Philosophy: Values what 20-year recruiters trust
        - Authenticity signals (hard to fake) > BS metrics (easy to fake)
        - Career progression (extremely hard to fake)
        - Template language (instant red flag)
        - Achievement density (substance over fluff)
        """
        
        # BASELINE: Start at 55
        score = 55.0
        
        # QUANTIFIED PROBLEMS: +2 per problem (cap +20)
        # Reduced weight - metrics are easy to fabricate ("99.99% uptime", "$500K saved")
        if problems > 0:
            problem_bonus = min(problems * 2.0, 20)  # Cap at +20
            score += problem_bonus
        
        # AUTHENTICITY MARKERS: +3.5 per marker (cap +14)
        # INCREASED weight - unsexy work is hard to fake
        # (incidents, debugging, legacy, manual work, production issues)
        if unsexy_count > 0:
            unsexy_bonus = min(unsexy_count * 3.5, 14)  # Cap at +14
            score += unsexy_bonus
        
        # CAREER GROWTH: +1 per growth point (cap +5)
        # Internal promotions, acquisition survival - extremely hard to fake
        if career_growth > 0:
            growth_bonus = min(career_growth * 1.0, 5)
            score += growth_bonus
        
        # LONG TENURE BONUS: +3 for significant staying power
        # 20+ years OR 15+ with progression - validates expertise
        if has_long_tenure:
            score += 3
        
        # AI PROBABILITY: SEVERE penalty for template resumes
        # Heavy buzzwords = trying to game ATS systems
        if ai_probability > 0.15:
            excess = ai_probability - 0.15
            if ai_probability > 0.40:
                # Obvious template (40%+) - massive penalty
                ai_penalty = 20 + (excess * 60)
            else:
                # Moderate template (15-40%) - significant penalty
                ai_penalty = excess * 40
            score -= ai_penalty
        
        # ACHIEVEMENT DENSITY: Smart penalty for verbose padding
        # Long resumes OK if packed with substance
        if word_count > 1000:
            achievement_density = problems / max(word_count, 1) * 1000
            
            if achievement_density >= 8.0:
                # High density (8+ problems/1000 words) = substantial content
                verbosity_penalty = 0
            elif achievement_density >= 6.0:
                # Medium density = slight concern
                verbosity_penalty = (word_count - 1000) / 100 * 2
            else:
                # Low density = padding with fluff
                verbosity_penalty = min((word_count - 1000) / 50 * 10, 40)
            
            score -= verbosity_penalty
        
        # LENGTH PENALTIES: Too brief or absurdly long
        if word_count < 150:
            score -= 15  # Insufficient detail
        
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
    
    def generate_verdict(self, score: float, problems: int, unsexy_count: int, ai_prob: float, career_growth: int, has_tenure: bool, word_count: int = 0) -> str:
        """Generate specific, actionable verdict with concrete details"""
        
        # Calculate achievement density
        achievement_density = (problems / max(word_count, 1) * 1000) if word_count > 0 else 0
        
        # Build specific details list
        details = []
        
        # Problems/metrics
        if problems >= 10:
            details.append(f"{problems} quantified metrics")
        elif problems >= 5:
            details.append(f"{problems} measurable achievements")
        elif problems >= 2:
            details.append(f"{problems} specific metrics")
        elif problems == 1:
            details.append("only 1 quantified achievement")
        else:
            details.append("no quantified achievements")
        
        # Authenticity signals
        if unsexy_count >= 3:
            details.append("clear evidence of real work (incidents, legacy, debugging)")
        elif unsexy_count >= 1:
            details.append("some authenticity markers")
        else:
            details.append("no evidence of difficult/unglamorous work")
        
        # Career growth
        if career_growth >= 3 and has_tenure:
            details.append("strong career progression at one company (15+ years)")
        elif career_growth >= 2:
            details.append("internal promotions detected")
        
        # Verbosity flag
        is_verbose = word_count > 1000 and achievement_density < 6.0
        if is_verbose:
            details.append(f"verbose ({word_count} words, low density)")
        
        # AI concerns
        if ai_prob >= 0.40:
            details.append(f"MAJOR RED FLAG: {ai_prob:.0%} AI-templated")
        elif ai_prob >= 0.15:
            details.append(f"{ai_prob:.0%} template language")
        
        # Generate verdict based on score
        if score >= 95:
            return f"Elite candidate: {', '.join(details)}. Fast-track to hiring manager."
        
        elif score >= 85:
            action = "Skip phone screen → technical interview"
            if ai_prob > 0.10:
                action = "Proceed to tech interview but verify claims carefully"
            return f"Excellent: {', '.join(details)}. {action}."
        
        elif score >= 70:
            concerns = []
            if problems < 5:
                concerns.append("ask for 3-5 specific quantified examples")
            if unsexy_count == 0:
                concerns.append("probe on handling production issues")
            if ai_prob > 0.15:
                concerns.append("verify technical depth")
            
            action = " - " + ", ".join(concerns) if concerns else ""
            return f"Solid candidate: {', '.join(details)}{action}."
        
        elif score >= 55:
            issues = []
            if problems <= 1:
                issues.append(f"only {problems} metric")
            if is_verbose:
                issues.append("resume padded with fluff")
            if ai_prob > 0.10:
                issues.append(f"template language ({ai_prob:.0%})")
            if unsexy_count == 0:
                issues.append("no proof of real work")
            
            concern_text = " Concerns: " + ", ".join(issues) if issues else ""
            return f"Fair but risky: {', '.join(details)}.{concern_text}. Technical assessment required."
        
        elif score >= 40:
            red_flags = []
            if is_verbose:
                red_flags.append("padded resume (low achievement density)")
            if problems <= 3:
                red_flags.append(f"only {problems} metrics for {word_count} words")
            if ai_prob >= 0.08:
                red_flags.append(f"{ai_prob:.0%} buzzword density")
            if unsexy_count == 0:
                red_flags.append("no authenticity markers")
            
            flag_text = "; ".join(red_flags) if red_flags else "multiple concerns"
            return f"High risk: {', '.join(details)}. Red flags: {flag_text}. Only proceed if desperate."
        
        else:
            return f"Reject: {', '.join(details)}. Too many red flags."
    
    def detect_fraud_signals(self, text: str, word_count: int) -> List[str]:
        """
        Detect fraud patterns that experienced recruiters spot instantly
        Based on 20+ years of recruiting experience
        
        Returns list of specific red flags (not score penalties, just warnings)
        """
        fraud_flags = []
        text_lower = text.lower()
        
        # 1. MISSING SCHOOL NAME
        # "Bachelor of Technology" with no university = suspicious
        education_keywords = ['bachelor', 'master', 'mba', 'degree', 'b.s.', 'm.s.', 'b.tech', 'm.tech']
        university_keywords = ['university', 'college', 'institute', 'school']
        
        has_degree = any(keyword in text_lower for keyword in education_keywords)
        has_institution = any(keyword in text_lower for keyword in university_keywords)
        
        if has_degree and not has_institution:
            fraud_flags.append("No school/university listed for degree (suspicious)")
        
        # 2. EXCESSIVE BULLET POINTS
        # Count bullet patterns (•, *, -, numbers at line start)
        bullet_patterns = [
            r'^\s*[•\*\-]\s',  # Bullet markers
            r'^\s*\d+\.',       # Numbered lists
        ]
        
        lines = text.split('\n')
        bullet_count = 0
        for line in lines:
            for pattern in bullet_patterns:
                if re.match(pattern, line.strip()):
                    bullet_count += 1
                    break
        
        # Estimate job count (very rough - look for date patterns)
        job_dates = re.findall(r'\d{4}\s*[-–]\s*\d{4}|\d{4}\s*[-–]\s*(?:current|present)', text_lower)
        estimated_jobs = max(len(job_dates), 1)
        
        bullets_per_job = bullet_count / estimated_jobs if estimated_jobs > 0 else 0
        
        if bullets_per_job > 15:
            fraud_flags.append(f"Excessive bullets ({bullet_count} bullets for ~{estimated_jobs} jobs = keyword stuffing?)")
        elif bullet_count > 25:
            fraud_flags.append(f"Extremely high bullet count ({bullet_count} total - padding resume?)")
        
        # 3. GENERIC EMAIL PATTERN
        # Extract email from text
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        
        if emails:
            email = emails[0].lower()
            # Check for patterns like: firstname123@, name2025@, randomnumber@
            if re.search(r'[a-z]+\d{1,4}@', email):
                fraud_flags.append(f"Generic email pattern ({email.split('@')[0]}@ - employer-issued?)")
        
        # 4. SUSPICIOUS TENURE PATTERN (H1B gaming)
        # Look for multiple jobs with 23-25 month durations
        # Extract all durations in months (rough heuristic)
        duration_patterns = re.findall(r'(\d{1,2})\s*(?:months?|mos)', text_lower)
        
        if len(duration_patterns) >= 2:
            # Check if multiple jobs are 23-25 months
            visa_gaming_count = sum(1 for m in duration_patterns if 23 <= int(m) <= 25)
            if visa_gaming_count >= 2:
                fraud_flags.append("Multiple 23-25 month tenures (possible visa gaming pattern)")
        
        return fraud_flags
    
    def identify_red_flags(self, text: str, ai_probability: float, problems: int, unsexy_count: int) -> List[str]:
        """Identify resume red flags - both scoring issues and fraud signals"""
        flags = []
        
        # SCORING RED FLAGS (affect trust score)
        if ai_probability > 0.5:
            flags.append(f"Very high AI/template probability ({ai_probability:.0%})")
        elif ai_probability > 0.3:
            flags.append(f"High AI/template probability ({ai_probability:.0%})")
        
        if problems == 0:
            flags.append("No quantified achievements or measurable impact")
        elif problems == 1:
            flags.append("Only 1 specific, quantified achievement")
        
        if unsexy_count == 0:
            flags.append("No evidence of handling difficult/unglamorous work")
        
        # FRAUD SIGNALS (recruiter red flags - not score penalties)
        fraud_signals = self.detect_fraud_signals(text, len(text.split()))
        flags.extend(fraud_signals)
        
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
        
        # Generate outputs with all context including word count
        verdict = self.generate_verdict(raw_score, problems, unsexy_count, ai_probability, career_growth, has_long_tenure, word_count)
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
