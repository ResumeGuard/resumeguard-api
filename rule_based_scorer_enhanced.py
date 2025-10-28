# rule_based_scorer_enhanced.py
# Enhanced Tier 1 Resume Authenticity Scoring
# Integrates with authenticity_service_det.py Core-4 Engine
#
# Philosophy:
# - Deterministic rule-based detection (no probabilistic AI)
# - Focus on patterns elite recruiters recognize
# - Distinguish AI-polish from AI-fabrication
# - Conservative scoring to avoid false positives

from typing import Dict, List, Tuple, Any, Optional
import re
from collections import Counter
from datetime import datetime

# ==============================================
# TIER 1: HIGH-CONFIDENCE DETECTION PATTERNS
# ==============================================

# ChatGPT signature verbs (from your research)
CHATGPT_VERBS_TIER1 = {
    "spearheaded", "leveraged", "orchestrated", "facilitated", 
    "championed", "pioneered", "synergized"
}

CHATGPT_VERBS_TIER2 = {
    "revolutionized", "transformed", "optimized", "streamlined",
    "enhanced", "drove"
}

# ChatGPT signature phrases
CHATGPT_PHRASES = [
    r"spearheaded\s+initiatives?\s+that",
    r"leveraged\s+cutting-edge",
    r"orchestrated\s+cross-functional",
    r"facilitated\s+seamless",
    r"championed\s+digital\s+transformation",
    r"pioneered\s+innovative",
    r"drove\s+operational\s+excellence",
    r"delivered\s+robust\s+solutions",
    r"implemented\s+best-in-class",
    r"executed\s+strategic\s+initiatives"
]

# Buzzword combinations (red flags when clustered)
BUZZWORD_CLUSTERS = [
    r"synergistic\s+approach",
    r"robust\s+solutions?",
    r"cutting-edge\s+technolog(y|ies)",
    r"world-class\s+outcomes?",
    r"seamless\s+integration",
    r"strategic\s+initiatives?",
    r"operational\s+excellence",
    r"digital\s+transformation",
    r"innovative\s+methodolog(y|ies)",
    r"dynamic\s+environment",
    r"stakeholder\s+engagement",
    r"cross-functional\s+teams?"
]

# Generic company name patterns
GENERIC_COMPANY_PATTERNS = [
    r"tech\s+solutions?",
    r"global\s+(systems?|technologies?|solutions?)",
    r"innovative?\s+(systems?|technologies?|solutions?)",
    r"digital\s+(systems?|technologies?|solutions?)",
    r"advanced\s+(systems?|technologies?|solutions?)",
    r"integrated?\s+(systems?|technologies?|solutions?)",
    r"software\s+solutions?",
    r"consulting\s+group",
    r"international\s+(inc|corp|llc|ltd)",
    r"enterprises?\s+(inc|corp|llc|ltd)"
]

# Vague metric patterns (percentages without context)
VAGUE_METRIC_PATTERNS = [
    r"improved?\s+(?:efficiency|performance|productivity)\s+by\s+\d+%",
    r"increased?\s+(?:efficiency|performance|productivity)\s+by\s+\d+x?",
    r"reduced?\s+costs?\s+by\s+\d+%",
    r"enhanced?\s+(?:efficiency|performance|productivity)\s+by\s+\d+%",
    r"achieved?\s+\d+%\s+(?:efficiency|improvement|increase)"
]

# Round number detection (too many perfect percentages)
ROUND_NUMBER_PATTERN = r"\b(10|20|25|30|40|50|60|75|80|90|100)%"

# Tools/technologies database (simplified - expand as needed)
TECH_RELEASE_DATES = {
    "kubernetes": 2014,
    "docker": 2013,
    "react": 2013,
    "terraform": 2014,
    "aws lambda": 2014,
    "chatgpt": 2022,
    "github copilot": 2021,
    "next.js": 2016,
    "tailwind": 2017,
    "typescript": 2012,
    "graphql": 2015
}

# ==============================================
# HELPER FUNCTIONS
# ==============================================

def tokenize_text(text: str) -> List[str]:
    """Convert text to lowercase word tokens."""
    return re.findall(r'\b\w+\b', text.lower())

def extract_sentences(text: str) -> List[str]:
    """Extract sentences from text."""
    # Split on periods, exclamation marks, question marks, or bullet points
    sentences = re.split(r'[.!?]|\n\s*[-•]\s*', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]

def extract_bullets(text: str) -> List[str]:
    """Extract bullet points from text."""
    bullets = re.findall(r'(?m)^\s*[-•]\s*(.+)$', text)
    return [b.strip() for b in bullets if len(b.strip()) > 10]

def calculate_sentence_variance(sentences: List[str]) -> float:
    """Calculate coefficient of variation for sentence lengths."""
    if len(sentences) < 3:
        return 0.0
    
    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    
    if mean == 0:
        return 0.0
    
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    std_dev = variance ** 0.5
    
    return std_dev / mean

def extract_years_from_role(role_text: str) -> Optional[int]:
    """Extract start year from role text."""
    # Look for patterns like "2020 - 2023" or "Jan 2020"
    year_match = re.search(r'(19|20)\d{2}', role_text)
    if year_match:
        return int(year_match.group(0))
    return None

# ==============================================
# DETECTION FUNCTIONS
# ==============================================

def detect_chatgpt_verbs(text: str) -> Dict[str, Any]:
    """
    Count ChatGPT signature verbs.
    High counts = likely AI-generated.
    """
    tokens = tokenize_text(text)
    
    tier1_count = sum(1 for t in tokens if t in CHATGPT_VERBS_TIER1)
    tier2_count = sum(1 for t in tokens if t in CHATGPT_VERBS_TIER2)
    
    # Count specific worst offenders
    spearheaded_count = tokens.count("spearheaded")
    leveraged_count = tokens.count("leveraged")
    orchestrated_count = tokens.count("orchestrated")
    
    total_chatgpt_verbs = tier1_count + tier2_count
    
    return {
        "tier1_count": tier1_count,
        "tier2_count": tier2_count,
        "spearheaded_count": spearheaded_count,
        "leveraged_count": leveraged_count,
        "orchestrated_count": orchestrated_count,
        "total_count": total_chatgpt_verbs,
        "severity": (
            "high" if tier1_count >= 3 or spearheaded_count >= 2 else
            "medium" if total_chatgpt_verbs >= 5 else
            "low"
        )
    }

def detect_chatgpt_phrases(text: str) -> Dict[str, Any]:
    """
    Detect signature ChatGPT phrase patterns.
    These are dead giveaways of AI generation.
    """
    text_lower = text.lower()
    matches = []
    
    for pattern in CHATGPT_PHRASES:
        found = re.findall(pattern, text_lower)
        if found:
            matches.extend(found)
    
    return {
        "phrase_count": len(matches),
        "matches": matches[:5],  # First 5 for logging
        "severity": (
            "high" if len(matches) >= 3 else
            "medium" if len(matches) >= 1 else
            "low"
        )
    }

def detect_buzzword_clusters(text: str) -> Dict[str, Any]:
    """
    Detect clusters of corporate buzzwords.
    Multiple occurrences = red flag.
    """
    text_lower = text.lower()
    matches = []
    
    for pattern in BUZZWORD_CLUSTERS:
        found = re.findall(pattern, text_lower)
        if found:
            matches.extend(found)
    
    return {
        "cluster_count": len(matches),
        "matches": matches[:5],
        "severity": (
            "high" if len(matches) >= 5 else
            "medium" if len(matches) >= 3 else
            "low"
        )
    }

def detect_repetitive_structure(text: str) -> Dict[str, Any]:
    """
    Detect repetitive sentence/bullet structure.
    All bullets starting the same way = AI pattern.
    """
    bullets = extract_bullets(text)
    
    if len(bullets) < 3:
        return {"is_repetitive": False, "severity": "low"}
    
    # Extract first 2 words from each bullet
    starters = []
    for bullet in bullets:
        words = bullet.split()
        if len(words) >= 2:
            starters.append(f"{words[0]} {words[1]}".lower())
    
    if not starters:
        return {"is_repetitive": False, "severity": "low"}
    
    # Count most common starter
    starter_counts = Counter(starters)
    most_common = starter_counts.most_common(1)[0]
    repetition_rate = most_common[1] / len(starters)
    
    return {
        "is_repetitive": repetition_rate >= 0.4,  # 40%+ of bullets start same way
        "repetition_rate": round(repetition_rate, 3),
        "most_common_starter": most_common[0],
        "severity": (
            "high" if repetition_rate >= 0.6 else
            "medium" if repetition_rate >= 0.4 else
            "low"
        )
    }

def detect_uniform_bullet_length(text: str) -> Dict[str, Any]:
    """
    Detect if all bullets are suspiciously similar length.
    ChatGPT loves consistency.
    """
    bullets = extract_bullets(text)
    
    if len(bullets) < 3:
        return {"is_uniform": False, "severity": "low"}
    
    lengths = [len(b.split()) for b in bullets]
    mean_length = sum(lengths) / len(lengths)
    
    # Calculate variance
    variance = sum((l - mean_length) ** 2 for l in lengths) / len(lengths)
    std_dev = variance ** 0.5
    
    # Coefficient of variation
    cv = std_dev / mean_length if mean_length > 0 else 0
    
    # CV < 0.25 = too uniform (from your research)
    return {
        "is_uniform": cv < 0.25,
        "coefficient_of_variation": round(cv, 3),
        "mean_length": round(mean_length, 1),
        "severity": (
            "high" if cv < 0.20 else
            "medium" if cv < 0.25 else
            "low"
        )
    }

def detect_vague_metrics(text: str) -> Dict[str, Any]:
    """
    Detect metrics without context (baselines, endpoints).
    Real metrics show from X to Y.
    """
    text_lower = text.lower()
    vague_matches = []
    
    for pattern in VAGUE_METRIC_PATTERNS:
        found = re.findall(pattern, text_lower)
        if found:
            vague_matches.extend(found)
    
    # Check for round numbers (too many perfect percentages)
    round_numbers = re.findall(ROUND_NUMBER_PATTERN, text)
    round_number_count = len(round_numbers)
    
    # Check for baseline → endpoint patterns (good sign)
    baseline_pattern = r"from\s+\$?\d+[\w\s]*\s+to\s+\$?\d+|reduced?\s+\$?\d+[\w\s]*\s+to\s+\$?\d+"
    has_baselines = len(re.findall(baseline_pattern, text_lower)) > 0
    
    return {
        "vague_metric_count": len(vague_matches),
        "round_number_count": round_number_count,
        "has_baseline_metrics": has_baselines,
        "severity": (
            "high" if len(vague_matches) >= 3 and not has_baselines else
            "medium" if len(vague_matches) >= 2 and round_number_count >= 4 else
            "low"
        )
    }

def detect_generic_companies(text: str) -> Dict[str, Any]:
    """
    Detect generic/vague company names.
    Real companies have specific names.
    """
    text_lower = text.lower()
    matches = []
    
    for pattern in GENERIC_COMPANY_PATTERNS:
        found = re.findall(pattern, text_lower)
        if found:
            matches.extend(found)
    
    return {
        "generic_company_count": len(matches),
        "matches": matches[:3],
        "severity": (
            "high" if len(matches) >= 2 else
            "medium" if len(matches) >= 1 else
            "low"
        )
    }

def detect_tools_predate_availability(roles: List[Dict], tech_db: Dict[str, int] = TECH_RELEASE_DATES) -> Dict[str, Any]:
    """
    Check if resume claims using tools before they existed.
    Requires parsed roles with dates and descriptions.
    """
    violations = []
    
    for role in roles:
        # Extract year from role
        year = None
        if "start" in role:
            year_match = re.search(r'(19|20)\d{2}', str(role["start"]))
            if year_match:
                year = int(year_match.group(0))
        
        if not year:
            continue
        
        # Check description for tools
        description = role.get("description", "").lower()
        
        for tool, release_year in tech_db.items():
            if tool in description and year < release_year:
                violations.append({
                    "tool": tool,
                    "claimed_year": year,
                    "release_year": release_year,
                    "company": role.get("company", "Unknown")
                })
    
    return {
        "violation_count": len(violations),
        "violations": violations[:3],
        "severity": (
            "high" if len(violations) >= 2 else
            "medium" if len(violations) >= 1 else
            "low"
        )
    }

def detect_excessive_tool_count(text: str) -> Dict[str, Any]:
    """
    Count total tools/technologies listed.
    >20 tools is often resume padding.
    """
    # Look for skills section
    skills_match = re.search(r'(?i)(skills?|technologies?|technical|tools?)[\s:]+(.+?)(?=\n\n|\n[A-Z]|\Z)', text, re.DOTALL)
    
    if not skills_match:
        return {"tool_count": 0, "severity": "low"}
    
    skills_text = skills_match.group(2)
    
    # Count comma-separated items and words in title case (likely tech names)
    items = re.split(r'[,;•|]', skills_text)
    tool_count = len([item.strip() for item in items if len(item.strip()) > 2])
    
    return {
        "tool_count": tool_count,
        "severity": (
            "high" if tool_count > 30 else
            "medium" if tool_count > 20 else
            "low"
        )
    }

def detect_no_specific_context(text: str) -> Dict[str, Any]:
    """
    Check if resume lacks specific details:
    - No project names
    - No company-specific details
    - All generic descriptions
    """
    bullets = extract_bullets(text)
    
    if len(bullets) < 3:
        return {"lacks_context": False, "severity": "low"}
    
    # Check for specificity indicators
    has_project_names = bool(re.search(r'project\s+[\w-]+|system\s+[\w-]+|\w+\s+platform', text, re.I))
    has_product_names = bool(re.search(r'(developed?|built?|created?)\s+[\w-]+(?:\s+[\w-]+){0,2}(?:\s+for|\s+to|\s+that)', text, re.I))
    has_specific_metrics = bool(re.search(r'from\s+\$?\d+.*to\s+\$?\d+|\d+\+?\s+(users?|customers?|services?|applications?)', text, re.I))
    
    generic_count = sum([
        not has_project_names,
        not has_product_names,
        not has_specific_metrics
    ])
    
    return {
        "lacks_context": generic_count >= 2,
        "has_project_names": has_project_names,
        "has_product_names": has_product_names,
        "has_specific_metrics": has_specific_metrics,
        "severity": (
            "high" if generic_count == 3 else
            "medium" if generic_count == 2 else
            "low"
        )
    }

def detect_perfect_grammar(text: str) -> Dict[str, Any]:
    """
    Check for unnaturally perfect grammar.
    Real resumes have minor imperfections.
    """
    # Check for contractions (humans use them sometimes)
    has_contractions = bool(re.search(r"\w+\'(t|s|re|ve|ll|d)\b", text))
    
    # Check for casual language
    casual_patterns = [r'\bbuilt?\b', r'\bgot\b', r'\bhelped?\b', r'\bworked? on\b']
    has_casual = any(re.search(pattern, text, re.I) for pattern in casual_patterns)
    
    # Check for sentence fragments (natural in bullets)
    sentences = extract_sentences(text)
    has_fragments = any(len(s.split()) < 5 for s in sentences[:5])
    
    perfection_score = sum([
        not has_contractions,
        not has_casual,
        not has_fragments
    ])
    
    return {
        "is_too_perfect": perfection_score == 3,
        "has_contractions": has_contractions,
        "has_casual_language": has_casual,
        "severity": (
            "medium" if perfection_score == 3 else
            "low"
        )
    }

# ==============================================
# MAIN EVALUATION FUNCTION
# ==============================================

def evaluate_resume_enhanced(resume_text: str, roles: Optional[List[Dict]] = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Enhanced Tier 1 resume authenticity evaluation.
    
    Args:
        resume_text: Full resume text
        roles: Optional list of parsed roles with dates (for timeline checks)
        verbose: Return detailed breakdown
    
    Returns:
        Dictionary with score and flags
    """
    
    base_score = 100
    deductions = []
    bonuses = []
    
    # Run all detection functions
    chatgpt_verbs = detect_chatgpt_verbs(resume_text)
    chatgpt_phrases = detect_chatgpt_phrases(resume_text)
    buzzword_clusters = detect_buzzword_clusters(resume_text)
    repetitive_structure = detect_repetitive_structure(resume_text)
    uniform_bullets = detect_uniform_bullet_length(resume_text)
    vague_metrics = detect_vague_metrics(resume_text)
    generic_companies = detect_generic_companies(resume_text)
    excessive_tools = detect_excessive_tool_count(resume_text)
    lacks_context = detect_no_specific_context(resume_text)
    perfect_grammar = detect_perfect_grammar(resume_text)
    
    # Timeline-based checks (if roles provided)
    if roles:
        tools_predate = detect_tools_predate_availability(roles)
    else:
        tools_predate = {"violation_count": 0, "severity": "low"}
    
    # ========================================
    # CATEGORY 1: LANGUAGE/TONE (max -25)
    # ========================================
    language_deductions = 0
    
    # ChatGPT verbs (strongest signal)
    if chatgpt_verbs["spearheaded_count"] >= 2:
        language_deductions += 10
        deductions.append(("language", f"'Spearheaded' used {chatgpt_verbs['spearheaded_count']}x (AI signature)", -10))
    elif chatgpt_verbs["tier1_count"] >= 3:
        language_deductions += 8
        deductions.append(("language", f"{chatgpt_verbs['tier1_count']} Tier-1 ChatGPT verbs detected", -8))
    elif chatgpt_verbs["total_count"] >= 5:
        language_deductions += 5
        deductions.append(("language", f"{chatgpt_verbs['total_count']} AI-style verbs detected", -5))
    
    # ChatGPT phrases
    if chatgpt_phrases["severity"] == "high":
        language_deductions += 8
        deductions.append(("language", f"{chatgpt_phrases['phrase_count']} signature ChatGPT phrases found", -8))
    elif chatgpt_phrases["severity"] == "medium":
        language_deductions += 4
        deductions.append(("language", "ChatGPT-style phrasing detected", -4))
    
    # Buzzword clusters
    if buzzword_clusters["severity"] == "high":
        language_deductions += 6
        deductions.append(("language", f"{buzzword_clusters['cluster_count']} buzzword clusters", -6))
    elif buzzword_clusters["severity"] == "medium":
        language_deductions += 3
        deductions.append(("language", "Moderate buzzword usage", -3))
    
    # Perfect grammar (minor flag)
    if perfect_grammar["is_too_perfect"]:
        language_deductions += 3
        deductions.append(("language", "Unnaturally perfect grammar/tone", -3))
    
    language_deductions = min(language_deductions, 25)
    
    # ========================================
    # CATEGORY 2: STRUCTURE (max -15)
    # ========================================
    structure_deductions = 0
    
    # Repetitive structure
    if repetitive_structure["severity"] == "high":
        structure_deductions += 8
        deductions.append(("structure", f"Highly repetitive bullet structure ({repetitive_structure['repetition_rate']:.0%})", -8))
    elif repetitive_structure["severity"] == "medium":
        structure_deductions += 4
        deductions.append(("structure", "Repetitive bullet patterns detected", -4))
    
    # Uniform bullet length
    if uniform_bullets["severity"] == "high":
        structure_deductions += 7
        deductions.append(("structure", f"Suspiciously uniform bullets (CV={uniform_bullets['coefficient_of_variation']:.2f})", -7))
    elif uniform_bullets["severity"] == "medium":
        structure_deductions += 4
        deductions.append(("structure", "Low sentence variance (AI-like)", -4))
    
    structure_deductions = min(structure_deductions, 15)
    
    # ========================================
    # CATEGORY 3: SPECIFICITY (max -20)
    # ========================================
    specificity_deductions = 0
    
    # Vague metrics
    if vague_metrics["severity"] == "high":
        specificity_deductions += 10
        deductions.append(("specificity", f"{vague_metrics['vague_metric_count']} metrics lack context/baselines", -10))
    elif vague_metrics["severity"] == "medium":
        specificity_deductions += 5
        deductions.append(("specificity", "Several vague or unverifiable metrics", -5))
    
    # Generic companies
    if generic_companies["severity"] == "high":
        specificity_deductions += 8
        deductions.append(("specificity", f"{generic_companies['generic_company_count']} generic company names", -8))
    elif generic_companies["severity"] == "medium":
        specificity_deductions += 4
        deductions.append(("specificity", "Vague/generic company names detected", -4))
    
    # Lacks specific context
    if lacks_context["severity"] == "high":
        specificity_deductions += 7
        deductions.append(("specificity", "No specific projects, products, or measurable details", -7))
    elif lacks_context["severity"] == "medium":
        specificity_deductions += 4
        deductions.append(("specificity", "Limited specific project/product details", -4))
    
    specificity_deductions = min(specificity_deductions, 20)
    
    # ========================================
    # CATEGORY 4: TECHNICAL COHERENCE (max -15)
    # ========================================
    technical_deductions = 0
    
    # Tools predate availability
    if tools_predate["severity"] == "high":
        technical_deductions += 10
        deductions.append(("technical", f"{tools_predate['violation_count']} tools used before release dates", -10))
    elif tools_predate["severity"] == "medium":
        technical_deductions += 6
        deductions.append(("technical", "Tool usage predates availability", -6))
    
    # Excessive tool count
    if excessive_tools["severity"] == "high":
        technical_deductions += 8
        deductions.append(("technical", f"{excessive_tools['tool_count']} tools listed (likely padding)", -8))
    elif excessive_tools["severity"] == "medium":
        technical_deductions += 4
        deductions.append(("technical", f"{excessive_tools['tool_count']} tools (borderline excessive)", -4))
    
    technical_deductions = min(technical_deductions, 15)
    
    # ========================================
    # BONUSES (max +15)
    # ========================================
    authenticity_bonus = 0
    
    # Has baseline metrics (good sign!)
    if vague_metrics["has_baseline_metrics"]:
        authenticity_bonus += 5
        bonuses.append(("authenticity", "Contains baseline → endpoint metrics", +5))
    
    # Has specific project context
    if lacks_context.get("has_project_names", False):
    authenticity_bonus += 3
    bonuses.append((
        "authenticity",
        "Names specific projects/products",
        +3
    ))
    
    # Has casual/natural language
    if perfect_grammar["has_casual_language"]:
        authenticity_bonus += 2
        bonuses.append(("authenticity", "Uses natural, casual language", +2))
    
    # Low ChatGPT verb count (human-like)
    if chatgpt_verbs["total_count"] <= 1:
        authenticity_bonus += 5
        bonuses.append(("authenticity", "Minimal AI-style vocabulary", +5))
    
    authenticity_bonus = min(authenticity_bonus, 15)
    
    # ========================================
    # FINAL CALCULATION
    # ========================================
    total_deductions = (
        language_deductions +
        structure_deductions +
        specificity_deductions +
        technical_deductions
    )
    
    final_score = base_score - total_deductions + authenticity_bonus
    final_score = max(min(final_score, 100), 0)
    
    # Determine risk bucket
    if final_score >= 85:
        bucket = "authentic"
        confidence = "high"
    elif final_score >= 70:
        bucket = "likely_authentic"
        confidence = "medium"
    elif final_score >= 55:
        bucket = "mixed_signals"
        confidence = "medium"
    elif final_score >= 40:
        bucket = "likely_fabricated"
        confidence = "high"
    else:
        bucket = "high_risk"
        confidence = "high"
    
    # Build result
    result = {
        "final_score": final_score,
        "bucket": bucket,
        "confidence": confidence,
        "total_deductions": total_deductions,
        "total_bonuses": authenticity_bonus
    }
    
    if verbose:
        result["breakdown"] = {
            "deductions": deductions,
            "bonuses": bonuses,
            "category_scores": {
                "language": language_deductions,
                "structure": structure_deductions,
                "specificity": specificity_deductions,
                "technical": technical_deductions
            },
            "detection_details": {
                "chatgpt_verbs": chatgpt_verbs,
                "chatgpt_phrases": chatgpt_phrases,
                "buzzword_clusters": buzzword_clusters,
                "repetitive_structure": repetitive_structure,
                "uniform_bullets": uniform_bullets,
                "vague_metrics": vague_metrics,
                "generic_companies": generic_companies,
                "excessive_tools": excessive_tools,
                "lacks_context": lacks_context,
                "perfect_grammar": perfect_grammar,
                "tools_predate": tools_predate
            }
        }
    
    return result


# ==============================================
# BACKWARD COMPATIBILITY WITH OLD FUNCTION
# ==============================================

def evaluate_resume(flags: dict, verbose: bool = False) -> dict:
    """
    Original function signature for backward compatibility.
    This maintains your existing integration.
    """
    base_score = 100
    log = []

    # Timeline Category (max -20)
    timeline_deductions = 0
    if flags.get("repeated_24mo_roles"):
        timeline_deductions += 10
        log.append(("timeline", "Repeated 24-month roles", -10))
    if flags.get("future_dated_roles"):
        timeline_deductions += 7
        log.append(("timeline", "Future-dated roles", -7))
    if flags.get("implausible_continuity"):
        timeline_deductions += 4
        log.append(("timeline", "No career gap across long time", -4))
    if flags.get("tools_predate_availability"):
        timeline_deductions += 6
        log.append(("timeline", "Used tools before public release", -6))
    timeline_deductions = min(timeline_deductions, 20)

    # Tool Stack Category (max -20)
    tool_deductions = 0
    if flags.get("tool_count_excessive"):
        tool_deductions += 7
        log.append(("tools", "Tool list is overinflated (>20)", -7))
    if flags.get("tools_unrelated"):
        tool_deductions += 5
        log.append(("tools", "Unrelated tools listed", -5))
    if flags.get("tools_not_used_in_context"):
        tool_deductions += 5
        log.append(("tools", "Tools not tied to any job/project", -5))
    tool_deductions = min(tool_deductions, 20)

    # Tone/Language Category (max -10)
    tone_deductions = 0
    if flags.get("buzzword_count_high"):
        tone_deductions += 5
        log.append(("tone", "Buzzword-heavy phrasing", -5))
    if flags.get("repetitive_structure"):
        tone_deductions += 4
        log.append(("tone", "Repetitive structure / cadence", -4))
    if flags.get("perfect_sentence_tone"):
        tone_deductions += 3
        log.append(("tone", "Over-sanitized/LLM-like tone", -3))
    tone_deductions = min(tone_deductions, 10)

    # Digital Signals (max -15)
    digital_deductions = 0
    if flags.get("linkedin_missing"):
        digital_deductions += 5
        log.append(("digital", "Missing LinkedIn or web signal", -5))
    if flags.get("voip_phone_number"):
        digital_deductions += 3
        log.append(("digital", "VOIP or masked phone number", -3))
    if flags.get("resume_farm_email"):
        digital_deductions += 5
        log.append(("digital", "Email from known resume farm", -5))
    digital_deductions = min(digital_deductions, 15)

    # Detail Adjustments
    detail_score = 0
    if flags.get("specific_metrics_present"):
        detail_score += 5
        log.append(("detail", "Uses project-specific metrics", +5))
    if flags.get("realistic_tech_usage"):
        detail_score += 5
        log.append(("detail", "Tech stack matches known use cases", +5))
    if flags.get("generic_bullet_points"):
        detail_score -= 5
        log.append(("detail", "Generic, vague bullet points", -5))
    if flags.get("no_tangible_impact"):
        detail_score -= 5
        log.append(("detail", "No measurable results listed", -5))
    detail_score = max(min(detail_score, 10), -10)

    # Education/Location
    edu_loc_score = 0
    if flags.get("vague_education"):
        edu_loc_score -= 3
        log.append(("edu/location", "Vague or missing education detail", -3))
    if flags.get("phone_location_mismatch"):
        edu_loc_score -= 2
        log.append(("edu/location", "Phone area code mismatch", -2))
    if flags.get("education_timeline_verified"):
        edu_loc_score += 3
        log.append(("edu/location", "Education timeline matches job history", +3))
    edu_loc_score = max(min(edu_loc_score, 5), -5)

    # Final score calculation
    deductions_total = (
        timeline_deductions +
        tool_deductions +
        tone_deductions +
        digital_deductions
    )
    final_score = base_score - deductions_total + detail_score + edu_loc_score
    final_score = max(min(final_score, 100), 0)

    if verbose:
        return {
            "final_score": final_score,
            "deductions_total": deductions_total,
            "breakdown": log
        }
    return {"final_score": final_score}
