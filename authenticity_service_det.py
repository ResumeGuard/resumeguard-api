# authenticity_service_det.py
# Core-4 authenticity scorer with DETERMINISTIC mode:
# - Canonicalize text (paste vs file -> identical)
# - as_of_date reference clock (no day-to-day drift)
# - Content hashing + version stamp
# - In-memory cache keyed by (content_hash, as_of_month, versions)
#
# Run:
#   pip install fastapi uvicorn pydantic
#   uvicorn authenticity_service_det:app --reload
#
# Endpoints:
#   POST /score/authenticity  (recommended)
#   POST /score/cadence
#   POST /score/timeline
#   POST /score/skill_timeline
#   POST /score/title_progression

from __future__ import annotations
from dataclasses import dataclass
from datetime import date, timedelta, datetime
from typing import Dict, Any, List, Optional, Literal, Tuple
import re, math, calendar, hashlib
from collections import Counter
try:
    from .rule_based_scorer import evaluate_resume
except ImportError:
    from rule_based_scorer import evaluate_resume  # fallback for some environments

# =========================
# ===== VERSION STAMP =====
# =========================

VERSIONS = {
    "core": "authenticity_core4_v1.1_det",
    "cadence": "cadence_v1.0",
    "timeline": "timeline_v1.0",
    "skill_timeline": "skill_timeline_v1.0",
    "title_progression": "title_progression_v1.0",
    "weights": "W_0.35_0.25_0.25_0.15",
    "tech_baseline": "devops_2025-10-01",
}
RULESET_HASH = "rsh_6b0d3"  # bump if you change rules materially

# =============================
# ===== DETERMINISM LAYER =====
# =============================

# Canonicalization: make paste vs file identical
UNICODE_MAP = {
    "\u2018":"'", "\u2019":"'", "\u201C":'"', "\u201D":'"',
    "\u2013":"-", "\u2014":"-", "\u00B7":"-", "\u2022":"-", "•":"-", "–":"-", "—":"-",
}

def canonicalize(text: str) -> str:
    if not text: return ""
    t = text
    # unify unicode punctuation
    for k,v in UNICODE_MAP.items():
        t = t.replace(k, v)
    # de-hyphenate soft line breaks: micro-\nservices -> microservices
    t = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', t)
    # normalize newlines, collapse runs of whitespace to single spaces
    t = t.replace("\t", " ")
    t = re.sub(r"[ \u00A0]+", " ", t)          # collapse spaces
    t = re.sub(r"\r\n|\r", "\n", t)            # normalize newlines
    t = re.sub(r"[ \t]+\n", "\n", t)           # trim trailing spaces
    # normalize bullets to "- "
    t = re.sub(r"(?m)^\s*-\s*", "- ", t)
    # strip page headers/footers like "Page X of Y" and standalone numbers
    t = re.sub(r"(?mi)^\s*page\s+\d+(\s+of\s+\d+)?\s*$", "", t)
    t = re.sub(r"(?m)^\s*\d+\s*$", "", t)
    # unify month names (for downstream parsing robustness)
    t = re.sub(r"(?i)\bsept\b", "Sep", t)
    return t.strip()

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def month_floor(d: date) -> date:
    return date(d.year, d.month, 1)

def parse_as_of(as_of_str: Optional[str]) -> date:
    """Use the first of the provided month; default to current month UTC floored."""
    if as_of_str:
        # accept YYYY-MM or YYYY-MM-DD
        try:
            if len(as_of_str) == 7:
                y,m = as_of_str.split("-")
                return date(int(y), int(m), 1)
            return month_floor(datetime.fromisoformat(as_of_str).date())
        except Exception:
            pass
    today = datetime.utcnow().date()
    return month_floor(today)

# In-memory cache: {(content_hash, as_of_month_iso, ruleset_key): result}
CACHE: Dict[Tuple[str,str,str], Dict[str,Any]] = {}

# =========================
# ===== CADENCE ENGINE =====
# =========================

SENT_SPLIT = re.compile(r'(?<=[\.\!\?])\s+(?=[A-Z0-9])')
BULLET_SPLIT = re.compile(r'(?m)^\s*-\s*|\n')   # bullets normalized to "- " by canonicalize
WORD_RE = re.compile(r"[A-Za-z']+")
ADJ_TAGS = {"JJ","JJR","JJS"}
VERB_TAGS = {"VB","VBD","VBG","VBN","VBP","VBZ"}
COMMON_OPENERS = {"developed","implemented","led","designed","built","managed","optimized"}

CADENCE_WEIGHTS = {
  "llm_human_cadence": 0.35,
  "sentence_variance": 0.15,
  "verb_diversity":    0.15,
  "adj_density":       0.10,
  "keyword_repetition":0.15,
  "burstiness":        0.10
}
CADENCE_THRESH = { "bucket": { "likely_ai": 0, "possibly_ai": 40, "likely_human": 65 } }

def split_sections(text: str) -> Dict[str, str]:
    lowers = text.lower()
    def grab(header, nxt):
        i = lowers.find(header)
        if i == -1: return ""
        j = min([lowers.find(n, i+1) for n in nxt if lowers.find(n, i+1) != -1] or [len(lowers)])
        return text[i:j].strip()
    return {
        "summary":   grab("summary", {"experience","work experience","employment","skills","education"}),
        "experience":grab("experience", {"education","skills","projects","certifications"}),
        "education": grab("education", {"skills","projects","certifications"}),
        "skills":    grab("skills", {"experience","work experience","education","projects"}) or ""
    }

def sentences(text: str) -> List[str]:
    raw = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    bullets = [b.strip() for b in BULLET_SPLIT.split(text) if b.strip()]
    return list({*raw, *bullets})

def tokenize(text: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(text)]

def pos_tags(words: List[str]) -> List[tuple]:
    # lightweight POS guesser; deterministic
    tagged = []
    for w in words:
        if w.endswith("ly"): tagged.append((w,"RB"))
        elif w.endswith("ing") or w in COMMON_OPENERS: tagged.append((w,"VB"))
        elif w in {"scalable","robust","seamless","impactful","reliable","secure"}: tagged.append((w,"JJ"))
        else: tagged.append((w,"NN"))
    return tagged

def sentence_length_variance(text: str) -> float:
    sents = sentences(text)
    lengths = [len(tokenize(s)) for s in sents if len(tokenize(s)) > 0]
    if len(lengths) < 3: return 0.20
    mean = sum(lengths)/len(lengths)
    var  = sum((l-mean)**2 for l in lengths)/len(lengths)
    return round(math.sqrt(var)/mean, 4)

def verb_diversity(text: str) -> float:
    sents = sentences(text)
    verbs = []
    for s in sents:
        ws = tokenize(s)
        tags = pos_tags(ws)
        verbs.extend([w for w,t in tags if t in VERB_TAGS])
    if not sents: return 0.0
    return round(len(set(verbs)) / max(1, len(sents)), 4)

def adjective_density(text: str) -> float:
    ws = tokenize(text)
    tags = pos_tags(ws)
    adjs = sum(1 for _,t in tags if t in ADJ_TAGS)
    return round(adjs / max(1, len(ws)), 4)

def opener_repetition(text: str) -> float:
    sents = sentences(text)
    starts = []
    for s in sents:
        ws = tokenize(s)
        if not ws: continue
        starts.append(ws[0])
    if not starts: return 0.0
    c = Counter(starts)
    top = sum(c[w] for w in c if w in COMMON_OPENERS)
    return round(top / len(starts), 4)

def keyword_repetition(text: str, n=2) -> float:
    ws = tokenize(text)
    ngrams = [" ".join(ws[i:i+n]) for i in range(len(ws)-n+1)]
    c = Counter(ngrams)
    if not ngrams: return 0.0
    repeats = sum(v for k,v in c.items() if v >= 3)
    return round(repeats / len(ngrams), 4)

def burstiness_proxy(text: str) -> float:
    ws = tokenize(text)
    c = Counter(ws)
    total = len(ws) or 1
    p = [v/total for v in c.values()]
    ent = -sum(pi*math.log(pi+1e-12) for pi in p)
    max_ent = math.log(len(c)+1e-12)
    if max_ent <= 0: return 0.5
    normalized = ent / max_ent
    return round(1 - normalized, 4)  # higher = more predictable (AI-ish)

def llm_human_cadence_score(section_text: str) -> float:
    # Deterministic stub (replace with temperature=0 model call + caching if desired)
    v = sentence_length_variance(section_text)
    rep = opener_repetition(section_text)
    ad = adjective_density(section_text)
    raw = (1.0 if v >= 0.10 else 0.0) * 0.5 + (1 - min(rep,1.0)) * 0.3 + (1 - min(abs(ad-0.015)/0.015, 1.0)) * 0.2
    return max(0.0, min(1.0, round(raw, 4)))

def section_cadence_score(text: str) -> Dict[str, Any]:
    sv = sentence_length_variance(text)
    vd = verb_diversity(text)
    ad = adjective_density(text)
    kr = keyword_repetition(text, n=2)
    bp = burstiness_proxy(text)
    llm = llm_human_cadence_score(text)
    norm = {
        "sentence_variance": min(1.0, round(sv/0.20, 4)),
        "verb_diversity":   min(1.0, round(vd/0.50, 4)),
        "adj_density":      max(0.0, round(1 - max(0.0, (ad-0.02)/0.03), 4)),
        "keyword_repetition":max(0.0, round(1 - min(1.0, kr/0.02), 4)),
        "burstiness":        max(0.0, round(1 - min(1.0, bp/0.50), 4)),
        "llm_human_cadence": llm
    }
    score = int(round(sum(norm[k]*CADENCE_WEIGHTS[k] for k in CADENCE_WEIGHTS) * 100))
    evid = []
    if sv < 0.10: evid.append(("low_sentence_variance", sv, f"{round(sv*100,1)}% sentence length variance"))
    rep = opener_repetition(text)
    if rep > 0.50: evid.append(("repetitive_openers", rep, f"{round(rep*100,1)}% bullets start with common openers"))
    if ad > 0.02: evid.append(("adj_density", ad, f"Adjectives ≈{round(ad*100,2)}% of tokens"))
    if kr > 0.02: evid.append(("keyword_repetition", kr, f"Repeated bigrams ≈{round(kr*100,2)}%"))
    if bp > 0.50: evid.append(("low_burstiness", bp, "Low textual entropy (predictable phrasing)"))
    return {"score": score, "metrics": norm, "evidence": evid}

def overall_cadence(resume_text: str) -> Dict[str, Any]:
    sections = split_sections(resume_text)
    per = {}
    evid_all = []
    weights = {"experience":0.6,"summary":0.2,"skills":0.15,"education":0.05}
    agg = 0.0; wsum = 0.0
    for name,content in sections.items():
        if not content or len(content) < 60:
            per[name] = None
            continue
        out = section_cadence_score(content)
        per[name] = out["score"]
        evid_all.extend([{"id":i,"value":v,"evidence":e,"section":name} for (i,v,e) in out["evidence"]])
    for s,w in weights.items():
        if per.get(s) is not None:
            agg += per[s]*w; wsum += w
    final = int(round(agg / (wsum or 1)))
    if final < CADENCE_THRESH["bucket"]["possibly_ai"]:
        bucket, conf = "likely_ai", ("high" if len(evid_all) >= 2 else "medium")
    elif final < CADENCE_THRESH["bucket"]["likely_human"]:
        bucket, conf = "possibly_ai","medium"
    else:
        bucket, conf = "likely_human", ("medium" if len(evid_all) <= 1 else "high")
    return {
        "score": final, "bucket": bucket, "confidence": conf,
        "section_scores": per, "signals": evid_all,
        "version": VERSIONS["cadence"]
    }

# ==========================
# ===== TIMELINE ENGINE =====
# ==========================

MONTHS = {
    'jan':1,'january':1,'feb':2,'february':2,'mar':3,'march':3,'apr':4,'april':4,'may':5,
    'jun':6,'june':6,'jul':7,'july':7,'aug':8,'august':8,'sep':9,'sept':9,'september':9,
    'oct':10,'october':10,'nov':11,'november':11,'dec':12,'december':12
}
DATE_PAT = re.compile(r'(?i)\b([A-Za-z]{3,9})\s+(\d{4})\b|(\d{1,2})[/-](\d{4})\b|(\d{4})\b')

@dataclass
class RoleTL:
    idx: int
    title: str
    company: str
    start_raw: str
    end_raw: str
    employment_type: str = "unknown"
    is_current: bool = False
    description: str = ""

@dataclass
class NormRole:
    idx: int
    title: str
    company: str
    start: Optional[date]
    end: Optional[date]
    precision_start: str
    precision_end: str
    employment_type: str
    meta: Dict[str, Any]

def last_day_of_month(y: int, m: int) -> int:
    return calendar.monthrange(y, m)[1]

def parse_point(s: str, is_end=False):
    s = (s or "").strip()
    if not s: return None, "unknown"
    if re.search(r'(?i)present|current|now', s):
        return None, "open"
    m = re.search(r'(?i)\b([A-Za-z]{3,9})\s+(\d{4})\b', s)
    if m:
        month = MONTHS[m.group(1).lower()]
        year = int(m.group(2))
        if is_end: return date(year, month, last_day_of_month(year, month)), "month"
        return date(year, month, 1), "month"
    m = re.search(r'\b(\d{1,2})[/-](\d{4})\b', s)
    if m:
        month = int(m.group(1)); year = int(m.group(2))
        if is_end: return date(year, month, last_day_of_month(year, month)), "month"
        return date(year, month, 1), "month"
    m = re.search(r'\b(19\d{2}|20\d{2})\b', s)
    if m:
        year = int(m.group(1))
        if is_end: return date(year, 12, 31), "year"
        return date(year, 1, 1), "year"
    return None, "unknown"

def normalize_roles(roles: List[RoleTL]) -> List[NormRole]:
    out = []
    for r in roles:
        s, ps = parse_point(r.start_raw, is_end=False)
        e, pe = parse_point(r.end_raw,   is_end=True)
        out.append(NormRole(
            idx=r.idx, title=r.title, company=r.company,
            start=s, end=e, precision_start=ps, precision_end=pe,
            employment_type=(r.employment_type or "unknown").lower(),
            meta={"is_current": r.is_current}
        ))
    out.sort(key=lambda x: (x.start or date.min, x.end or date.max))
    return out

def days_overlap(a: NormRole, b: NormRole, as_of: date) -> int:
    # open ranges use as_of instead of "today"
    a_end = a.end or as_of
    b_end = b.end or as_of
    latest_start = max(a.start or date.min, b.start or date.min)
    earliest_end = min(a_end, b_end)
    delta = (earliest_end - latest_start).days + 1
    return max(0, delta)

def role_is_flexible(nr: NormRole) -> bool:
    return nr.employment_type in {"contract","part-time","freelance","internship"}

def timeline_analysis(norm: List[NormRole], as_of: date) -> Dict[str,Any]:
    signals, overlaps, gaps = [], [], []
    present_fulltime = [r for r in norm if r.meta.get("is_current") and r.employment_type == "full-time"]
    if len(present_fulltime) >= 2:
        signals.append({"id":"multi_present_full_time","severity":"major",
                        "evidence":f"{len(present_fulltime)} roles marked Present and full-time"})
    for i in range(len(norm)-1):
        a, b = norm[i], norm[i+1]
        a_end = a.end or as_of
        b_start = b.start or as_of
        if a_end and b_start:
            g = (b_start - a_end).days - 1
            if g >= 180:
                gaps.append({"from_idx":a.idx,"to_idx":b.idx,"days":g})
        ov = days_overlap(a,b,as_of)
        if ov > 0:
            grace = 30
            both_day_precise = (a.precision_end=="day" and b.precision_start=="day")
            if both_day_precise: grace = 3
            flex = role_is_flexible(a) or role_is_flexible(b)
            allow_days = grace
            if flex and a.start and b.end:
                shorter = min( (a_end - a.start).days+1, (b.end or as_of - b.start).days+1 )
                allow_days = max(allow_days, int(0.5 * shorter))
            if ov > allow_days:
                sev = "minor" if ov<=60 else "moderate" if ov<=120 else "major"
                overlaps.append({"a_idx":a.idx,"b_idx":b.idx,"days":ov,"severity":sev,
                                 "evidence":f"Overlap {ov} days (allow {allow_days})"})
            else:
                signals.append({"id":"same_month_handoff","severity":"info",
                                "evidence":f"Handoff overlap {ov} days (within grace)","role_index":a.idx})
        if a.precision_end != b.precision_start:
            if a.precision_end=="year" or b.precision_start=="year":
                signals.append({"id":"granularity_shift","severity":"minor",
                                "evidence":f"Inconsistent precision: end({a.precision_end}) → start({b.precision_start})",
                                "role_index":a.idx})
    by_end_month = {}
    for r in norm:
        if r.end:
            key = (r.end.year, r.end.month)
            by_end_month[key] = by_end_month.get(key, 0) + 1
    for (y,m),cnt in by_end_month.items():
        if cnt >= 3:
            signals.append({"id":"end_date_pileup","severity":"moderate",
                            "evidence":f"{cnt} roles end in {y}-{m:02d}"})
    return {"signals":signals,"overlaps":overlaps,"gaps":gaps}

def score_timeline(norm: List[NormRole], analysis: Dict[str,Any]) -> Dict[str,Any]:
    score = 100
    max_overlap_ded, max_compress_ded, max_total_ded = 16, 12, 35
    overlap_ded = 0
    for ov in analysis["overlaps"]:
        overlap_ded += 4 if ov["severity"]=="minor" else 8 if ov["severity"]=="moderate" else 12
    for s in analysis["signals"]:
        if s["id"]=="multi_present_full_time": overlap_ded += 12
    overlap_ded = min(overlap_ded, max_overlap_ded)
    compress_ded = 0
    for s in analysis["signals"]:
        if s["id"]=="end_date_pileup": compress_ded += 8
        if s["id"]=="granularity_shift": compress_ded += 5
    compress_ded = min(compress_ded, max_compress_ded)
    total_ded = min(overlap_ded + compress_ded, max_total_ded)
    score = max(0, score - total_ded)
    bucket = "clean" if score>=85 else "caution" if score>=65 else "problematic"
    conf = "high" if (score>=85 or score<65 or len(analysis["overlaps"])>=1 or any(s["id"]=="multi_present_full_time" for s in analysis["signals"])) else "medium"
    return {"score":score,"bucket":bucket,"confidence":conf}

def analyze_timeline(roles_input: List[Dict[str,Any]], as_of: date) -> Dict[str,Any]:
    roles = [RoleTL(idx=i,
                    title=r.get("title",""), company=r.get("company",""),
                    start_raw=r.get("start",""), end_raw=r.get("end",""),
                    employment_type=r.get("employment_type","unknown"),
                    is_current=bool(r.get("is_current", False)),
                    description=r.get("description","")) for i,r in enumerate(roles_input)]
    norm = normalize_roles(roles)
    analysis = timeline_analysis(norm, as_of)
    scoring = score_timeline(norm, analysis)
    out_norm = [{
        "idx": r.idx, "title": r.title, "company": r.company,
        "start_iso": r.start.isoformat() if r.start else None,
        "end_iso": (r.end or None).isoformat() if r.end else None,
        "precision": {"start": r.precision_start, "end": r.precision_end},
        "employment_type": r.employment_type,
        "confidence": 0.9 if r.precision_start!="unknown" else 0.6
    } for r in norm]
    return {
        "score": scoring["score"], "bucket": scoring["bucket"], "confidence": scoring["confidence"],
        "signals": analysis["signals"],
        "timeline": { "normalized_roles": out_norm, "overlaps": analysis["overlaps"], "gaps": analysis["gaps"] },
        "version": VERSIONS["timeline"]
    }

# ==================================
# ===== SKILL-TIMELINE ENGINE ======
# ==================================

TECH_BASELINE = {
  "kubernetes": {"release":"2015-07-21","enterprise":"2017-01-01","synonyms":["k8s","kubernetes"]},
  "terraform": {"release":"2014-07-28","enterprise":"2016-01-01","synonyms":["terraform"]},
  "helm": {"release":"2016-01-19","enterprise":"2017-10-01","synonyms":["helm","helm charts"]},
  "argo cd": {"release":"2018-11-26","enterprise":"2020-01-01","synonyms":["argo cd","argocd","argo-cd"]},
  "crossplane": {"release":"2018-12-05","enterprise":"2023-07-01","synonyms":["crossplane"]},
  "github actions":{"release":"2019-11-13","enterprise":"2020-06-01","synonyms":["github actions","gha"]},
  "backstage":{"release":"2020-03-16","enterprise":"2022-04-01","synonyms":["backstage"]}
}
ADVANCED_SKILLS = {"kubernetes","crossplane","service mesh","argo cd","istio","anthos","aks","eks","gke","kafka","flink"}

def to_date(s: Optional[str]) -> Optional[date]:
    return date.fromisoformat(s) if s else None

def parse_iso_or_none(s: Optional[str]) -> Optional[date]:
    try: return to_date(s) if s else None
    except Exception: return None

def canonical_skill(token: str) -> Optional[str]:
    t = token.lower().strip()
    for k,v in TECH_BASELINE.items():
        if t == k or t in v["synonyms"]:
            return k
    alias = {"k8s":"kubernetes","aks":"kubernetes","eks":"kubernetes","gha":"github actions"}
    return alias.get(t)

def extract_skills(text: str) -> List[str]:
    toks = [w.strip().lower() for w in re.split(r'[^A-Za-z0-9\+\#\.\- ]', text)]
    out = []
    for t in toks:
        if not t or len(t) > 30: continue
        c = canonical_skill(t)
        if c: out.append(c)
    return list(dict.fromkeys(out))

def analyze_skill_timeline(roles_in: List[Dict[str,Any]], as_of: date, header_claims: Dict[str,Any]|None=None) -> Dict[str,Any]:
    roles = []
    for r in roles_in:
        roles.append({
            "idx": r.get("idx",0),
            "title": r.get("title",""),
            "start_iso": r.get("start_iso") or r.get("start"),
            "end_iso": r.get("end_iso") or r.get("end"),
            "description": r.get("description","") or ""
        })
    roles.sort(key=lambda x: parse_iso_or_none(x["start_iso"]) or date.min)

    skill_map: Dict[str, Dict[str,Any]] = {}
    signals: List[Dict[str,Any]] = []

    for role in roles:
        sdate = parse_iso_or_none(role["start_iso"])
        edate = parse_iso_or_none(role["end_iso"]) or as_of
        if not sdate: 
            continue
        skills = extract_skills(role["description"])
        for sk in skills:
            if sk not in skill_map:
                skill_map[sk] = {"skill":sk,"first_claim":sdate.isoformat(),"roles":[role["idx"]],"status":"ok"}
            else:
                skill_map[sk]["roles"].append(role["idx"])
            base = TECH_BASELINE.get(sk)
            if base:
                rel = date.fromisoformat(base["release"])
                ent = date.fromisoformat(base["enterprise"])
                if sdate < (rel - timedelta(days=30)):
                    signals.append({"id":"impossible_claim","severity":"major","skill":sk,"role_idx":role["idx"],
                                    "evidence":f"First claimed {sdate.isoformat()} but release {rel.isoformat()}"})
                    skill_map[sk]["status"] = "impossible_before_release"
                if ent and sdate < (ent - timedelta(days=90)):
                    if re.search(r'\b(production|enterprise|at scale|multi[- ]region|mission[- ]critical)\b',
                                 role["description"].lower()):
                        signals.append({"id":"implausible_enterprise_claim","severity":"moderate","skill":sk,"role_idx":role["idx"],
                                        "evidence":f"Enterprise claim in {sdate.isoformat()} but baseline {ent.isoformat()}"})
                        if skill_map[sk]["status"] == "ok":
                            skill_map[sk]["status"] = "early_enterprise_claim"

    role_start = {r["idx"]: parse_iso_or_none(r["start_iso"]) for r in roles}
    role_end   = {r["idx"]: parse_iso_or_none(r["end_iso"]) or as_of for r in roles}
    added_by_role: Dict[int, List[str]] = {r["idx"]: [] for r in roles}
    for sk,info in skill_map.items():
        fc = date.fromisoformat(info["first_claim"])
        owning = None
        for r in roles:
            s,e = role_start[r["idx"]], role_end[r["idx"]]
            if s and e and s <= fc <= e:
                owning = r["idx"]; break
        if owning is not None:
            added_by_role[owning].append(sk)

    for r in roles:
        s,e = role_start[r["idx"]], role_end[r["idx"]]
        if not (s and e): continue
        span_days = max(1, (e - s).days)
        new_sk = added_by_role.get(r["idx"], [])
        if len(new_sk) >= 9 and span_days >= 30:
            signals.append({"id":"stack_burst","severity":"minor","role_idx":r["idx"],
                            "evidence":f"Added {len(new_sk)} new tools within first 30 days window."})
        months = max(1, span_days // 30)
        if len(new_sk) / months > 3:
            signals.append({"id":"sustained_overload","severity":"moderate","role_idx":r["idx"],
                            "evidence":f"{len(new_sk)} new tools over ~{months} months (>3/mo)."})

    score = 100
    caps = {"impossible":24,"enterprise":16,"bursts":16}
    ded = {"impossible":0,"enterprise":0,"bursts":0}
    for s in signals:
        if s["id"]=="impossible_claim": ded["impossible"] += 12
        elif s["id"]=="implausible_enterprise_claim": ded["enterprise"] += 8
        elif s["id"] in {"stack_burst","sustained_overload"}: ded["bursts"] += 4 if s["id"]=="stack_burst" else 6
    total_ded = min(ded["impossible"],caps["impossible"]) + min(ded["enterprise"],caps["enterprise"]) + min(ded["bursts"],caps["bursts"])
    score = max(0, score - total_ded)
    bucket = "coherent" if score>=85 else "caution" if score>=65 else "incoherent"
    conf = "high" if (score>=85 or score<65 or any(s["severity"]=="major" for s in signals) or len(signals)==0) else "medium"
    skill_map_list = [dict(skill=v["skill"], first_claim=v["first_claim"], roles=v["roles"], status=v.get("status","ok"))
                      for v in skill_map.values()]
    return {
        "score": score, "bucket": bucket, "confidence": conf,
        "signals": signals, "skill_map": skill_map_list,
        "version": VERSIONS["skill_timeline"]
    }

# =====================================
# ===== TITLE PROGRESSION ENGINE ======
# =====================================

LEVELS = {0:"intern",1:"junior",2:"mid",3:"senior",4:"lead_staff_mgr",5:"principal_sr_mgr",6:"architect_director"}
MIN_MONTHS = {(0,1):6, (1,2):18, (2,3):24, (3,4):24, (4,5):36, (5,6):36}

def months_between(a: date, b: date) -> int:
    return (b.year - a.year)*12 + (b.month - a.month)

def detect_track(title: str, desc: str) -> str:
    tl = title.lower(); dl = (desc or "").lower()
    if any(w in tl for w in ["director","vp","vice president","head of","sr manager","senior manager"]):
        return "Manager"
    if "manager" in tl and re.search(r'\b(direct reports|hired|managed team|managed \d+|performance reviews|budget|org)\b', dl):
        return "Manager"
    if "lead" in tl and re.search(r'\b(managed|led team|direct reports|people|hiring)\b', dl):
        return "Manager"
    return "IC"

def normalize_level(title: str, track: str) -> int:
    t = title.lower()
    if any(k in t for k in ["intern","apprentice","trainee"]): return 0
    if any(k in t for k in ["junior","jr","associate","entry"]): return 1
    if any(k in t for k in ["principal","principal engineer"]): return 5
    if any(k in t for k in ["architect","distinguished"]): return 6
    if any(k in t for k in ["manager","sr manager","senior manager"]): return 4 if "sr" not in t else 5
    if any(k in t for k in ["staff","lead"]): return 4
    if any(k in t for k in ["senior","sr"]): return 3
    return 2

def analyze_title_progression(roles_in: List[Dict[str,Any]], as_of: date, domain_hint: str="general") -> Dict[str,Any]:
    roles = [{
        "idx": r.get("idx",0), "title": r.get("title",""),
        "start_iso": r.get("start_iso") or r.get("start"),
        "end_iso": r.get("end_iso") or r.get("end"),
        "employment_type": (r.get("employment_type") or "unknown").lower(),
        "description": r.get("description","") or ""
    } for r in roles_in]
    # Fill null ends with as_of for span calculations only
    roles.sort(key=lambda r: to_date(r["start_iso"]) or date.min)

    levels = []
    for r in roles:
        tr = detect_track(r["title"], r["description"])
        lv = normalize_level(r["title"], tr)
        levels.append({"idx": r["idx"], "title": r["title"], "track": tr, "level": lv})

    signals = []
    jumps = []
    for i in range(len(roles)-1):
        a,b = roles[i], roles[i+1]
        a_start = to_date(a["start_iso"]) or to_date(a["end_iso"]) or as_of
        a_end   = to_date(a["end_iso"])   or to_date(b["start_iso"]) or as_of
        b_start = to_date(b["start_iso"]) or to_date(b["end_iso"])   or as_of
        months = max(0, months_between(a_start, b_start))
        la = levels[i]["level"]; lb = levels[i+1]["level"]
        track_a, track_b = levels[i]["track"], levels[i+1]["track"]
        jump = lb - la

        min_req = None
        if (la, lb) in MIN_MONTHS: min_req = MIN_MONTHS[(la, lb)]
        elif jump == 1: min_req = 24 if la >= 2 else 18
        elif jump >= 2: min_req = 36

        if min_req is not None and months < min_req:
            diff = min_req - months
            if diff <= 6: sev, sid = "minor", "fast_jump"
            elif diff <= 18: sev, sid = "moderate", "accelerated_jump"
            else: sev, sid = "major", "implausible_jump"
            signals.append({"id": sid, "severity": sev, "from_idx": a["idx"], "to_idx": b["idx"],
                            "evidence": f"{LEVELS[la].title()} → {LEVELS[lb].title()} in {months} months (benchmark {min_req}+)"} )
        if jump >= 3:
            signals.append({"id":"large_jump_no_context","severity":"major","from_idx":a["idx"],"to_idx":b["idx"],
                            "evidence":f"+{jump} level jump without explicit context."})
        if track_a == "Manager" and track_b == "IC":
            if not re.search(r'\b(reorg|layoff|chose ic|hands[- ]on|individual contributor)\b', (a["description"]+" "+b["description"]).lower()):
                if months <= 12:
                    signals.append({"id":"track_regression_without_context","severity":"minor",
                                    "from_idx":a["idx"],"to_idx":b["idx"],
                                    "evidence":"Manager → IC within 12 months without context."})
        if lb < la - 1:
            signals.append({"id":"title_regression","severity":"moderate","from_idx":a["idx"],"to_idx":b["idx"],
                            "evidence":f"Regression {LEVELS[la]} → {LEVELS[lb]}."})

        span_m = max(1, months_between(a_start, a_end))
        if la >= 4 and span_m < 3:
            signals.append({"id":"short_high_title","severity":"minor","role_idx":a["idx"],
                            "evidence":f"High title held {span_m} months."})
        jumps.append({"from_idx":a["idx"],"to_idx":b["idx"],"months":months,"size":f"+{jump}"})

    for i in range(len(roles)-2):
        a,b,c = levels[i:i+3]
        a_s = to_date(roles[i]["start_iso"]); c_e = to_date(roles[i+2]["end_iso"]) or as_of
        if (b["level"]>a["level"]) and (c["level"]>b["level"]) and a_s and c_e and months_between(a_s, c_e) <= 18:
            signals.append({"id":"rapid_promotion_pileup","severity":"moderate","evidence":"3 promotions in ≤18 months."})

    score = 100
    caps = {"jumps":24,"osc":16,"pile":12}
    ded = {"jumps":0,"osc":0,"pile":0}
    for s in signals:
        if s["id"] in {"implausible_jump","large_jump_no_context"}: ded["jumps"] += 12
        elif s["id"]=="accelerated_jump": ded["jumps"] += 8
        elif s["id"]=="fast_jump": ded["jumps"] += 4
        elif s["id"] in {"title_regression","track_regression_without_context"}: ded["osc"] += 8 if s["id"]=="title_regression" else 6
        elif s["id"]=="rapid_promotion_pileup": ded["pile"] += 8
        elif s["id"]=="short_high_title": ded["pile"] += 4
    total = min(ded["jumps"],caps["jumps"]) + min(ded["osc"],caps["osc"]) + min(ded["pile"],caps["pile"])
    score = max(0, score - total)
    bucket = "plausible" if score>=85 else "caution" if score>=65 else "implausible"
    conf = "high" if (len(signals)==0 or any(s["severity"]=="major" for s in signals) or total>=12) else "medium"
    return {
      "score": score, "bucket": bucket, "confidence": conf,
      "signals": signals, "ladder": {"levels": levels, "jumps": jumps},
      "version": VERSIONS["title_progression"]
    }

# =====================================
# ======== FASTAPI + COMBINER =========
# =====================================

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import os

# ===============================
# === BASIC API KEY SECURITY ====
# ===============================
# The key is read from the environment variable RG_API_KEY (set in Railway)
API_KEY = os.getenv("RG_API_KEY", "")

app = FastAPI(title="Resume Authenticity Core-4 (Deterministic)", version=VERSIONS["core"])

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import os

API_KEY = os.getenv("RG_API_KEY", "")

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Allow docs and OpenAPI schema to load without a key
        if request.url.path in ("/docs", "/openapi.json", "/redoc"):
            return await call_next(request)
        # Enforce key on all other routes
        if API_KEY and request.headers.get("x-api-key") != API_KEY:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return await call_next(request)

app.add_middleware(APIKeyMiddleware)


from fastapi import Header, HTTPException
import os

# Simple API key from environment (e.g., RG_API_KEY on Railway)
API_KEY = os.getenv("RG_API_KEY", "")


# ----- Request models -----
class CadenceRequest(BaseModel):
    resume_text: str
    as_of_date: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

class RoleInTL(BaseModel):
    title: str
    company: Optional[str] = ""
    start: str
    end: str
    employment_type: Optional[Literal["full-time","contract","part-time","internship","freelance","unknown"]] = "unknown"
    is_current: Optional[bool] = False
    description: Optional[str] = ""
    location: Optional[str] = None

class TimelineRequest(BaseModel):
    roles: List[RoleInTL]
    as_of_date: Optional[str] = None
    meta: Optional[Dict[str,Any]] = None

class RoleInCommon(BaseModel):
    idx: int
    title: str
    start_iso: Optional[str] = None
    end_iso: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    employment_type: Optional[str] = "unknown"
    company: Optional[str] = ""
    description: Optional[str] = ""

class SkillTimelineRequest(BaseModel):
    roles: List[RoleInCommon]
    as_of_date: Optional[str] = None
    header_claims: Optional[Dict[str, Any]] = None

class TitleProgressionRequest(BaseModel):
    roles: List[RoleInCommon]
    as_of_date: Optional[str] = None
    domain_hint: Optional[str] = "general"

class AuthenticityRequest(BaseModel):
    resume_text: str
    roles_for_timeline: List[RoleInTL]
    roles_normalized: Optional[List[RoleInCommon]] = None
    header_claims: Optional[Dict[str, Any]] = None
    domain_hint: Optional[str] = "general"
    as_of_date: Optional[str] = None
    force_rescore: Optional[bool] = False

def version_block() -> Dict[str,str]:
    out = dict(VERSIONS)
    out["ruleset_hash"] = RULESET_HASH
    return out

# ----- Endpoints -----

@app.post("/score/cadence")
def score_cadence(req: CadenceRequest):
    asof = parse_as_of(req.as_of_date)
    canon = canonicalize(req.resume_text)
    cadence = overall_cadence(canon)
    return {
        **cadence,
        "determinism": {
            "as_of_date": asof.isoformat(),
            "content_hash": sha256_hex(canon),
            "cached": False
        },
        "scoring_version": version_block()
    }

@app.post("/score/timeline")
def score_timeline_endpoint(req: TimelineRequest):
    asof = parse_as_of(req.as_of_date)
    payload = [r.model_dump() for r in req.roles]
    result = analyze_timeline(payload, asof)
    return {
        **result,
        "determinism": {"as_of_date": asof.isoformat(), "cached": False},
        "scoring_version": version_block()
    }

@app.post("/score/skill_timeline")
def score_skill_timeline(req: SkillTimelineRequest):
    asof = parse_as_of(req.as_of_date)
    payload = [r.model_dump() for r in req.roles]
    result = analyze_skill_timeline(payload, asof, req.header_claims or {})
    return {
        **result,
        "determinism": {"as_of_date": asof.isoformat(), "cached": False},
        "scoring_version": version_block()
    }

@app.post("/score/title_progression")
def score_title_progression(req: TitleProgressionRequest):
    asof = parse_as_of(req.as_of_date)
    payload = [r.model_dump() for r in req.roles]
    result = analyze_title_progression(payload, asof, req.domain_hint or "general")
    return {
        **result,
        "determinism": {"as_of_date": asof.isoformat(), "cached": False},
        "scoring_version": version_block()
    }

@app.post("/score/authenticity")
def score_authenticity(req: AuthenticityRequest, x_api_key: str | None = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    asof = parse_as_of(req.as_of_date)

    # 1) Canonicalize + hash for caching
    canon = canonicalize(req.resume_text)
    content_hash = sha256_hex(canon)
    cache_key = (content_hash, asof.isoformat(), VERSIONS["core"] + "|" + VERSIONS["weights"] + "|" + RULESET_HASH)

    if not req.force_rescore and cache_key in CACHE:
        cached = CACHE[cache_key]
        return {**cached, "determinism": {**cached["determinism"], "cached": True}}

    # 2) Cadence
    cadence = overall_cadence(canon)

    # 3) Timeline normalize + score
    tl_payload = [r.model_dump() for r in req.roles_for_timeline]
    timeline = analyze_timeline(tl_payload, asof)

    # 4) Skill-Timeline uses normalized role dates if provided, else derive from timeline
    if req.roles_normalized:
        norm = [r.model_dump() for r in req.roles_normalized]
    else:
        norm = []
        for r in timeline["timeline"]["normalized_roles"]:
            norm.append({
                "idx": r["idx"],
                "title": r["title"],
                "start_iso": r["start_iso"],
                "end_iso": r["end_iso"],
                "employment_type": r["employment_type"],
                "company": r["company"],
                "description": ""  # stitch below
            })
        # stitch descriptions by index (stable)
        for i, src in enumerate(req.roles_for_timeline):
            if i < len(norm):
                norm[i]["description"] = src.description or ""

    skill_tl = analyze_skill_timeline(norm, asof, req.header_claims or {})
    title_prog = analyze_title_progression(norm, asof, req.domain_hint or "general")

    # 5) Combine — baseline weights
    W = {"cadence":0.35,"timeline":0.25,"skill_tl":0.25,"title":0.15}
    c = cadence["score"]/100
    t = timeline["score"]/100
    s = skill_tl["score"]/100
    p = title_prog["score"]/100
    combined = int(round((c*W["cadence"] + t*W["timeline"] + s*W["skill_tl"] + p*W["title"]) * 100))

    if combined >= 85: bucket = "authentic"
    elif combined >= 65: bucket = "mixed_signals"
    else: bucket = "high_risk"

    red_flags = 0
    for sub in (timeline, skill_tl, title_prog):
        red_flags += sum(1 for sgl in sub.get("signals", []) if sgl.get("severity") in {"major","moderate"})
    conf = "high" if (combined>=85 or combined<65 or red_flags>=2) else "medium"

    result = {
        "authenticity_score": combined,
        "bucket": bucket,
        "confidence": conf,
        "weights": W,
        "components": {
            "cadence": cadence,
            "timeline": timeline,
            "skill_timeline": skill_tl,
            "title_progression": title_prog
        },
        "determinism": {
            "as_of_date": asof.isoformat(),
            "content_hash": content_hash,
            "cached": False
        },
        "scoring_version": version_block(),
        "version": VERSIONS["core"]
    }

    CACHE[cache_key] = result
    return result
@app.post("/score/rule")
def score_rule_based():
    """
    Test endpoint for rule-based scoring with simulated flags.
    Replace hardcoded flags with extractor logic later.
    """
    flags = {
        "repeated_24mo_roles": True,
        "future_dated_roles": False,
        "implausible_continuity": True,
        "tools_predate_availability": False,
        "tool_count_excessive": True,
        "tools_unrelated": True,
        "tools_not_used_in_context": False,
        "buzzword_count_high": True,
        "repetitive_structure": False,
        "perfect_sentence_tone": True,
        "linkedin_missing": True,
        "voip_phone_number": False,
        "resume_farm_email": False,
        "specific_metrics_present": True,
        "realistic_tech_usage": True,
        "generic_bullet_points": False,
        "no_tangible_impact": False,
        "vague_education": False,
        "phone_location_mismatch": False,
        "education_timeline_verified": True
    }

    result = evaluate_resume(flags, verbose=True)
    return result

