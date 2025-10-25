{\rtf1\ansi\ansicpg1252\cocoartf2759
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww22280\viewh17800\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # rule_based_scorer.py\
\
def evaluate_resume(flags: dict, verbose: bool = False) -> dict:\
    base_score = 100\
    log = []\
\
    # ----------------------------\
    # Timeline Category (max -20)\
    timeline_deductions = 0\
    if flags.get("repeated_24mo_roles"):\
        timeline_deductions += 10\
        log.append(("timeline", "Repeated 24-month roles", -10))\
    if flags.get("future_dated_roles"):\
        timeline_deductions += 7\
        log.append(("timeline", "Future-dated roles", -7))\
    if flags.get("implausible_continuity"):\
        timeline_deductions += 4\
        log.append(("timeline", "No career gap across long time", -4))\
    if flags.get("tools_predate_availability"):\
        timeline_deductions += 6\
        log.append(("timeline", "Used tools before public release", -6))\
    timeline_deductions = min(timeline_deductions, 20)\
\
    # ----------------------------\
    # Tool Stack Category (max -20)\
    tool_deductions = 0\
    if flags.get("tool_count_excessive"):\
        tool_deductions += 7\
        log.append(("tools", "Tool list is overinflated (>20)", -7))\
    if flags.get("tools_unrelated"):\
        tool_deductions += 5\
        log.append(("tools", "Unrelated tools listed", -5))\
    if flags.get("tools_not_used_in_context"):\
        tool_deductions += 5\
        log.append(("tools", "Tools not tied to any job/project", -5))\
    tool_deductions = min(tool_deductions, 20)\
\
    # ----------------------------\
    # Tone/Language Category (max -10)\
    tone_deductions = 0\
    if flags.get("buzzword_count_high"):\
        tone_deductions += 5\
        log.append(("tone", "Buzzword-heavy phrasing", -5))\
    if flags.get("repetitive_structure"):\
        tone_deductions += 4\
        log.append(("tone", "Repetitive structure / cadence", -4))\
    if flags.get("perfect_sentence_tone"):\
        tone_deductions += 3\
        log.append(("tone", "Over-sanitized/LLM-like tone", -3))\
    tone_deductions = min(tone_deductions, 10)\
\
    # ----------------------------\
    # Digital Signals (max -15)\
    digital_deductions = 0\
    if flags.get("linkedin_missing"):\
        digital_deductions += 5\
        log.append(("digital", "Missing LinkedIn or web signal", -5))\
    if flags.get("voip_phone_number"):\
        digital_deductions += 3\
        log.append(("digital", "VOIP or masked phone number", -3))\
    if flags.get("resume_farm_email"):\
        digital_deductions += 5\
        log.append(("digital", "Email from known resume farm", -5))\
    digital_deductions = min(digital_deductions, 15)\
\
    # ----------------------------\
    # Detail Adjustments (\'b110)\
    detail_score = 0\
    if flags.get("specific_metrics_present"):\
        detail_score += 5\
        log.append(("detail", "Uses project-specific metrics", +5))\
    if flags.get("realistic_tech_usage"):\
        detail_score += 5\
        log.append(("detail", "Tech stack matches known use cases", +5))\
    if flags.get("generic_bullet_points"):\
        detail_score -= 5\
        log.append(("detail", "Generic, vague bullet points", -5))\
    if flags.get("no_tangible_impact"):\
        detail_score -= 5\
        log.append(("detail", "No measurable results listed", -5))\
    detail_score = max(min(detail_score, 10), -10)\
\
    # ----------------------------\
    # Education/Location (\'b15)\
    edu_loc_score = 0\
    if flags.get("vague_education"):\
        edu_loc_score -= 3\
        log.append(("edu/location", "Vague or missing education detail", -3))\
    if flags.get("phone_location_mismatch"):\
        edu_loc_score -= 2\
        log.append(("edu/location", "Phone area code mismatch", -2))\
    if flags.get("education_timeline_verified"):\
        edu_loc_score += 3\
        log.append(("edu/location", "Education timeline matches job history", +3))\
    edu_loc_score = max(min(edu_loc_score, 5), -5)\
\
    # ----------------------------\
    # Final score calculation\
    deductions_total = (\
        timeline_deductions +\
        tool_deductions +\
        tone_deductions +\
        digital_deductions\
    )\
    final_score = base_score - deductions_total + detail_score + edu_loc_score\
    final_score = max(min(final_score, 100), 0)\
\
    if verbose:\
        return \{\
            "final_score": final_score,\
            "deductions_total": deductions_total,\
            "breakdown": log\
        \}\
    return \{"final_score": final_score\}\
}