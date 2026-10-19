import re

SUSPICIOUS_PATTERNS = {
    "upfront payment request": r"\b(upfront|advance|registration|application|processing) fee\b|\bpay (?:a|the) fee\b",
    "unusually high pay claim": r"\b(?:earn|salary|income|pay)\b.{0,35}\b(?:\$|\d{3,})",
    "informal contact channel": r"\b(?:whatsapp|telegram|gmail\.com|yahoo\.com)\b",
    "urgent hiring language": r"\b(urgent(?:ly)?|immediate(?:ly)?|start today|no interview)\b",
    "sensitive information request": r"\b(bank account|credit card|social security|passport number)\b",
}


def risk_indicators(posting: dict) -> list[dict]:
    text = " ".join(str(posting.get(k) or "") for k in (
        "title", "description", "requirements", "company_profile", "benefits", "location"))
    lower = text.lower()
    findings = [{"indicator": label, "detail": "Matching language appears in the supplied posting."}
                for label, pattern in SUSPICIOUS_PATTERNS.items() if re.search(pattern, lower)]
    if not str(posting.get("company_profile") or "").strip():
        findings.append({"indicator": "company profile not provided",
                         "detail": "The posting does not include a company profile."})
    if str(posting.get("has_company_logo", "")).strip().lower() in {"0", "false", "no"}:
        findings.append({"indicator": "company logo not provided",
                         "detail": "The source posting marks the company logo as absent."})
    if not findings:
        findings.append({"indicator": "no listed warning pattern matched",
                         "detail": "Absence of these patterns does not establish that a posting is genuine."})
    return findings

