import re
from typing import Dict, List

from .schemas import Composition


RISK_PATTERNS = [
    r"\blike\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?",
    r"\bsound(?:s)?\s+like\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?",
    r"\bin\s+the\s+style\s+of\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?",
    r"\bcopy\s+.+\bsong\b",
    r"\bclone\s+.+\bvoice\b",
]


def commercial_review(composition: Composition) -> Dict[str, object]:
    text = " ".join(
        [
            composition.title,
            composition.style,
            composition.mood,
            " ".join(composition.lyrics),
            " ".join(composition.style_notes),
            " ".join(composition.originality_notes),
            " ".join(composition.commercial_notes),
            composition.disclaimer,
        ]
    )
    warnings: List[str] = []

    for pattern in RISK_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            warnings.append("Potential artist/song imitation wording detected. Convert it to broad genre language before commercial use.")
            break

    if "commercial use" not in composition.disclaimer.lower():
        warnings.append("Disclaimer does not mention commercial-use review.")

    if not composition.originality_notes:
        warnings.append("Originality notes are missing.")

    if not composition.agent_trace:
        warnings.append("Agent trace is missing; commercial auditability is weaker.")

    checklist = [
        "Review generated chords, melody, lyrics, and audio before release.",
        "Avoid claiming the output is guaranteed unique.",
        "Keep prompt/output records for auditability.",
        "Replace direct living-artist references with broad genre descriptions.",
        "Use human review before monetized distribution.",
    ]
    notes = composition.commercial_notes or [
        "Draft is intended as an editable starting point, not a guaranteed rights-cleared final master."
    ]
    score = 100 if not warnings else max(45, 100 - len(warnings) * 18)
    return {
        "score": score,
        "warnings": warnings,
        "checklist": checklist,
        "notes": notes,
    }
