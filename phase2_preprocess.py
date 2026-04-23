"""
Phase 2: Preprocessing Pipeline
- Medication normalization (brand → generic, misspelling variants)
- Note-type tagging (discharge summary vs. nursing vs. other)
- Overnight shift detection for nursing notes
- Flag medications found in note text but not in structured medication list
- Saves preprocessed metadata as JSON
"""

import json
import re
import pandas as pd
from pathlib import Path
from typing import Optional


TRAINING_DIR = Path("training")
VALIDATION_DIR = Path("validation")

# ---------------------------------------------------------------------------
# Medication normalization tables
# ---------------------------------------------------------------------------

# Brand names and common aliases → generic names (lowercase)
BRAND_TO_GENERIC: dict[str, str] = {
    # Primary insomnia medications
    "ambien": "zolpidem",
    "ambien cr": "zolpidem",
    "edluar": "zolpidem",
    "intermezzo": "zolpidem",
    "zolpimist": "zolpidem",
    "lunesta": "eszopiclone",
    "sonata": "zaleplon",
    "restoril": "temazepam",
    "halcion": "triazolam",
    "prosom": "estazolam",
    "dalmane": "flurazepam",
    "rozerem": "ramelteon",
    "belsomra": "suvorexant",
    "dayvigo": "lemborexant",
    # Secondary insomnia medications
    "ativan": "lorazepam",
    "valium": "diazepam",
    "klonopin": "clonazepam",
    "xanax": "alprazolam",
    "catapres": "clonidine",
    "neurontin": "gabapentin",
    "benadryl": "diphenhydramine",
    "unisom": "diphenhydramine",
    "vistaril": "hydroxyzine",
    "atarax": "hydroxyzine",
    "seroquel": "quetiapine",
    "zyprexa": "olanzapine",
    "zyprexia": "olanzapine",   # misspelling variant
    "remeron": "mirtazapine",
    "desyrel": "trazodone",
    "sinequan": "doxepin",
    "silenor": "doxepin",
    "campral": "acamprosate",
}

# Misspelling variants → canonical generic name (lowercase)
MISSPELLING_TO_GENERIC: dict[str, str] = {
    "clozepam": "clonazepam",
    "clonzepam": "clonazepam",
    "clonazepan": "clonazepam",
    "lorazapam": "lorazepam",
    "lorazapem": "lorazepam",
    "diazapam": "diazepam",
    "alprazolam": "alprazolam",   # keep as is but include for completeness
    "zolpiden": "zolpidem",
    "zolipdem": "zolpidem",
    "trazadone": "trazodone",
    "trazadose": "trazodone",
    "quetiapine": "quetiapine",
    "mirtazipine": "mirtazapine",
    "gabapentine": "gabapentin",
    "hydroxizine": "hydroxyzine",
    "diphenhydramine": "diphenhydramine",
}

ALL_NORMALIZATION: dict[str, str] = {**BRAND_TO_GENERIC, **MISSPELLING_TO_GENERIC}

PRIMARY_MEDS = frozenset({
    "estazolam", "eszopiclone", "flurazepam", "lemborexant", "quazepam",
    "ramelteon", "suvorexant", "temazepam", "triazolam", "zaleplon", "zolpidem"
})
SECONDARY_MEDS = frozenset({
    "acamprosate", "alprazolam", "clonazepam", "clonidine", "diazepam",
    "diphenhydramine", "doxepin", "gabapentin", "hydroxyzine", "lorazepam",
    "melatonin", "mirtazapine", "olanzapine", "quetiapine", "trazodone"
})
ALL_INSOMNIA_MEDS = PRIMARY_MEDS | SECONDARY_MEDS


def normalize_medication(name: str) -> str:
    """Return the canonical lowercase generic name, or the original lowercased."""
    key = name.lower().strip()
    return ALL_NORMALIZATION.get(key, key)


def normalize_text(text: str) -> str:
    """
    Replace brand names and misspellings in free text with canonical generic names.
    Uses whole-word matching to avoid partial replacements.
    """
    for brand, generic in sorted(ALL_NORMALIZATION.items(), key=lambda x: -len(x[0])):
        pattern = r"\b" + re.escape(brand) + r"\b"
        text = re.sub(pattern, generic, text, flags=re.IGNORECASE)
    return text


def extract_structured_meds(header_line: str) -> list[str]:
    """
    Parse medications from the enriched note header line, e.g.:
    'male patient in seventies prescribed Med1, Med2, Med3'
    Returns list of lowercase normalized generic names.
    """
    match = re.search(r"prescribed\s+(.+)$", header_line, re.IGNORECASE)
    if not match:
        return []
    meds_str = match.group(1)
    meds = [m.strip() for m in meds_str.split(",")]
    return [normalize_medication(m) for m in meds if m and m.lower() != "no drugs"]


def find_insomnia_meds_in_text(text: str) -> dict[str, list[str]]:
    """
    Scan free text for insomnia medication mentions.
    Returns dict with 'primary' and 'secondary' lists of matched names.
    """
    found_primary = []
    found_secondary = []
    normalized_text = normalize_text(text)
    for med in PRIMARY_MEDS:
        if re.search(r"\b" + re.escape(med) + r"\b", normalized_text, re.IGNORECASE):
            found_primary.append(med)
    for med in SECONDARY_MEDS:
        if re.search(r"\b" + re.escape(med) + r"\b", normalized_text, re.IGNORECASE):
            found_secondary.append(med)
    return {"primary": found_primary, "secondary": found_secondary}


def detect_note_type(text: str) -> str:
    t = text.lower()
    if "discharge summary" in t or "discharge date" in t:
        return "discharge_summary"
    if "nursing" in t or "nurse notes" in t:
        return "nursing"
    return "other"


def is_overnight_nursing(text: str) -> bool:
    """Heuristic: detect overnight shift markers in nursing notes."""
    patterns = [
        r"\b(19|20|21|22|23)[:h]\d{2}\b",
        r"\bnights?\s+shift\b",
        r"\bovernight\b",
        r"\b(1[9-9]|2[0-3])00\s*-\s*0[0-7]00\b",
        r"\bnight\s+nursing\b",
    ]
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


def preprocess_note(row: pd.Series) -> dict:
    note_id = str(row["note_id"])
    text = str(row["text"])

    # Extract first line as header
    lines = text.split("\n")
    header_line = lines[0] if lines else ""

    # Structured medication list from header
    structured_meds = extract_structured_meds(header_line)
    structured_insomnia = [m for m in structured_meds if m in ALL_INSOMNIA_MEDS]
    structured_primary = [m for m in structured_meds if m in PRIMARY_MEDS]
    structured_secondary = [m for m in structured_meds if m in SECONDARY_MEDS]

    # Medications found in free text
    text_meds = find_insomnia_meds_in_text(text)

    # Medications in text but not in structured list
    text_only_primary = [m for m in text_meds["primary"] if m not in structured_insomnia]
    text_only_secondary = [m for m in text_meds["secondary"] if m not in structured_insomnia]

    # Note type
    note_type = detect_note_type(text)
    overnight = is_overnight_nursing(text) if note_type == "nursing" else False

    # Normalized note text
    normalized_text = normalize_text(text)

    return {
        "note_id": note_id,
        "note_type": note_type,
        "is_overnight_nursing": overnight,
        "structured_insomnia_meds": structured_insomnia,
        "structured_primary_meds": structured_primary,
        "structured_secondary_meds": structured_secondary,
        "text_primary_meds": text_meds["primary"],
        "text_secondary_meds": text_meds["secondary"],
        "text_only_primary_meds": text_only_primary,
        "text_only_secondary_meds": text_only_secondary,
        "normalized_text": normalized_text,
    }


def process_split(split: str) -> dict[str, dict]:
    data_dir = TRAINING_DIR if split == "train" else VALIDATION_DIR
    corpus = pd.read_csv(data_dir / "corpus.csv")
    corpus["note_id"] = corpus["note_id"].astype(str)

    results = {}
    for _, row in corpus.iterrows():
        processed = preprocess_note(row)
        note_id = processed.pop("note_id")
        results[note_id] = processed

    return results


def validate_preprocessing(split: str, processed: dict[str, dict]):
    """Spot-check: verify format and report flagged notes."""
    data_dir = TRAINING_DIR if split == "train" else VALIDATION_DIR

    notes_with_text_only_meds = [
        nid for nid, d in processed.items()
        if d["text_only_primary_meds"] or d["text_only_secondary_meds"]
    ]
    overnight_nursing = [
        nid for nid, d in processed.items() if d["is_overnight_nursing"]
    ]
    notes_with_insomnia_meds = [
        nid for nid, d in processed.items() if d["structured_insomnia_meds"]
    ]

    print(f"\n{split.upper()} preprocessing summary:")
    print(f"  Total notes processed          : {len(processed)}")
    print(f"  Notes with structured insomnia meds: {len(notes_with_insomnia_meds)}")
    print(f"  Notes with text-only insomnia meds : {len(notes_with_text_only_meds)}")
    print(f"  Overnight nursing notes        : {len(overnight_nursing)}")

    note_types = {}
    for d in processed.values():
        t = d["note_type"]
        note_types[t] = note_types.get(t, 0) + 1
    for t, c in sorted(note_types.items()):
        print(f"  Note type '{t}': {c}")


def main():
    print("SMM4H HeaRD 2026 Task 2 — Phase 2: Preprocessing Pipeline")

    for split in ("train", "val"):
        processed = process_split(split)
        validate_preprocessing(split, processed)

        out_dir = TRAINING_DIR if split == "train" else VALIDATION_DIR
        out_path = out_dir / "preprocessed.json"
        with open(out_path, "w") as f:
            json.dump(processed, f, indent=2)
        print(f"  Saved: {out_path}")

    print("\nPhase 2 preprocessing complete.\n")


if __name__ == "__main__":
    main()
