"""
Phase 1: Data Audit
Loads training/validation data, reports class distributions, note length stats,
note types, and annotation integrity checks.
"""

import json
import re
import pandas as pd
import numpy as np
from pathlib import Path


TRAINING_DIR = Path("training")
VALIDATION_DIR = Path("validation")

PRIMARY_MEDS = {
    "estazolam", "eszopiclone", "flurazepam", "lemborexant", "quazepam",
    "ramelteon", "suvorexant", "temazepam", "triazolam", "zaleplon", "zolpidem"
}
SECONDARY_MEDS = {
    "acamprosate", "alprazolam", "clonazepam", "clonidine", "diazepam",
    "diphenhydramine", "doxepin", "gabapentin", "hydroxyzine", "lorazepam",
    "melatonin", "mirtazapine", "olanzapine", "quetiapine", "trazodone"
}


def load_corpus(split: str) -> pd.DataFrame:
    path = TRAINING_DIR / "corpus.csv" if split == "train" else VALIDATION_DIR / "corpus.csv"
    df = pd.read_csv(path)
    df["note_id"] = df["note_id"].astype(str)
    return df


def load_labels(split: str, task: int) -> dict:
    fname = f"subtask_{task}.json"
    path = TRAINING_DIR / fname if split == "train" else VALIDATION_DIR / fname
    with open(path) as f:
        return json.load(f)


def detect_note_type(text: str) -> str:
    t = text.lower()
    if "discharge summary" in t or "discharge date" in t:
        return "discharge_summary"
    if "nursing" in t or "nurse notes" in t:
        return "nursing"
    return "other"


def is_overnight_nursing(text: str) -> bool:
    """Heuristic: look for overnight shift timestamps (19:00-07:00 range)."""
    patterns = [
        r"\b(19|20|21|22|23):?\d{2}\b",  # evening timestamps
        r"\bnights?\s+shift\b",
        r"\bovernight\b",
        r"\b(1[9-9]|2[0-3])00\s*-\s*0[0-7]00\b",
    ]
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


def check_rule_consistency(note_id: str, s2: dict) -> list[str]:
    """Validate logical consistency of subtask-2 labels for one note."""
    issues = []
    entry = s2.get(note_id, {})
    def1 = entry.get("Definition 1", {}).get("label", "no")
    def2 = entry.get("Definition 2", {}).get("label", "no")
    rule_b = entry.get("Rule B", {}).get("label", "no")
    rule_c = entry.get("Rule C", {}).get("label", "no")

    # Rule C requires at least Def1 or Def2
    if rule_c == "yes" and def1 == "no" and def2 == "no":
        issues.append(f"{note_id}: Rule C=yes but Def1=no and Def2=no")
    return issues


def audit_split(split: str):
    print(f"\n{'='*60}")
    print(f" {split.upper()} SET AUDIT")
    print(f"{'='*60}")

    corpus = load_corpus(split)
    s1 = load_labels(split, 1)
    s2 = load_labels(split, 2)

    # --- Coverage check ---
    corpus_ids = set(corpus["note_id"].astype(str))
    s1_ids = set(s1.keys())
    s2_ids = set(s2.keys())

    print(f"\nNote counts:")
    print(f"  corpus.csv rows  : {len(corpus)}")
    print(f"  subtask_1.json   : {len(s1_ids)}")
    print(f"  subtask_2.json   : {len(s2_ids)}")

    missing_from_corpus = s1_ids - corpus_ids
    missing_from_s1 = corpus_ids - s1_ids
    if missing_from_corpus:
        print(f"  WARN: {len(missing_from_corpus)} note(s) in subtask_1 but NOT in corpus: {missing_from_corpus}")
    if missing_from_s1:
        print(f"  WARN: {len(missing_from_s1)} note(s) in corpus but NOT in subtask_1: {missing_from_s1}")

    # --- Subtask 1 class distribution ---
    yes_count = sum(1 for v in s1.values() if v.get("Insomnia") == "yes")
    no_count = len(s1) - yes_count
    print(f"\nSubtask 1 label distribution:")
    print(f"  yes (insomnia)   : {yes_count} ({yes_count/len(s1)*100:.1f}%)")
    print(f"  no               : {no_count}  ({no_count/len(s1)*100:.1f}%)")

    # --- Subtask 2 per-component distribution ---
    components = ["Definition 1", "Definition 2", "Rule B", "Rule C"]
    print(f"\nSubtask 2 per-component label distribution:")
    for comp in components:
        yes = sum(1 for v in s2.values() if v.get(comp, {}).get("label") == "yes")
        total = len(s2)
        print(f"  {comp:<14}: yes={yes}/{total} ({yes/total*100:.1f}%)")
    # Rule A is inferred (not stored): Def1 AND Def2
    rule_a_yes = sum(
        1 for v in s2.values()
        if v.get("Definition 1", {}).get("label") == "yes"
        and v.get("Definition 2", {}).get("label") == "yes"
    )
    print(f"  {'Rule A (inferred)':<14}: yes={rule_a_yes}/{len(s2)} ({rule_a_yes/len(s2)*100:.1f}%)")

    # --- Logical consistency ---
    all_issues = []
    for nid in s2:
        all_issues.extend(check_rule_consistency(nid, s2))
    if all_issues:
        print(f"\n  WARN: {len(all_issues)} logical inconsistency issue(s) found:")
        for iss in all_issues[:10]:
            print(f"    {iss}")
    else:
        print(f"\n  OK: No logical inconsistencies found in subtask_2 labels.")

    # --- Note length distribution ---
    corpus["word_count"] = corpus["text"].str.split().str.len()
    corpus["char_count"] = corpus["text"].str.len()
    # Rough token count: ~1.3 words per token for clinical text
    corpus["approx_tokens"] = (corpus["word_count"] / 0.75).astype(int)

    print(f"\nNote length distribution (words):")
    stats = corpus["word_count"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95])
    for k, v in stats.items():
        print(f"  {k:<8}: {v:.0f}")

    over_4096 = (corpus["approx_tokens"] > 4096).sum()
    print(f"  Notes likely exceeding 4096 tokens: {over_4096}")

    # --- Note type distribution ---
    corpus["note_type"] = corpus["text"].apply(detect_note_type)
    print(f"\nNote type distribution:")
    for ntype, cnt in corpus["note_type"].value_counts().items():
        print(f"  {ntype:<20}: {cnt}")

    nursing_notes = corpus[corpus["note_type"] == "nursing"]
    overnight = nursing_notes["text"].apply(is_overnight_nursing).sum()
    print(f"  Overnight nursing notes (heuristic): {overnight}/{len(nursing_notes)}")

    # --- Evidence span statistics (subtask 2) ---
    all_spans = []
    span_counts = []
    span_components = ["Definition 1", "Definition 2", "Rule B", "Rule C"]
    for nid, entry in s2.items():
        note_spans = []
        for comp in span_components:
            spans = entry.get(comp, {}).get("span", [])
            note_spans.extend(spans)
        span_counts.append(len(note_spans))
        all_spans.extend(note_spans)

    span_lengths = []
    for sp in all_spans:
        parts = str(sp).split()
        if len(parts) == 2:
            span_lengths.append(int(parts[1]) - int(parts[0]))

    print(f"\nEvidence span statistics (subtask 2):")
    print(f"  Total spans across all notes   : {len(all_spans)}")
    if span_lengths:
        print(f"  Avg span length (chars)        : {np.mean(span_lengths):.1f}")
        print(f"  Median span length (chars)     : {np.median(span_lengths):.1f}")
        print(f"  Max span length (chars)        : {max(span_lengths)}")
    print(f"  Avg spans per note             : {np.mean(span_counts):.1f}")
    print(f"  Max spans per note             : {max(span_counts) if span_counts else 0}")

    return corpus


def main():
    print("SMM4H HeaRD 2026 Task 2 — Phase 1: Data Audit")
    train_corpus = audit_split("train")
    val_corpus = audit_split("val")

    # --- Overlap check ---
    train_ids = set(train_corpus["note_id"].astype(str))
    val_ids = set(val_corpus["note_id"].astype(str))
    overlap = train_ids & val_ids
    print(f"\n{'='*60}")
    if overlap:
        print(f"WARN: {len(overlap)} note IDs appear in both train and validation!")
    else:
        print("OK: No note ID overlap between train and validation.")

    print("\nPhase 1 audit complete.\n")


if __name__ == "__main__":
    main()
