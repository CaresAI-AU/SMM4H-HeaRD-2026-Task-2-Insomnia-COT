"""
Phase 4 (Subtask 1): Evaluation & Error Analysis

Computes Precision, Recall, F1, and builds a confusion matrix.
Performs error categorization on false positives and false negatives.

Usage:
  python subtask1_evaluate.py --predictions predictions_subtask1_baseline.json
  python subtask1_evaluate.py --predictions pred_a.json pred_b.json pred_c.json
                               --names baseline dynamic self_consist
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd


VALIDATION_DIR = Path("validation")
TRAINING_DIR = Path("training")

PRIMARY_MEDS = frozenset({
    "estazolam", "eszopiclone", "flurazepam", "lemborexant", "quazepam",
    "ramelteon", "suvorexant", "temazepam", "triazolam", "zaleplon", "zolpidem"
})
SECONDARY_MEDS = frozenset({
    "acamprosate", "alprazolam", "clonazepam", "clonidine", "diazepam",
    "diphenhydramine", "doxepin", "gabapentin", "hydroxyzine", "lorazepam",
    "melatonin", "mirtazapine", "olanzapine", "quetiapine", "trazodone"
})

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gold_labels(split: str = "val") -> dict[str, str]:
    data_dir = VALIDATION_DIR if split == "val" else TRAINING_DIR
    with open(data_dir / "subtask_1.json") as f:
        raw = json.load(f)
    return {str(k): v["Insomnia"] for k, v in raw.items()}


def load_gold_subtask2(split: str = "val") -> dict:
    data_dir = VALIDATION_DIR if split == "val" else TRAINING_DIR
    with open(data_dir / "subtask_2.json") as f:
        return json.load(f)


def load_corpus(split: str = "val") -> pd.DataFrame:
    data_dir = VALIDATION_DIR if split == "val" else TRAINING_DIR
    df = pd.read_csv(data_dir / "corpus.csv")
    df["note_id"] = df["note_id"].astype(str)
    return df


def load_predictions(path: str) -> dict[str, str]:
    with open(path) as f:
        raw = json.load(f)
    result = {}
    for note_id, pred in raw.items():
        if isinstance(pred, dict):
            # Try both formats: {"Insomnia": "yes"/"no"} or legacy {"final_label": "yes"/"no"}
            label = pred.get("Insomnia") or pred.get("final_label", "no")
            result[str(note_id)] = label
        else:
            result[str(note_id)] = str(pred)
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(gold: dict[str, str], pred: dict[str, str]) -> dict:
    """Compute P, R, F1 for the positive class (yes=insomnia)."""
    tp = fp = fn = tn = 0
    for note_id, true_label in gold.items():
        pred_label = pred.get(note_id, "no")
        if true_label == "yes" and pred_label == "yes":
            tp += 1
        elif true_label == "no" and pred_label == "yes":
            fp += 1
        elif true_label == "yes" and pred_label == "no":
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    accuracy = (tp + tn) / len(gold) if gold else 0.0

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Precision": round(precision * 100, 1),
        "Recall": round(recall * 100, 1),
        "F1": round(f1 * 100, 1),
        "Accuracy": round(accuracy * 100, 1),
    }


def print_confusion_matrix(metrics: dict):
    tp, fp, fn, tn = metrics["TP"], metrics["FP"], metrics["FN"], metrics["TN"]
    print(f"\n  Confusion Matrix (positive = 'yes/insomnia'):")
    print(f"  {'':15}  Pred YES  Pred NO")
    print(f"  {'True YES':15}  {tp:>8}  {fn:>7}")
    print(f"  {'True NO':15}  {fp:>8}  {tn:>7}")


def print_metrics_table(results: list[tuple[str, dict]]):
    """Print a comparison table of metrics for multiple configurations."""
    header = f"{'Config':<25} {'P':>6} {'R':>6} {'F1':>6} {'Acc':>6}  TP  FP  FN  TN"
    print(f"\n{header}")
    print("-" * len(header))
    for name, m in results:
        print(f"{name:<25} {m['Precision']:>6.1f} {m['Recall']:>6.1f} {m['F1']:>6.1f} "
              f"{m['Accuracy']:>6.1f}  {m['TP']:>2}  {m['FP']:>2}  {m['FN']:>2}  {m['TN']:>2}")


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

ERROR_CATEGORIES = [
    "rule_b_missed",          # Primary medication present but model said no
    "rule_c_missed",          # Secondary med + symptoms but model said no
    "rule_a_missed",          # Def1 + Def2 present but model said no
    "comorbidity_confusion",  # Symptoms likely from comorbidity, model said yes
    "medication_hallucination",  # Model claimed a medication not in note
    "implicit_mention_missed",   # Implicit sleep language not caught
    "night_shift_error",         # Overnight nursing note misclassified
    "unknown",
]


def _contains_primary_med(text: str) -> bool:
    t = text.lower()
    return any(med in t for med in PRIMARY_MEDS)


def _contains_secondary_med(text: str) -> bool:
    t = text.lower()
    return any(med in t for med in SECONDARY_MEDS)


def _has_sleep_symptoms(text: str) -> bool:
    patterns = [
        "insomnia", "trouble sleeping", "can't sleep", "cannot sleep",
        "difficulty sleeping", "trouble initiating", "woke up", "waking up",
        "unable to sleep", "sleep disturbance", "restless", "awake all night"
    ]
    t = text.lower()
    return any(p in t for p in patterns)


def _is_overnight_nursing(text: str) -> bool:
    import re
    patterns = [r"\bnights?\s+shift\b", r"\bovernight\b", r"\b(19|20|21|22|23):\d{2}\b"]
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False


def categorize_error(
    note_id: str,
    true_label: str,
    pred_label: str,
    note_text: str,
    s2_gold: dict,
) -> str:
    """Categorize a prediction error into a meaningful type."""
    entry = s2_gold.get(note_id, {})

    if true_label == "yes" and pred_label == "no":
        # False negative — we missed insomnia
        rule_b = entry.get("Rule B", {}).get("label", "no")
        rule_a = entry.get("Rule A", {}).get("label") or (
            "yes" if entry.get("Definition 1", {}).get("label") == "yes"
            and entry.get("Definition 2", {}).get("label") == "yes" else "no"
        )
        rule_c = entry.get("Rule C", {}).get("label", "no")

        if rule_b == "yes":
            return "rule_b_missed"
        if rule_a == "yes":
            return "rule_a_missed"
        if rule_c == "yes":
            return "rule_c_missed"
        if not _has_sleep_symptoms(note_text):
            return "implicit_mention_missed"
        return "unknown"

    elif true_label == "no" and pred_label == "yes":
        # False positive — we over-detected insomnia
        if _is_overnight_nursing(note_text):
            return "night_shift_error"
        # Check if model may have hallucinated a medication
        if not _contains_primary_med(note_text) and not _contains_secondary_med(note_text):
            return "medication_hallucination"
        return "comorbidity_confusion"

    return "unknown"


def analyze_errors(
    gold: dict[str, str],
    pred: dict[str, str],
    corpus: pd.DataFrame,
    s2_gold: dict,
) -> list[dict]:
    corpus_map = dict(zip(corpus["note_id"], corpus["text"]))
    errors = []

    for note_id, true_label in gold.items():
        pred_label = pred.get(note_id, "no")
        if true_label == pred_label:
            continue

        note_text = corpus_map.get(note_id, "")
        error_type = categorize_error(note_id, true_label, pred_label, note_text, s2_gold)

        errors.append({
            "note_id": note_id,
            "true_label": true_label,
            "pred_label": pred_label,
            "error_type": error_type,
            "error_category": "FN" if true_label == "yes" else "FP",
        })

    return errors


def print_error_analysis(errors: list[dict]):
    if not errors:
        print("\n  No errors found.")
        return

    fps = [e for e in errors if e["error_category"] == "FP"]
    fns = [e for e in errors if e["error_category"] == "FN"]

    print(f"\n  Error analysis: {len(errors)} total errors")
    print(f"  False Positives (over-detected insomnia): {len(fps)}")
    print(f"  False Negatives (missed insomnia):        {len(fns)}")

    from collections import Counter
    type_counts = Counter(e["error_type"] for e in errors)
    print(f"\n  Error type breakdown:")
    for etype, cnt in type_counts.most_common():
        print(f"    {etype:<35}: {cnt}")

    if fps:
        print(f"\n  False Positive note IDs: {[e['note_id'] for e in fps]}")
    if fns:
        print(f"  False Negative note IDs: {[e['note_id'] for e in fns]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_one(pred_path: str, name: str, gold: dict, s2_gold: dict, corpus: pd.DataFrame) -> dict:
    print(f"\n{'='*60}")
    print(f" Evaluating: {name}")
    print(f" Predictions file: {pred_path}")
    print(f"{'='*60}")

    pred = load_predictions(pred_path)

    # Check coverage
    missing = set(gold.keys()) - set(pred.keys())
    extra = set(pred.keys()) - set(gold.keys())
    if missing:
        print(f"  WARN: {len(missing)} gold note(s) missing from predictions")
    if extra:
        print(f"  INFO: {len(extra)} extra predictions not in gold set")

    metrics = compute_metrics(gold, pred)
    print(f"\n  Precision : {metrics['Precision']:.1f}%")
    print(f"  Recall    : {metrics['Recall']:.1f}%")
    print(f"  F1        : {metrics['F1']:.1f}%")
    print(f"  Accuracy  : {metrics['Accuracy']:.1f}%")
    print_confusion_matrix(metrics)

    errors = analyze_errors(gold, pred, corpus, s2_gold)
    print_error_analysis(errors)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Subtask 1 Evaluation & Error Analysis")
    parser.add_argument("--predictions", nargs="+", required=True,
                        help="Path(s) to prediction JSON file(s)")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Human-readable names for each prediction file (for the comparison table)")
    parser.add_argument("--split", default="val", choices=["train", "val"],
                        help="Split to evaluate against")
    args = parser.parse_args()

    if args.names and len(args.names) != len(args.predictions):
        parser.error("--names must have the same number of entries as --predictions")

    names = args.names or [Path(p).stem for p in args.predictions]

    gold = load_gold_labels(args.split)
    s2_gold = load_gold_subtask2(args.split)
    corpus = load_corpus(args.split)

    print(f"\nGold labels loaded: {len(gold)} notes "
          f"({sum(1 for v in gold.values() if v=='yes')} yes, "
          f"{sum(1 for v in gold.values() if v=='no')} no)")

    all_results = []
    for pred_path, name in zip(args.predictions, names):
        metrics = evaluate_one(pred_path, name, gold, s2_gold, corpus)
        all_results.append((name, metrics))

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(" Comparison Table")
        print_metrics_table(all_results)

    # Save summary CSV
    rows = []
    for name, m in all_results:
        rows.append({"config": name, **m})
    df = pd.DataFrame(rows)
    out_path = "evaluation_subtask1.csv"
    df.to_csv(out_path, index=False)
    print(f"\nEvaluation summary saved to {out_path}")


if __name__ == "__main__":
    main()
