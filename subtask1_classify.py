"""
Phase 3: Subtask 1 — Binary Classification (Insomnia yes/no)

Modes:
  baseline      — Fixed 8-shot (4 yes + 4 no) few-shot + CoT prompt, greedy decoding
  dynamic       — Dynamic few-shot retrieval via sentence embeddings (top-4 yes + top-4 no)
  self_consist  — Like 'baseline' but runs 5x at temp=0.7 and takes majority vote

Usage:
  python subtask1_classify.py --mode baseline 
  python subtask1_classify.py --mode dynamic --embed_model ncbi/MedCPT-Query-Encoder
  python subtask1_classify.py --mode self_consist --n_votes 5

Environment variables:
  LLM_API_KEY    — API key (default: $OPENAI_API_KEY)
  LLM_BASE_URL   — OpenAI-compatible endpoint base URL (default: https://api.openai.com/v1)
  LLM_MODEL      — Model name override
"""

import argparse
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Optional
from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAINING_DIR = Path("training")
VALIDATION_DIR = Path("validation")
TEST_DIR = Path("test")

SEED = 4351
# ---------------------------------------------------------------------------
# Insomnia rules (embedded in prompt)
# ---------------------------------------------------------------------------
INSOMNIA_RULES_BLOCK = """## Insomnia Definitions and Rules

**Definition 1 (Difficulty Sleeping)** — The patient has difficulty sleeping if they report:
1. Trouble initiating sleep
2. Trouble maintaining sleep
3. Waking up earlier than desired
4. An explicit mention of insomnia

**Definition 2 (Daytime Impairment)** — The patient has daytime impairment if they report:
1. Fatigue or malaise
2. Impaired attention, concentration, or memory
3. Impaired social/family/occupational/academic performance
4. Mood disturbance or irritability
5. Daytime sleepiness
6. Behavioral problems (hyperactivity, impulsivity, aggression)
7. Decreased motivation, energy, or initiative
8. Proneness to errors or accidents
9. Concerns or dissatisfaction with sleep
10. An explicit mention of insomnia

**Rule A**: Patient has insomnia if they meet BOTH Definition 1 AND Definition 2.

**Rule B**: Patient has insomnia if prescribed any PRIMARY insomnia medication:
Estazolam, Eszopiclone, Flurazepam, Lemborexant, Quazepam, Ramelteon, Suvorexant, Temazepam, Triazolam, Zaleplon, Zolpidem.

**Rule C**: Patient has insomnia if prescribed any SECONDARY insomnia medication AND reports any symptom from Definition 1 OR Definition 2:
Acamprosate, Alprazolam, Clonazepam, Clonidine, Diazepam, Diphenhydramine, Doxepin, Gabapentin, Hydroxyzine, Lorazepam, Melatonin, Mirtazapine, Olanzapine, Quetiapine, Trazodone.

**Insomnia status**: Label "yes" if Rule A OR Rule B OR Rule C is met; "no" otherwise.

**Important exclusions**:
- Symptoms must NOT be fully explained by another documented condition (comorbidity exclusion).
- For overnight nursing notes: only count symptoms from Definition 2 if they are explicit concerns about sleep or directly mention insomnia.
- Do NOT include medications used primarily for sedation, agitation, or pain (e.g., Haloperidol, Midazolam, Tramadol) as insomnia medications.
- Primary medications (Rule B) take precedence over secondary medications (Rule C).
"""

SYSTEM_PROMPT = (
    "You are a clinical NLP expert specializing in insomnia phenotyping from clinical notes. "
    "You apply diagnostic rules strictly and cite evidence from the note."
)

JSON_FORMAT_INSTRUCTION = """
Respond ONLY with a valid JSON object in this exact format:
{
  "definition_1": {"label": "yes" or "no", "explanation": "brief evidence from note or 'not found'"},
  "definition_2": {"label": "yes" or "no", "explanation": "brief evidence from note or 'not found'"},
  "rule_a":       {"label": "yes" or "no"},
  "rule_b":       {"label": "yes" or "no", "explanation": "medication name found in note or 'none'"},
  "rule_c":       {"label": "yes" or "no", "explanation": "medication name and symptom found or 'none'"},
  "final_label":  "yes" or "no"
}
Do not include any text outside the JSON object.
"""

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

def load_corpus(split: str) -> pd.DataFrame:
    data_dir = ""
    if split == "train":
        data_dir = TRAINING_DIR
    elif split == "val":
        data_dir = VALIDATION_DIR
    else:
        data_dir = TEST_DIR
    #data_dir = TRAINING_DIR if split == "train" else VALIDATION_DIR
    df = pd.read_csv(data_dir / "corpus.csv")
    df["note_id"] = df["note_id"].astype(str)
    return df


def load_labels(split: str) -> dict[str, str]:
    data_dir = TRAINING_DIR if split == "train" else VALIDATION_DIR
    with open(data_dir / "subtask_1.json") as f:
        raw = json.load(f)
    return {str(k): v["Insomnia"] for k, v in raw.items()}


def load_subtask2(split: str) -> dict:
    data_dir = TRAINING_DIR if split == "train" else VALIDATION_DIR
    with open(data_dir / "subtask_2.json") as f:
        return json.load(f)


def truncate_note(text: str, max_words: int = 1800) -> str:
    """Truncate note to fit within context window while preserving the header."""
    lines = text.split("\n")
    header = lines[0] if lines else ""
    body = "\n".join(lines[1:])
    body_words = body.split()
    if len(body_words) > max_words:
        body = " ".join(body_words[:max_words]) + "\n[... note truncated ...]"
    return header + "\n" + body


# ---------------------------------------------------------------------------
# Few-shot example construction
# ---------------------------------------------------------------------------

def build_cot_reasoning(note_id: str, s2_labels: dict, label: str) -> str:
    """Build chain-of-thought reasoning from subtask-2 annotations."""
    entry = s2_labels.get(note_id, {})

    def get(comp):
        return entry.get(comp, {}).get("label", "no")

    def get_text(comp):
        texts = entry.get(comp, {}).get("text", [])
        if texts:
            return "; ".join(f'"{t}"' for t in texts[:2])
        return "not found"

    def1_label = get("Definition 1")
    def2_label = get("Definition 2")
    rule_a = get("Rule A") if "Rule A" in entry else ("yes" if def1_label == "yes" and def2_label == "yes" else "no")
    rule_b = get("Rule B")
    rule_c = get("Rule C")

    def1_text = get_text("Definition 1")
    def2_text = get_text("Definition 2")
    rule_b_text = get_text("Rule B")
    rule_c_text = get_text("Rule C")

    reasoning = {
        "definition_1": {"label": def1_label, "explanation": def1_text},
        "definition_2": {"label": def2_label, "explanation": def2_text},
        "rule_a": {"label": rule_a},
        "rule_b": {"label": rule_b, "explanation": rule_b_text},
        "rule_c": {"label": rule_c, "explanation": rule_c_text},
        "final_label": label,
    }
    return json.dumps(reasoning, indent=2)


def select_fixed_few_shot_examples(
    corpus: pd.DataFrame,
    labels: dict[str, str],
    s2_labels: dict,
    n_per_class: int = 4,
    seed: int = 42,
) -> list[dict]:
    """Select n_per_class examples per class for the fixed few-shot set."""
    rng = np.random.default_rng(seed)
    yes_ids = [nid for nid, lbl in labels.items() if lbl == "yes"]
    no_ids = [nid for nid, lbl in labels.items() if lbl == "no"]

    rng.shuffle(yes_ids)
    rng.shuffle(no_ids)

    selected_ids = yes_ids[:n_per_class] + no_ids[:n_per_class]
    corpus_map = dict(zip(corpus["note_id"], corpus["text"]))

    examples = []
    for nid in selected_ids:
        if nid not in corpus_map:
            continue
        text = truncate_note(corpus_map[nid], max_words=600)
        label = labels[nid]
        reasoning = build_cot_reasoning(nid, s2_labels, label)
        examples.append({"note_id": nid, "text": text, "label": label, "reasoning": reasoning})
    return examples


def format_few_shot_block(examples: list[dict]) -> str:
    """Format few-shot examples as a prompt string."""
    blocks = []
    for i, ex in enumerate(examples, 1):
        block = f"""--- Example {i} ---
Clinical Note:
{ex['text']}

Analysis:
{ex['reasoning']}
"""
        blocks.append(block)
    return "\n".join(blocks)


def build_prompt(note_text: str, few_shot_examples: list[dict]) -> list[dict]:
    """Build the full few-shot + CoT prompt as a message list."""
    few_shot_block = format_few_shot_block(few_shot_examples)

    user_content = f"""{INSOMNIA_RULES_BLOCK}

## Instructions
Read the clinical note below and determine whether the patient has insomnia according to the rules above. Apply each rule systematically, citing evidence from the note.

## Few-Shot Examples
{few_shot_block}

## Clinical Note to Classify
{truncate_note(note_text, max_words=1800)}

## Task
{JSON_FORMAT_INSTRUCTION}"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Dynamic few-shot retrieval
# ---------------------------------------------------------------------------

def build_embeddings(corpus: pd.DataFrame, model_name: str = "BAAI/bge-large-en-v1.5"):
    """Embed all training notes using a sentence transformer."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Install sentence-transformers: pip install sentence-transformers")

    print(f"Loading embedding model: {model_name}")
    emb_model = SentenceTransformer(model_name)
    texts = corpus["text"].apply(lambda t: truncate_note(t, max_words=400)).tolist()
    print(f"Embedding {len(texts)} training notes...")
    embeddings = emb_model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return emb_model, embeddings


def retrieve_dynamic_examples(
    query_text: str,
    corpus: pd.DataFrame,
    labels: dict[str, str],
    s2_labels: dict,
    embeddings: np.ndarray,
    emb_model,
    n_per_class: int = 4,
    exclude_ids: Optional[set] = None,
) -> list[dict]:
    """Retrieve top-n_per_class yes and no examples most similar to query."""
    from sentence_transformers import util as st_util

    query_emb = emb_model.encode(
        truncate_note(query_text, max_words=400),
        normalize_embeddings=True
    )
    sims = np.dot(embeddings, query_emb)

    corpus_map = dict(zip(corpus["note_id"], corpus["text"]))
    note_ids = corpus["note_id"].tolist()

    yes_candidates = [(sims[i], nid) for i, nid in enumerate(note_ids)
                      if labels.get(nid) == "yes" and (exclude_ids is None or nid not in exclude_ids)]
    no_candidates = [(sims[i], nid) for i, nid in enumerate(note_ids)
                     if labels.get(nid) == "no" and (exclude_ids is None or nid not in exclude_ids)]

    yes_candidates.sort(reverse=True)
    no_candidates.sort(reverse=True)

    selected = []
    for _, nid in yes_candidates[:n_per_class]:
        text = truncate_note(corpus_map[nid], max_words=600)
        reasoning = build_cot_reasoning(nid, s2_labels, "yes")
        selected.append({"note_id": nid, "text": text, "label": "yes", "reasoning": reasoning})
    for _, nid in no_candidates[:n_per_class]:
        text = truncate_note(corpus_map[nid], max_words=600)
        reasoning = build_cot_reasoning(nid, s2_labels, "no")
        selected.append({"note_id": nid, "text": text, "label": "no", "reasoning": reasoning})

    return selected


# ---------------------------------------------------------------------------
# LLM inference
# ---------------------------------------------------------------------------

def get_client() -> OpenAI:
    api_key = os.environ.get("LLM_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def parse_json_response(response_text: str) -> Optional[dict]:
    """Extract and parse JSON from LLM response, handling markdown code fences."""
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", response_text).strip()
    text = text.rstrip("`").strip()
    # Find the JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def apply_logical_constraints(parsed: dict) -> dict:
    """Enforce hard logical rules as post-processing."""
    def1 = parsed.get("definition_1", {}).get("label", "no")
    def2 = parsed.get("definition_2", {}).get("label", "no")
    rule_a = parsed.get("rule_a", {})
    rule_b = parsed.get("rule_b", {})
    rule_c = parsed.get("rule_c", {})

    # Rule A requires both Def1 and Def2
    if not (def1 == "yes" and def2 == "yes"):
        rule_a["label"] = "no"

    # Final label follows from rules
    final = "yes" if (rule_a.get("label") == "yes" or
                      rule_b.get("label") == "yes" or
                      rule_c.get("label") == "yes") else "no"
    parsed["rule_a"] = rule_a
    parsed["final_label"] = final
    return parsed


def call_llm(
    client: OpenAI,
    messages: list[dict],
    model: str,
    temperature: float = 0.8,
    max_retries: int = 3,
) -> Optional[dict]:
    """Call the LLM and return parsed JSON output."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=512,
                seed=SEED,

                # for GPT variance
                #reasoning_effort="high"
            )
            text = response.choices[0].message.content or ""
            parsed = parse_json_response(text)
            if parsed and "final_label" in parsed:
                return apply_logical_constraints(parsed)
            print(f"  [WARN] Could not parse JSON (attempt {attempt+1}): {text[:200]}")
        except Exception as e:
            print(f"  [ERROR] LLM call failed (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return None


def majority_vote(predictions: list[Optional[dict]]) -> Optional[dict]:
    """Aggregate multiple predictions by majority vote on final_label."""
    labels = [p["final_label"] for p in predictions if p and "final_label" in p]
    if not labels:
        return None
    winner = Counter(labels).most_common(1)[0][0]
    # Return the first prediction that matches the winner
    for p in predictions:
        if p and p.get("final_label") == winner:
            return p
    return predictions[0]


# ---------------------------------------------------------------------------
# Main classification pipeline
# ---------------------------------------------------------------------------

def run_classification(
    mode: str,
    model: str,
    n_votes: int = 5,
    embed_model: str = "BAAI/bge-large-en-v1.5",
    output_path: Optional[str] = None,
    val_only: bool = True,
):
    print(f"\nSubtask 1 Classification — mode={mode}, model={model}")

    # Load data
    train_corpus = load_corpus("train")
    train_labels = load_labels("train")
    train_s2 = load_subtask2("train")
    val_corpus = load_corpus("val")
    val_labels = load_labels("val")
    test_corpus = load_corpus("test")

    client = get_client()
    print(f"LLM client initialized with base URL: {client.base_url}")
    print(f"client is {client}")

    # Build fixed few-shot examples (used in baseline and self-consistency)
    n_per_class = 5
    fixed_examples = select_fixed_few_shot_examples(train_corpus, train_labels, train_s2, n_per_class=n_per_class)
    print(f"Fixed few-shot examples: {len(fixed_examples)} "
          f"({sum(1 for e in fixed_examples if e['label']=='yes')} yes, "
          f"{sum(1 for e in fixed_examples if e['label']=='no')} no)")

    # Build embeddings for dynamic mode
    emb_model_obj = None
    train_embeddings = None
    if mode == "dynamic":
        emb_model_obj, train_embeddings = build_embeddings(train_corpus, embed_model)

    # Run on validation set
    corpus_to_run = test_corpus
    #labels_to_run = val_labels

    predictions = {}
    total = len(corpus_to_run)

    for idx, row in corpus_to_run.iterrows():
        note_id = str(row["note_id"])
        note_text = str(row["text"])
        print(f"  [{idx+1}/{total}] Note {note_id}", end=" ")

        # Select few-shot examples
        if mode == "dynamic" and emb_model_obj is not None:
            examples = retrieve_dynamic_examples(
                note_text, train_corpus, train_labels, train_s2,
                train_embeddings, emb_model_obj,
                exclude_ids={note_id},
                n_per_class=n_per_class
            )
        else:
            examples = fixed_examples

        messages = build_prompt(note_text, examples)

        if mode == "self_consist":
            # Run n_votes times with temperature=0.7
            votes = []
            for _ in range(n_votes):
                result = call_llm(client, messages, model, temperature=0.9)
                votes.append(result)
            pred = majority_vote(votes)
            if pred:
                pred["_votes"] = [v.get("final_label") if v else None for v in votes]
        else:
            pred = call_llm(client, messages, model, temperature=1)

        predictions[note_id] = pred or {"final_label": "no", "_error": "parse_failed"}
        print(f"→ {predictions[note_id].get('final_label', 'ERROR')}")

    # Convert to required format: {note_id: {"Insomnia": "yes"/"no"}}
    output_predictions = {}
    for note_id, pred in predictions.items():
        final_label = pred.get("final_label", "no") if isinstance(pred, dict) else "no"
        output_predictions[note_id] = {"Insomnia": final_label}

    # Save predictions
    if output_path is None:
        output_path = f"predictions_subtask1_{mode}.json"
    with open(output_path, "w") as f:
        json.dump(output_predictions, f, indent=2)
    print(f"\nPredictions saved to {output_path}")

    # Optionally save detailed predictions for debugging
    debug_path = output_path.replace(".json", "_debug.json")
    with open(debug_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"Detailed predictions saved to {debug_path}")

    return predictions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Subtask 1: Insomnia Binary Classification")
    parser.add_argument("--mode", choices=["baseline", "dynamic", "self_consist"],
                        default="baseline", help="Classification mode")
    parser.add_argument("--model", default=None,
                        help="Model name (overrides LLM_MODEL env var)")
    parser.add_argument("--n_votes", type=int, default=5,
                        help="Number of votes for self-consistency mode")
    parser.add_argument("--embed_model", default="BAAI/bge-large-en-v1.5",
                        help="Sentence transformer model for dynamic mode")
    parser.add_argument("--output", default=None,
                        help="Output JSON file path for predictions")
    args = parser.parse_args()

    model = args.model or os.environ.get("LLM_MODEL", "meta-llama/Llama-3.1-70B-Instruct")

    run_classification(
        mode=args.mode,
        model=model,
        n_votes=args.n_votes,
        embed_model=args.embed_model,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
