# #SMM4H-HeaRD 2026: Binary Insomnia Classification from Clinical Notes Using LLMs with Chain-of-Thought Reasoning


This paper describes our system for Subtask 1 of the SMM4H HeaRD 2026 Task 2, which focuses on binary insomnia phenotyping from MIMIC-III clinical notes. Our approach leverages large language models (LLMs) with chain-of-thought (CoT) reasoning and implements three complementary strategies: (1) baseline fixed few-shot prompting, (2) dynamic example retrieval using semantic embeddings, and (3) self-consistency voting. 

## 1. Data

The `text_mimic_notes.py` Python script is designed to retrieve clinical notes and patient information from the [MIMIC-III Clinical Database (v1.4)](https://physionet.org/content/mimiciii/1.4/). The script takes a text file containing note IDs, and merges it with the content of the notes from MIMIC-III, including additional demographic and prescription information.

```bash
python text_mimic_notes.py --note_ids_path ./training/training_note_ids.txt  --mimic_path ./mimic-iii --output_path ./training/corpus.csv


python text_mimic_notes.py --note_ids_path ./validation/validation_note_ids.txt  --mimic_path ./mimic-iii --output_path ./validation/corpus.csv


python text_mimic_notes.py --note_ids_path ./test/test_note_ids.txt  --mimic_path ./mimic-iii --output_path ./test/corpus.csv
```

## 2. System Architecture

### 2.1 Overview

Our system architecture implements an LLM-based classification pipeline with three core components:

1. **Data Preprocessing:** Note enrichment, medication normalization, and note type identification
2. **LLM-based Classification:** Few-shot prompting with chain-of-thought reasoning
3. **Post-processing:** Logical constraint enforcement and structured output formatting

The system is designed to operate in three distinct modes (baseline, dynamic, self-consistency), each offering different tradeoffs between computational cost, context utilization, and prediction robustness.

### 2.2 Preprocessing Pipeline

#### 2.2.1 Medication Normalization

Clinical notes contain medication names in various forms (brand names, generic names, dosage forms). We implement a robust medication extraction and normalization pipeline:

- **Structured extraction:** Parse medication names from the enriched note header
- **Text-based extraction:** Pattern matching for primary and secondary insomnia medications within note text
- **Normalization:** Map medication mentions to canonical forms (e.g., "Ambien" → "zolpidem", "Remeron" → "mirtazapine")

The system maintains two medication lists:
- **Primary insomnia medications (11 total):** Estazolam, Eszopiclone, Flurazepam, Lemborexant, Quazepam, Ramelteon, Suvorexant, Temazepam, Triazolam, Zaleplon, Zolpidem
- **Secondary insomnia medications (15 total):** Acamprosate, Alprazolam, Clonazepam, Clonidine, Diazepam, Diphenhydramine, Doxepin, Gabapentin, Hydroxyzine, Lorazepam, Melatonin, Mirtazapine, Olanzapane, Quetiapine, Trazodone

#### 2.2.2 Note Type Classification

Notes are classified into three categories based on metadata and heuristic patterns:
- **Discharge summaries:** Comprehensive patient summaries at hospital discharge
- **Nursing notes:** Shift-based nursing observations
- **Other:** Miscellaneous clinical documentation

Special handling is applied to overnight nursing notes (timestamps 1900-0700), where Definition 2 symptoms are evaluated more conservatively due to context (symptoms documented during overnight shifts may not reflect daytime impairment).

#### 2.2.3 Note Truncation

To optimize LLM token usage while preserving critical information, notes are strategically truncated:
- **Few-shot examples:** Truncated to 600 words for context efficiency
- **Test notes:** Truncated to 1,800 words (retaining enriched header)
- **Truncation strategy:** Preserve header line (demographics + medications), truncate body text with indicator when truncation occurs

---

## 3. Methods

### 3.1 Baseline Approach: Fixed Few-Shot Prompting

#### 3.1.1 Few-Shot Example Selection

The baseline system uses 8-shot learning with balanced class representation:
- **4 positive examples:** Notes with insomnia label "yes"
- **4 negative examples:** Notes with insomnia label "no"
- **Selection criteria:** Random sampling with fixed seed (seed=4351) for reproducibility
- **Example format:** Clinical note text + gold-standard reasoning in structured JSON format

#### 3.1.2 Chain-of-Thought (CoT) Reasoning

Each few-shot example includes explicit chain-of-thought reasoning that demonstrates the step-by-step evaluation process:

```json
{
  "definition_1": {"label": "yes" or "no", "explanation": "brief evidence from note or 'not found'"},
  "definition_2": {"label": "yes" or "no", "explanation": "brief evidence from note or 'not found'"},
  "rule_a":       {"label": "yes" or "no"},
  "rule_b":       {"label": "yes" or "no", "explanation": "medication name found in note or 'none'"},
  "rule_c":       {"label": "yes" or "no", "explanation": "medication name and symptom found or 'none'"},
  "final_label":  "yes" or "no"
}
```

This structured reasoning format serves multiple purposes:
1. **Teaches the LLM** the evaluation logic through demonstration
2. **Enforces systematic evaluation** of all rule components
3. **Provides interpretability** for predictions
4. **Enables post-hoc analysis** of model reasoning

#### 3.1.3 Prompt Engineering

The system prompt establishes the LLM's role and expertise:
```
You are a clinical NLP expert specializing in insomnia phenotyping from clinical 
notes. You apply diagnostic rules strictly and cite evidence from the note.
```

The user prompt contains:
1. **Insomnia rules definition** (Definitions 1 & 2, Rules A, B, C)
2. **Critical exclusion rules:**
   - Comorbidity exclusion: Symptoms fully explained by other documented conditions must be excluded
   - Night-shift nursing note handling: Conservative Definition 2 evaluation
   - Medication precedence: Primary medications (Rule B) take precedence over secondary (Rule C)
3. **Step-by-step instructions** for systematic evaluation
4. **Few-shot examples** demonstrating correct reasoning
5. **Test note** to classify
6. **Output format specification** (structured JSON)

#### 3.1.4 LLM Configuration

- **Model:** GPT-4.1 (via Azure OpenAI endpoint) or meta-llama/Llama-3.1-70B-Instruct
- **Temperature:** 0.0 (greedy decoding for deterministic predictions)
- **Max tokens:** 768 (sufficient for structured JSON response)
- **API:** OpenAI-compatible interface for model flexibility

### 3.2 Dynamic Few-Shot Retrieval

The dynamic approach enhances baseline prompting by selecting contextually relevant examples for each test note.

#### 3.2.1 Semantic Embedding

We employ sentence-transformers with the BAAI/bge-large-en-v1.5 model:
- **Embedding dimension:** 1,024
- **Normalization:** L2 normalization for cosine similarity
- **Text preparation:** Notes truncated to 400 words before embedding (for consistency)

#### 3.2.2 Example Retrieval Algorithm

For each test note:
1. **Encode test note** using sentence-transformers
2. **Compute similarity** with all training notes (dot product of normalized embeddings)
3. **Rank candidates** by similarity separately for positive and negative classes
4. **Select top-4 positive** and **top-4 negative** examples
5. **Exclude test note ID** from candidates (prevent train-test leakage in cross-validation scenarios)

#### 3.2.3 Rationale

Dynamic retrieval offers several advantages:
- **Context-aware examples:** Examples similar to test note may better demonstrate relevant reasoning patterns
- **Improved few-shot coverage:** Different test notes receive different examples, potentially improving generalization
- **Adaptability:** System adapts to diverse note characteristics (length, note type, clinical context)

### 3.3 Self-Consistency Voting

The self-consistency approach enhances prediction robustness through multiple sampling and majority voting [2].

#### 3.3.1 Multi-Pass Sampling

- **Number of passes:** 5 independent inferences per test note
- **Temperature:** 0.7 (enables stochastic sampling while maintaining quality)
- **Base prompt:** Identical to baseline fixed few-shot prompt

#### 3.3.2 Component-Level Voting

Rather than simple majority vote on the final label, we implement component-level voting:

1. **Vote on each component** independently:
   - Definition 1: yes/no
   - Definition 2: yes/no
   - Rule A: yes/no
   - Rule B: yes/no
   - Rule C: yes/no

2. **Select majority label** for each component (breaks ties arbitrarily)

3. **Select representative explanation:** Choose explanation from a vote that matches the majority label

4. **Apply logical constraints** to ensure consistency:
   - Rule A = Definition 1 AND Definition 2
   - Final label = Rule A OR Rule B OR Rule C

#### 3.3.3 Rationale

Component-level voting provides several benefits:
- **Fine-grained robustness:** Reduces impact of stochastic errors in individual components
- **Maintains logical consistency:** Post-voting constraint enforcement ensures valid rule combinations
- **Preserves interpretability:** Voted explanations provide insight into model confidence
- **Handles ambiguous cases:** Multiple perspectives on borderline cases may improve accuracy
