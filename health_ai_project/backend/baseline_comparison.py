"""
MedAI – Baseline Model Comparison Script
=========================================
Trains and evaluates baseline ML models against ClinicalBERT for the
research paper's "Model Comparison" table.

Baselines:
    1. Naive Bayes (TF-IDF)
    2. SVM (TF-IDF)
    3. Random Forest (TF-IDF)
    4. Logistic Regression (TF-IDF)
    5. ClinicalBERT (fine-tuned)

Usage:
    1. Place the Kaggle CSV dataset in: health_ai_project/backend/data/
    2. Run:  python baseline_comparison.py

Outputs:
    - Comparison table (printed + saved to baseline_results.txt)
    - Per-model classification reports
"""

import os
import sys
import json
import time
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Baseline model comparison")
parser.add_argument(
    "--data",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "data"),
    help="Path to folder containing dataset CSV file(s)",
)
parser.add_argument(
    "--test-size",
    type=float,
    default=0.2,
    help="Test split fraction (default: 0.2)",
)
parser.add_argument(
    "--output",
    type=str,
    default="baseline_results.txt",
    help="Output file (default: baseline_results.txt)",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "ml_engine", "clinicalbert-disease")
MAX_LEN = 128
SEED = 42

with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
    model_config = json.load(f)

ID2LABEL = {int(k): v for k, v in model_config["id2label"].items()}
LABEL2ID = {v: int(k) for k, v in model_config["id2label"].items()}
NUM_CLASSES = len(ID2LABEL)
DISEASE_NAMES = [ID2LABEL[i] for i in range(NUM_CLASSES)]


def load_dataset(data_dir: str):
    """Load CSV dataset (same loader as evaluate_model.py)."""
    import pandas as pd

    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not csv_files:
        print(f"\nERROR: No CSV files found in '{data_dir}'")
        print("Download from: https://www.kaggle.com/datasets/choongqianzheng/disease-and-symptoms-dataset")
        sys.exit(1)

    all_texts, all_labels = [], []
    for csv_file in csv_files:
        filepath = os.path.join(data_dir, csv_file)
        df = pd.read_csv(filepath)
        cols_lower = {c.lower().strip(): c for c in df.columns}

        if "disease" in cols_lower and "symptoms" in cols_lower:
            disease_col, symptom_col = cols_lower["disease"], cols_lower["symptoms"]
            for _, row in df.iterrows():
                disease = str(row[disease_col]).strip().lower()
                symptoms = str(row[symptom_col]).strip()
                if disease in LABEL2ID and symptoms:
                    all_texts.append(symptoms)
                    all_labels.append(LABEL2ID[disease])
        elif "label" in cols_lower and "text" in cols_lower:
            for _, row in df.iterrows():
                label = str(row[cols_lower["label"]]).strip().lower()
                text = str(row[cols_lower["text"]]).strip()
                if label in LABEL2ID and text:
                    all_texts.append(text)
                    all_labels.append(LABEL2ID[label])
        elif "disease" in cols_lower:
            disease_col = cols_lower["disease"]
            symptom_cols = [c for c in df.columns if c.lower() != "disease"]
            for _, row in df.iterrows():
                disease = str(row[disease_col]).strip().lower()
                if disease in LABEL2ID:
                    parts = [str(row[sc]).strip() for sc in symptom_cols
                             if pd.notna(row[sc]) and str(row[sc]).strip()]
                    if parts:
                        all_texts.append(", ".join(parts))
                        all_labels.append(LABEL2ID[disease])

    if not all_texts:
        print("ERROR: No valid data extracted."); sys.exit(1)

    print(f"Loaded {len(all_texts)} samples, {len(set(all_labels))} diseases")
    return all_texts, all_labels


def evaluate_clinicalbert(test_texts, test_labels, device):
    """Evaluate ClinicalBERT on the test set."""
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification

    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    all_preds = []
    batch_size = 16
    start = time.time()

    for i in range(0, len(test_texts), batch_size):
        batch = test_texts[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            preds = torch.argmax(model(**enc).logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())

    elapsed = time.time() - start
    return np.array(all_preds), elapsed


def run_comparison():
    """Train baselines and compare with ClinicalBERT."""
    print("=" * 70)
    print("  MedAI — Baseline Model Comparison")
    print("=" * 70)

    # --- Load data ---
    texts, labels = load_dataset(args.data)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=args.test_size, random_state=SEED, stratify=labels
    )
    print(f"Train: {len(train_texts)} | Test: {len(test_texts)}")

    # --- TF-IDF vectorization ---
    print("\nBuilding TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = tfidf.fit_transform(train_texts)
    X_test = tfidf.transform(test_texts)
    print(f"  Vocabulary: {len(tfidf.vocabulary_)} features")

    # --- Define baseline models ---
    baselines = [
        ("Naive Bayes", MultinomialNB(alpha=1.0)),
        ("SVM (Linear)", LinearSVC(max_iter=5000, random_state=SEED)),
        ("Random Forest", RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=SEED, n_jobs=-1
        )),
        ("Logistic Regression", LogisticRegression(
            max_iter=2000, random_state=SEED, n_jobs=-1
        )),
    ]

    results = []
    detailed_reports = []

    # --- Evaluate each baseline ---
    for name, clf in baselines:
        print(f"\nTraining: {name} ...")
        start = time.time()
        clf.fit(X_train, train_labels)
        train_time = time.time() - start

        start = time.time()
        preds = clf.predict(X_test)
        infer_time = time.time() - start

        acc = accuracy_score(test_labels, preds)
        prec = precision_score(test_labels, preds, average="macro", zero_division=0)
        rec = recall_score(test_labels, preds, average="macro", zero_division=0)
        f1 = f1_score(test_labels, preds, average="macro", zero_division=0)
        wf1 = f1_score(test_labels, preds, average="weighted", zero_division=0)

        results.append({
            "name": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "weighted_f1": wf1,
            "train_time": train_time,
            "infer_time": infer_time,
        })

        report = classification_report(
            test_labels, preds, target_names=DISEASE_NAMES, digits=4, zero_division=0
        )
        detailed_reports.append((name, report))
        print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f} | Time: {train_time:.1f}s")

    # --- Evaluate ClinicalBERT ---
    print("\nEvaluating: ClinicalBERT (Proposed) ...")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_preds, bert_time = evaluate_clinicalbert(test_texts, test_labels, device)

    acc = accuracy_score(test_labels, bert_preds)
    prec = precision_score(test_labels, bert_preds, average="macro", zero_division=0)
    rec = recall_score(test_labels, bert_preds, average="macro", zero_division=0)
    f1 = f1_score(test_labels, bert_preds, average="macro", zero_division=0)
    wf1 = f1_score(test_labels, bert_preds, average="weighted", zero_division=0)

    results.append({
        "name": "ClinicalBERT (Proposed)",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "weighted_f1": wf1,
        "train_time": 0,
        "infer_time": bert_time,
    })

    report = classification_report(
        test_labels, bert_preds, target_names=DISEASE_NAMES, digits=4, zero_division=0
    )
    detailed_reports.append(("ClinicalBERT (Proposed)", report))
    print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f} | Time: {bert_time:.1f}s")

    # --- Build comparison table ---
    lines = []
    lines.append("=" * 90)
    lines.append("  MedAI — Model Comparison Results (Research Paper Table)")
    lines.append("=" * 90)
    lines.append("")
    lines.append(f"  Dataset: Disease and Symptoms Dataset (Kaggle)")
    lines.append(f"  Test Samples: {len(test_texts)}")
    lines.append(f"  Classes: {NUM_CLASSES}")
    lines.append("")

    # Table header
    lines.append("-" * 90)
    lines.append(
        f"  {'Model':<28} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} "
        f"{'F1-Score':>10} {'W-F1':>10}"
    )
    lines.append("-" * 90)

    for r in results:
        marker = " **" if "Proposed" in r["name"] else ""
        lines.append(
            f"  {r['name']:<28} {r['accuracy']:>9.4f} {r['precision']:>10.4f} "
            f"{r['recall']:>10.4f} {r['f1']:>10.4f} {r['weighted_f1']:>10.4f}{marker}"
        )

    lines.append("-" * 90)
    lines.append("")

    # Improvement over best baseline
    best_baseline_f1 = max(r["f1"] for r in results if "Proposed" not in r["name"])
    proposed_f1 = results[-1]["f1"]
    improvement = ((proposed_f1 - best_baseline_f1) / best_baseline_f1) * 100 if best_baseline_f1 > 0 else 0

    lines.append(f"  Best Baseline F1:   {best_baseline_f1:.4f}")
    lines.append(f"  ClinicalBERT F1:    {proposed_f1:.4f}")
    lines.append(f"  Improvement:        {improvement:+.2f}%")
    lines.append("")

    # Detailed reports
    lines.append("=" * 90)
    lines.append("  DETAILED PER-MODEL CLASSIFICATION REPORTS")
    lines.append("=" * 90)
    for name, report in detailed_reports:
        lines.append(f"\n{'─' * 70}")
        lines.append(f"  {name}")
        lines.append(f"{'─' * 70}")
        lines.append(report)

    output_text = "\n".join(lines)
    print("\n" + output_text)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(output_text)
    print(f"\nResults saved to: {args.output}")

    # --- Bar chart ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [r["name"] for r in results]
        accs = [r["accuracy"] * 100 for r in results]
        f1s = [r["f1"] * 100 for r in results]

        x = np.arange(len(names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width / 2, accs, width, label="Accuracy (%)", color="#4C8BF5")
        bars2 = ax.bar(x + width / 2, f1s, width, label="Macro F1 (%)", color="#EA4335")

        ax.set_ylabel("Score (%)", fontsize=12)
        ax.set_title("Model Comparison — Disease Classification Performance", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        fig.savefig("model_comparison.png", dpi=150)
        plt.close()
        print("Bar chart saved to: model_comparison.png")
    except ImportError:
        print("(matplotlib not installed — skipping chart)")

    print("\nDone.")


if __name__ == "__main__":
    run_comparison()
