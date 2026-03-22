"""
MedAI – ClinicalBERT Model Evaluation Script
=============================================
Evaluates the fine-tuned ClinicalBERT model on the Disease and Symptoms Dataset.

Usage:
    1. Download the dataset from Kaggle:
       https://www.kaggle.com/datasets/choongqianzheng/disease-and-symptoms-dataset
    2. Place the CSV file(s) in: health_ai_project/backend/data/
    3. Run:  python evaluate_model.py

Outputs:
    - Classification report (precision, recall, F1 per disease)
    - Overall accuracy, macro/weighted averages
    - Confusion matrix heatmap saved as PNG
    - Results saved to evaluation_results.txt
"""

import os
import sys
import json
import time
import argparse
import numpy as np

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate ClinicalBERT on disease dataset")
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
    help="Fraction of data to use as test set (default: 0.2)",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=16,
    help="Batch size for inference (default: 16)",
)
parser.add_argument(
    "--output",
    type=str,
    default="evaluation_results.txt",
    help="Output file for results (default: evaluation_results.txt)",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Imports (heavy libraries loaded after arg parsing for --help speed)
# ---------------------------------------------------------------------------
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "ml_engine", "clinicalbert-disease")
MAX_LEN = 128
SEED = 42

# Load label mapping from model config
with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
    model_config = json.load(f)

ID2LABEL = {int(k): v for k, v in model_config["id2label"].items()}
LABEL2ID = {v: int(k) for k, v in model_config["id2label"].items()}
NUM_CLASSES = len(ID2LABEL)
DISEASE_NAMES = [ID2LABEL[i] for i in range(NUM_CLASSES)]


def load_dataset(data_dir: str):
    """
    Load the Disease and Symptoms Dataset from CSV files.
    Supports multiple common CSV formats from the Kaggle dataset.
    """
    import pandas as pd

    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not csv_files:
        print(f"\nERROR: No CSV files found in '{data_dir}'")
        print("Please download the dataset from:")
        print("  https://www.kaggle.com/datasets/choongqianzheng/disease-and-symptoms-dataset")
        print(f"And place the CSV file(s) in: {data_dir}")
        sys.exit(1)

    print(f"Found CSV files: {csv_files}")

    all_texts = []
    all_labels = []

    for csv_file in csv_files:
        filepath = os.path.join(data_dir, csv_file)
        df = pd.read_csv(filepath)
        print(f"\n  Loading: {csv_file} ({len(df)} rows, columns: {list(df.columns)})")

        # Auto-detect column format
        cols_lower = {c.lower().strip(): c for c in df.columns}

        # Format 1: "disease" + "symptoms" columns
        if "disease" in cols_lower and "symptoms" in cols_lower:
            disease_col = cols_lower["disease"]
            symptom_col = cols_lower["symptoms"]
            for _, row in df.iterrows():
                disease = str(row[disease_col]).strip().lower()
                symptoms = str(row[symptom_col]).strip()
                if disease in LABEL2ID and symptoms:
                    all_texts.append(symptoms)
                    all_labels.append(LABEL2ID[disease])

        # Format 2: "label" + "text" columns
        elif "label" in cols_lower and "text" in cols_lower:
            label_col = cols_lower["label"]
            text_col = cols_lower["text"]
            for _, row in df.iterrows():
                label = str(row[label_col]).strip().lower()
                text = str(row[text_col]).strip()
                if label in LABEL2ID and text:
                    all_texts.append(text)
                    all_labels.append(LABEL2ID[label])

        # Format 3: disease column + multiple symptom columns
        elif "disease" in cols_lower:
            disease_col = cols_lower["disease"]
            symptom_cols = [c for c in df.columns if c.lower() != "disease"]
            if symptom_cols:
                for _, row in df.iterrows():
                    disease = str(row[disease_col]).strip().lower()
                    if disease in LABEL2ID:
                        symptoms_list = [
                            str(row[sc]).strip()
                            for sc in symptom_cols
                            if pd.notna(row[sc]) and str(row[sc]).strip()
                        ]
                        if symptoms_list:
                            text = ", ".join(symptoms_list)
                            all_texts.append(text)
                            all_labels.append(LABEL2ID[disease])
        else:
            print(f"  WARNING: Could not detect column format for {csv_file}")

    if not all_texts:
        print("\nERROR: No valid data could be extracted.")
        print("Ensure CSV contains disease names matching the 22 model classes.")
        sys.exit(1)

    print(f"\nTotal samples loaded: {len(all_texts)}")
    print(f"Unique diseases found: {len(set(all_labels))}")
    return all_texts, all_labels


def evaluate():
    """Run full evaluation pipeline."""
    print("=" * 70)
    print("  MedAI — ClinicalBERT Disease Classification Evaluation")
    print("=" * 70)

    # --- Load dataset ---
    print(f"\n[1/5] Loading dataset from: {args.data}")
    texts, labels = load_dataset(args.data)

    # --- Train/test split ---
    print(f"\n[2/5] Splitting data (test_size={args.test_size}, seed={SEED})")
    _, test_texts, _, test_labels = train_test_split(
        texts, labels, test_size=args.test_size, random_state=SEED, stratify=labels
    )
    print(f"  Test samples: {len(test_texts)}")

    # --- Load model ---
    print(f"\n[3/5] Loading ClinicalBERT model from: {MODEL_DIR}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    # --- Run inference ---
    print(f"\n[4/5] Running inference (batch_size={args.batch_size})...")
    all_preds = []
    all_confs = []
    start_time = time.time()

    for i in range(0, len(test_texts), args.batch_size):
        batch_texts = test_texts[i : i + args.batch_size]
        encoding = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            confs = probs.max(dim=-1).values

        all_preds.extend(preds.cpu().numpy())
        all_confs.extend(confs.cpu().numpy())

        done = min(i + args.batch_size, len(test_texts))
        if (done % 100 == 0) or done == len(test_texts):
            print(f"    Processed {done}/{len(test_texts)}")

    elapsed = time.time() - start_time
    avg_latency = (elapsed / len(test_texts)) * 1000

    # --- Compute metrics ---
    print(f"\n[5/5] Computing metrics...")

    accuracy = accuracy_score(test_labels, all_preds)
    macro_precision = precision_score(test_labels, all_preds, average="macro", zero_division=0)
    macro_recall = recall_score(test_labels, all_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(test_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(test_labels, all_preds, average="weighted", zero_division=0)

    # Top-3 accuracy
    # Re-run to get top-3 for accuracy
    top3_correct = 0
    for i in range(0, len(test_texts), args.batch_size):
        batch_texts = test_texts[i : i + args.batch_size]
        batch_labels = test_labels[i : i + args.batch_size]
        encoding = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding)
            probs = torch.softmax(outputs.logits, dim=-1)
            top3 = torch.topk(probs, k=min(3, NUM_CLASSES), dim=-1).indices

        for j, label in enumerate(batch_labels):
            if label in top3[j].cpu().numpy():
                top3_correct += 1

    top3_accuracy = top3_correct / len(test_texts)

    # Per-class report
    report = classification_report(
        test_labels,
        all_preds,
        target_names=DISEASE_NAMES,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(test_labels, all_preds)

    # --- Build results text ---
    results_lines = []
    results_lines.append("=" * 70)
    results_lines.append("  MedAI — ClinicalBERT Evaluation Results")
    results_lines.append("=" * 70)
    results_lines.append(f"\nModel: ClinicalBERT (BertForSequenceClassification)")
    results_lines.append(f"Parameters: {param_count:,}")
    results_lines.append(f"Device: {device}")
    results_lines.append(f"Test Samples: {len(test_texts)}")
    results_lines.append(f"Inference Time: {elapsed:.2f}s ({avg_latency:.1f} ms/sample)")
    results_lines.append("")
    results_lines.append("-" * 70)
    results_lines.append("  OVERALL METRICS")
    results_lines.append("-" * 70)
    results_lines.append(f"  Accuracy           : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    results_lines.append(f"  Macro Precision    : {macro_precision:.4f}")
    results_lines.append(f"  Macro Recall       : {macro_recall:.4f}")
    results_lines.append(f"  Macro F1-Score     : {macro_f1:.4f}")
    results_lines.append(f"  Weighted F1-Score  : {weighted_f1:.4f}")
    results_lines.append(f"  Top-3 Accuracy     : {top3_accuracy:.4f}  ({top3_accuracy*100:.2f}%)")
    results_lines.append(f"  Avg Confidence     : {np.mean(all_confs):.4f}")
    results_lines.append("")
    results_lines.append("-" * 70)
    results_lines.append("  PER-CLASS CLASSIFICATION REPORT")
    results_lines.append("-" * 70)
    results_lines.append(report)
    results_lines.append("")
    results_lines.append("-" * 70)
    results_lines.append("  CONFUSION MATRIX")
    results_lines.append("-" * 70)
    results_lines.append("")

    # Format confusion matrix with labels
    max_name_len = max(len(n) for n in DISEASE_NAMES)
    header = " " * (max_name_len + 2) + "  ".join(f"{i:>3}" for i in range(NUM_CLASSES))
    results_lines.append(header)
    for i, row in enumerate(cm):
        name = DISEASE_NAMES[i].ljust(max_name_len)
        row_str = "  ".join(f"{v:>3}" for v in row)
        results_lines.append(f"{name}  {row_str}")

    results_text = "\n".join(results_lines)

    # --- Print and save ---
    print(results_text)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(results_text)
    print(f"\nResults saved to: {args.output}")

    # --- Save confusion matrix heatmap ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=DISEASE_NAMES,
            yticklabels=DISEASE_NAMES,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_xlabel("Predicted Disease", fontsize=12)
        ax.set_ylabel("True Disease", fontsize=12)
        ax.set_title("ClinicalBERT — Disease Classification Confusion Matrix", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()

        cm_path = "confusion_matrix.png"
        fig.savefig(cm_path, dpi=150)
        plt.close()
        print(f"Confusion matrix heatmap saved to: {cm_path}")
    except ImportError:
        print("  (matplotlib/seaborn not installed — skipping heatmap)")

    print("\nDone.")


if __name__ == "__main__":
    evaluate()
