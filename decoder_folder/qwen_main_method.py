import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
import gc
import json
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, confusion_matrix

warnings.filterwarnings("ignore")


ROOT = Path("thesis/")
DATA_FILE = ROOT / "combined_arabic_balanced_1200_new_bestest_no_@.csv"

XED_ENGLISH = ROOT / "xed_english.csv"
BRIGHTER_ENGLISH = ROOT / "brighter_english.csv"
    
XED_ARABIC = ROOT / "xed_arabic.csv"
BRIGHTER_ARABIC = ROOT / "brighter_arabic.csv"

from scipy.linalg import orthogonal_procrustes

# CHINESE
XED_CHINESE = ROOT / "xed_chinese.csv"
XED_TW = ROOT / "xed_tw.csv"
BRIGHTER_CHINESE = ROOT / "brighter_chinese.csv"

chinese_eval_df = pd.concat(
    map(pd.read_csv, [XED_CHINESE, XED_TW, BRIGHTER_CHINESE]),
    ignore_index=True
)
chinese_eval_df["text"] = chinese_eval_df["text"].astype(str)
chinese_eval_df["emotion"] = chinese_eval_df["emotion"].astype(str)

# ESTONIAN
XED_ESTONIAN = ROOT / "xed_estonian.csv"

estonian_eval_df = pd.concat(
    map(pd.read_csv, [XED_ESTONIAN]),
    ignore_index=True
)
estonian_eval_df["text"] = estonian_eval_df["text"].astype(str)
estonian_eval_df["emotion"] = estonian_eval_df["emotion"].astype(str)

# RUSSIAN
XED_RUSSIAN = ROOT / "xed_russian.csv"
BRIGHTER_RUSSIAN = ROOT / "brighter_russian.csv"

russian_eval_df = pd.concat(
    map(pd.read_csv, [XED_RUSSIAN, BRIGHTER_RUSSIAN]),
    ignore_index=True
)
russian_eval_df["text"] = russian_eval_df["text"].astype(str)
russian_eval_df["emotion"] = russian_eval_df["emotion"].astype(str)

# SWAHILI
BRIGHTER_SWAHILI = ROOT / "brighter_swahili.csv"

swahili_eval_df = pd.concat(
    map(pd.read_csv, [BRIGHTER_SWAHILI]),
    ignore_index=True
)
swahili_eval_df["text"] = swahili_eval_df["text"].astype(str)
swahili_eval_df["emotion"] = swahili_eval_df["emotion"].astype(str)

# UKRAINIAN
BRIGHTER_UKRAINIAN = ROOT / "brighter_ukrainian.csv"

ukrainian_eval_df = pd.concat(
    map(pd.read_csv, [BRIGHTER_UKRAINIAN]),
    ignore_index=True
)
ukrainian_eval_df["text"] = ukrainian_eval_df["text"].astype(str)
ukrainian_eval_df["emotion"] = ukrainian_eval_df["emotion"].astype(str)

# GERMAN
XED_GERMAN = ROOT / "xed_german.csv"
BRIGHTER_GERMAN = ROOT / "brighter_german.csv"

german_eval_df = pd.concat(
    map(pd.read_csv, [BRIGHTER_GERMAN, XED_GERMAN]),
    ignore_index=True
)
german_eval_df["text"] = german_eval_df["text"].astype(str)
german_eval_df["emotion"] = german_eval_df["emotion"].astype(str)

english_eval_df = pd.concat(
    map(pd.read_csv, [XED_ENGLISH, BRIGHTER_ENGLISH]),
    ignore_index=True
)
english_eval_df["text"] = english_eval_df["text"].astype(str)
english_eval_df["emotion"] = english_eval_df["emotion"].astype(str)

arabic_eval_df = pd.concat(
    map(pd.read_csv, [XED_ARABIC, BRIGHTER_ARABIC]),
    ignore_index=True
)
arabic_eval_df["text"] = arabic_eval_df["text"].astype(str)
arabic_eval_df["emotion"] = arabic_eval_df["emotion"].astype(str)

MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]

MODEL_BATCH_SIZES = {
    "Qwen/Qwen2.5-1.5B-Instruct": 8,
    "Qwen/Qwen2.5-3B-Instruct": 4,
    "Qwen/Qwen2.5-7B-Instruct": 2,
}

BASE_OUTPUT_DIR = "outputs_multilingual"
MAX_LENGTH = 128
RANDOM_STATE = 42
MAX_TSNE_POINTS = 6000

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)


# Reproducibility
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

# Load data
df = pd.read_csv(DATA_FILE)
df["text"] = df["text"].astype(str)
df["emotion"] = df["emotion"].astype(str)

texts = df["text"].tolist()
labels = df["emotion"].tolist()
emotion_order = sorted(df["emotion"].unique())

print("Rows:", len(df))
print("Emotions:", emotion_order)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Helpers
def safe_model_id(model_name: str) -> str:
    return model_name.split("/")[-1].replace(" ", "_")

def mean_pooling(hidden_states, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    summed = torch.sum(hidden_states * input_mask_expanded, dim=1)
    counts = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts

def last_token_pooling(hidden_states, attention_mask):
    last_token_indices = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_indices, last_token_indices, :]

def encode_texts_decoder(text_list, tokenizer, model, device, batch_size=8, max_length=128):
    mean_embs = []
    last_embs = []

    with torch.no_grad():
        for i in tqdm(range(0, len(text_list), batch_size), desc="Encoding", leave=False):
            batch_texts = text_list[i:i + batch_size]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            outputs = model(
                **encoded,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden = outputs.hidden_states[-1]  # (B, T, d)

            mean_emb = mean_pooling(hidden, encoded["attention_mask"])
            last_emb = last_token_pooling(hidden, encoded["attention_mask"])

            mean_embs.append(mean_emb.float().cpu().numpy())
            last_embs.append(last_emb.float().cpu().numpy())

    return np.vstack(mean_embs), np.vstack(last_embs)

def compute_centroids(embeddings, df_local, emotion_order_local):
    centroids = []
    for emotion in emotion_order_local:
        mask = (df_local["emotion"] == emotion).values
        centroids.append(embeddings[mask].mean(axis=0))
    centroids = np.vstack(centroids)
    return normalize(centroids, norm="l2")

def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def append_text(path, text):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)

def dataframe_to_string(df_obj, title=None):
    parts = []
    if title:
        parts.append(title)
    parts.append(df_obj.to_string())
    return "\n".join(parts) + "\n"

def get_probe_space(probe, embeddings, use_probabilities=False):
    if use_probabilities:
        return probe.predict_proba(embeddings)
    scores = probe.decision_function(embeddings)
    if scores.ndim == 1:
        scores = scores[:, None]
    return scores
    

# Training
def train_probe(embeddings, labels_local, name):
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        labels_local,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels_local,
    )

    clf = LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_test, y_pred), 4),
        "macro_f1": round(f1_score(y_test, y_pred, average="macro"), 4),
        "weighted_f1": round(f1_score(y_test, y_pred, average="weighted"), 4),
    }

    report = classification_report(y_test, y_pred, digits=3)

    text = []
    text.append(f"=== {name} probe ===")
    text.append(f"Accuracy: {metrics['accuracy']}")
    text.append(f"Balanced accuracy: {metrics['balanced_accuracy']}")
    text.append(f"Macro F1: {metrics['macro_f1']}")
    text.append(f"Weighted F1: {metrics['weighted_f1']}")
    text.append(report)

    return clf, metrics, "\n".join(text) + "\n"

def predict_examples_decoder(text_list, tokenizer, model, device, probe, use_mean=True):
    mean_embs, last_embs = encode_texts_decoder(
        text_list,
        tokenizer,
        model,
        device,
        batch_size=8,
        max_length=MAX_LENGTH,
    )

    embs = mean_embs if use_mean else last_embs
    embs = normalize(embs, norm="l2")

    preds = probe.predict(embs)
    probs = probe.predict_proba(embs)

    lines = []
    for text, pred, prob in zip(text_list, preds, probs):
        top_idx = np.argsort(prob)[::-1][:3]
        top3 = [(probe.classes_[i], round(float(prob[i]), 3)) for i in top_idx]

        lines.append(f"TEXT: {text}")
        lines.append(f"Predicted: {pred}")
        lines.append(f"Top-3: {top3}")
        lines.append("")

    return "\n".join(lines)
    

def encode_eval_texts_decoder(eval_df, tokenizer, model, device, batch_size=8, max_length=128):
    texts = eval_df["text"].astype(str).tolist()
    mean_embs, last_embs = encode_texts_decoder(
        texts,
        tokenizer,
        model,
        device,
        batch_size=batch_size,
        max_length=max_length,
    )
    return normalize(mean_embs, norm="l2"), normalize(last_embs, norm="l2"), eval_df["emotion"].astype(str).tolist()

def evaluate_probe_on_single_label_text(embeddings, probe, gold_labels, class_order, name="Probe"):
    preds = probe.predict(embeddings)
    probs = probe.predict_proba(embeddings)

    lines = []
    lines.append(f"=== {name} ===")
    lines.append(f"Accuracy: {round(accuracy_score(gold_labels, preds), 4)}")
    lines.append(f"Balanced accuracy: {round(balanced_accuracy_score(gold_labels, preds), 4)}")
    lines.append(f"Macro F1: {round(f1_score(gold_labels, preds, average='macro'), 4)}")
    lines.append(f"Weighted F1: {round(f1_score(gold_labels, preds, average='weighted'), 4)}")
    lines.append(f"Log loss: {round(log_loss(gold_labels, probs, labels=class_order), 4)}")
    lines.append("")
    lines.append("Classification report:")
    lines.append(classification_report(gold_labels, preds, digits=3))

    cm = confusion_matrix(gold_labels, preds, labels=class_order)
    cm_df = pd.DataFrame(cm, index=class_order, columns=class_order)
    lines.append("Confusion matrix:")
    lines.append(cm_df.to_string())
    lines.append("")

    pred_idx = np.argmax(probs, axis=1)
    pred_conf = probs[np.arange(len(probs)), pred_idx]

    confidence_df = pd.DataFrame({
        "gold": gold_labels,
        "pred": preds,
        "confidence": pred_conf
    })

    per_class_conf = (
        confidence_df.groupby("gold")["confidence"]
        .agg(["mean", "std", "count"])
        .round(4)
        .rename_axis("emotion")
        .reset_index()
    )

    lines.append("Per-class confidence stats:")
    lines.append(per_class_conf.to_string(index=False))
    lines.append("")

    metrics = {
        "accuracy": round(accuracy_score(gold_labels, preds), 4),
        "balanced_accuracy": round(balanced_accuracy_score(gold_labels, preds), 4),
        "macro_f1": round(f1_score(gold_labels, preds, average="macro"), 4),
        "weighted_f1": round(f1_score(gold_labels, preds, average="weighted"), 4),
        "log_loss": round(log_loss(gold_labels, probs, labels=class_order), 4),
    }

    return "\n".join(lines) + "\n", metrics

def run_calibrated_evaluation(
    eval_df,
    tokenizer,
    model,
    device,
    batch_size,
    max_length,
    mean_probe,
    last_probe,
    class_order,
    dataset_name,
):
    
    eval_df = balance_eval_df(eval_df, class_order, random_state=RANDOM_STATE)
    
    mean_emb, last_emb, labels = encode_eval_texts_decoder(
        eval_df,
        tokenizer,
        model,
        device,
        batch_size=batch_size,
        max_length=max_length,
    )

    mean_X_calib, mean_X_test, y_calib, y_test = train_test_split(
        mean_emb,
        labels,
        test_size=0.5,
        random_state=42,
        stratify=labels,
    )

    last_X_calib, last_X_test, _, _ = train_test_split(
        last_emb,
        labels,
        test_size=0.5,
        random_state=42,
        stratify=labels,
    )

    text_blocks = []
    summary = {}

    text_blocks.append(f"\n===== {dataset_name.upper()} EVALUATION =====\n")
    text_blocks.append(f"Rows: {len(eval_df)}")
    text_blocks.append(f"Calibration size: {len(y_calib)}")
    text_blocks.append(f"Test size: {len(y_test)}\n")

    # Uncalibrated
    txt, metrics = evaluate_probe_on_single_label_text(
        mean_X_test, mean_probe, y_test, class_order,
        name=f"{dataset_name} | Mean pooled | uncalibrated"
    )
    text_blocks.append(txt)
    summary[f"{dataset_name}_mean_uncal_macro_f1"] = metrics["macro_f1"]
    summary[f"{dataset_name}_mean_uncal_logloss"] = metrics["log_loss"]

    txt, metrics = evaluate_probe_on_single_label_text(
        last_X_test, last_probe, y_test, class_order,
        name=f"{dataset_name} | Last token | uncalibrated"
    )
    text_blocks.append(txt)
    summary[f"{dataset_name}_last_uncal_macro_f1"] = metrics["macro_f1"]
    summary[f"{dataset_name}_last_uncal_logloss"] = metrics["log_loss"]

    # Sigmoid
    mean_probe_sigmoid = CalibratedClassifierCV(
        estimator=mean_probe,
        method="sigmoid",
        cv="prefit",
    )
    mean_probe_sigmoid.fit(mean_X_calib, y_calib)

    last_probe_sigmoid = CalibratedClassifierCV(
        estimator=last_probe,
        method="sigmoid",
        cv="prefit",
    )
    last_probe_sigmoid.fit(last_X_calib, y_calib)

    txt, metrics = evaluate_probe_on_single_label_text(
        mean_X_test, mean_probe_sigmoid, y_test, class_order,
        name=f"{dataset_name} | Mean pooled | sigmoid"
    )
    text_blocks.append(txt)
    summary[f"{dataset_name}_mean_sigmoid_macro_f1"] = metrics["macro_f1"]
    summary[f"{dataset_name}_mean_sigmoid_logloss"] = metrics["log_loss"]

    txt, metrics = evaluate_probe_on_single_label_text(
        last_X_test, last_probe_sigmoid, y_test, class_order,
        name=f"{dataset_name} | Last token | sigmoid"
    )
    text_blocks.append(txt)
    summary[f"{dataset_name}_last_sigmoid_macro_f1"] = metrics["macro_f1"]
    summary[f"{dataset_name}_last_sigmoid_logloss"] = metrics["log_loss"]

    # Isotonic
    mean_probe_isotonic = CalibratedClassifierCV(
        estimator=mean_probe,
        method="isotonic",
        cv="prefit",
    )
    mean_probe_isotonic.fit(mean_X_calib, y_calib)

    last_probe_isotonic = CalibratedClassifierCV(
        estimator=last_probe,
        method="isotonic",
        cv="prefit",
    )
    last_probe_isotonic.fit(last_X_calib, y_calib)

    txt, metrics = evaluate_probe_on_single_label_text(
        mean_X_test, mean_probe_isotonic, y_test, class_order,
        name=f"{dataset_name} | Mean pooled | isotonic"
    )
    text_blocks.append(txt)
    summary[f"{dataset_name}_mean_isotonic_macro_f1"] = metrics["macro_f1"]
    summary[f"{dataset_name}_mean_isotonic_logloss"] = metrics["log_loss"]

    txt, metrics = evaluate_probe_on_single_label_text(
        last_X_test, last_probe_isotonic, y_test, class_order,
        name=f"{dataset_name} | Last token | isotonic"
    )
    text_blocks.append(txt)
    summary[f"{dataset_name}_last_isotonic_macro_f1"] = metrics["macro_f1"]
    summary[f"{dataset_name}_last_isotonic_logloss"] = metrics["log_loss"]

    return "\n".join(text_blocks), summary

def balance_eval_df(eval_df, class_order, random_state=42):
    eval_df = eval_df[eval_df["emotion"].isin(class_order)].copy()

    target_n = eval_df["emotion"].value_counts().min()

    parts = []
    for emotion, group in eval_df.groupby("emotion"):
        parts.append(group.sample(n=target_n, random_state=random_state))

    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

def compute_label_centroids(X, y, class_order):
    y = np.array(y)
    centroids = []

    for emotion in class_order:
        mask = y == emotion
        centroids.append(X[mask].mean(axis=0))

    return normalize(np.vstack(centroids), norm="l2")

def fit_procrustes_from_centroids(X_calib, y_calib, target_centroids, class_order):
    source_centroids = compute_label_centroids(X_calib, y_calib, class_order)
    R, scale = orthogonal_procrustes(source_centroids, target_centroids)
    return R, scale

def apply_rotation(X, R):
    return normalize(X @ R, norm="l2")

def project_to_probe_subspace(X, probe):
    return X @ probe.coef_.copy().T

def evaluate_6d_logreg_text(Z_calib, Z_test, y_calib, y_test, class_order, name):
    clf = LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    clf.fit(Z_calib, y_calib)

    preds = clf.predict(Z_test)
    probs = clf.predict_proba(Z_test)

    lines = []
    lines.append(f"=== {name}: LogReg on 6D projected emotion space ===")
    lines.append(f"6D train shape: {Z_calib.shape}")
    lines.append(f"6D test shape: {Z_test.shape}")
    lines.append(f"Accuracy: {round(accuracy_score(y_test, preds), 4)}")
    lines.append(f"Balanced accuracy: {round(balanced_accuracy_score(y_test, preds), 4)}")
    lines.append(f"Macro F1: {round(f1_score(y_test, preds, average='macro'), 4)}")
    lines.append(f"Weighted F1: {round(f1_score(y_test, preds, average='weighted'), 4)}")
    lines.append(f"Log loss: {round(log_loss(y_test, probs, labels=class_order), 4)}")
    lines.append("")
    lines.append("Classification report:")
    lines.append(classification_report(y_test, preds, digits=3))

    cm = confusion_matrix(y_test, preds, labels=class_order)
    cm_df = pd.DataFrame(cm, index=class_order, columns=class_order)

    lines.append("Confusion matrix:")
    lines.append(cm_df.to_string())
    lines.append("")

    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_test, preds), 4),
        "macro_f1": round(f1_score(y_test, preds, average="macro"), 4),
        "weighted_f1": round(f1_score(y_test, preds, average="weighted"), 4),
        "log_loss": round(log_loss(y_test, probs, labels=class_order), 4),
    }

    return "\n".join(lines) + "\n", metrics

def run_crosslingual_projection_eval(
    eval_df,
    tokenizer,
    model,
    device,
    batch_size,
    max_length,
    target_mean_centroids,
    target_last_centroids,
    target_mean_probe,
    target_last_probe,
    class_order,
    dataset_name,
    target_name,
):
    eval_df = balance_eval_df(eval_df, class_order, random_state=RANDOM_STATE)

    mean_emb, last_emb, labels = encode_eval_texts_decoder(
        eval_df,
        tokenizer,
        model,
        device,
        batch_size=batch_size,
        max_length=max_length,
    )

    mean_X_calib, mean_X_test, y_calib, y_test = train_test_split(
        mean_emb,
        labels,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    last_X_calib, last_X_test, _, _ = train_test_split(
        last_emb,
        labels,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    text_blocks = []
    summary = {}

    text_blocks.append(f"{dataset_name.upper()} → {target_name.upper()} PROJECTION EVAL\n")
    text_blocks.append(f"Rows after balancing: {len(eval_df)}")
    text_blocks.append(f"Calibration size: {len(y_calib)}")
    text_blocks.append(f"Test size: {len(y_test)}\n")

    mean_R, mean_scale = fit_procrustes_from_centroids(
        mean_X_calib,
        y_calib,
        target_mean_centroids,
        class_order,
    )

    last_R, last_scale = fit_procrustes_from_centroids(
        last_X_calib,
        y_calib,
        target_last_centroids,
        class_order,
    )

    text_blocks.append(f"Mean Procrustes scale: {round(mean_scale, 4)}")
    text_blocks.append(f"Last-token Procrustes scale: {round(last_scale, 4)}\n")

    mean_X_calib_aligned = apply_rotation(mean_X_calib, mean_R)
    mean_X_test_aligned = apply_rotation(mean_X_test, mean_R)

    last_X_calib_aligned = apply_rotation(last_X_calib, last_R)
    last_X_test_aligned = apply_rotation(last_X_test, last_R)

    txt, metrics = evaluate_probe_on_single_label_text(
        mean_X_test_aligned,
        target_mean_probe,
        y_test,
        class_order,
        name=f"{dataset_name} → {target_name} | Mean pooled | Procrustes + target probe",
    )
    text_blocks.append(txt)
    summary[f"{dataset_name}_to_{target_name}_mean_proc_probe_macro_f1"] = metrics["macro_f1"]

    txt, metrics = evaluate_probe_on_single_label_text(
        last_X_test_aligned,
        target_last_probe,
        y_test,
        class_order,
        name=f"{dataset_name} → {target_name} | Last token | Procrustes + target probe",
    )
    text_blocks.append(txt)
    summary[f"{dataset_name}_to_{target_name}_last_proc_probe_macro_f1"] = metrics["macro_f1"]

    mean_Z_calib = project_to_probe_subspace(mean_X_calib_aligned, target_mean_probe)
    mean_Z_test = project_to_probe_subspace(mean_X_test_aligned, target_mean_probe)

    txt, metrics = evaluate_6d_logreg_text(
        mean_Z_calib,
        mean_Z_test,
        y_calib,
        y_test,
        class_order,
        name=f"{dataset_name} → {target_name} | Mean pooled | Procrustes",
    )
    text_blocks.append(txt)
    summary[f"{dataset_name}_to_{target_name}_mean_proc_6d_macro_f1"] = metrics["macro_f1"]

    last_Z_calib = project_to_probe_subspace(last_X_calib_aligned, target_last_probe)
    last_Z_test = project_to_probe_subspace(last_X_test_aligned, target_last_probe)

    txt, metrics = evaluate_6d_logreg_text(
        last_Z_calib,
        last_Z_test,
        y_calib,
        y_test,
        class_order,
        name=f"{dataset_name} → {target_name} | Last token | Procrustes",
    )
    text_blocks.append(txt)
    summary[f"{dataset_name}_to_{target_name}_last_proc_6d_macro_f1"] = metrics["macro_f1"]

    return "\n".join(text_blocks), summary


# Main loop
summary_rows = []

for model_name in MODELS:
    print(f"\n\n===== RUNNING MODEL: {model_name} =====")

    model_id = safe_model_id(model_name)
    output_dir = os.path.join(BASE_OUTPUT_DIR, model_id)
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "results.txt")
    save_text(log_file, f"MODEL: {model_name}\n\n")

    batch_size = MODEL_BATCH_SIZES.get(model_name, 4)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
    )
    model.to(device)
    model.eval()

    # Encode
    mean_emb, last_emb = encode_texts_decoder(
        texts,
        tokenizer,
        model,
        device,
        batch_size=batch_size,
        max_length=MAX_LENGTH,
    )

    append_text(log_file, f"Mean embeddings shape: {mean_emb.shape}\n")
    append_text(log_file, f"Last embeddings shape: {last_emb.shape}\n\n")

    # Normalize
    mean_emb = normalize(mean_emb, norm="l2")
    last_emb = normalize(last_emb, norm="l2")

    # Centroids
    mean_centroids = compute_centroids(mean_emb, df, emotion_order)
    last_centroids = compute_centroids(last_emb, df, emotion_order)

    mean_sim = pd.DataFrame(
        cosine_similarity(mean_centroids),
        index=emotion_order,
        columns=emotion_order,
    ).round(3)

    last_sim = pd.DataFrame(
        cosine_similarity(last_centroids),
        index=emotion_order,
        columns=emotion_order,
    ).round(3)

    append_text(log_file, dataframe_to_string(mean_sim, "Mean centroid cosine similarity"))
    append_text(log_file, dataframe_to_string(last_sim, "Last-token centroid cosine similarity"))
    
    
    # Linear probes
    mean_probe, mean_metrics, mean_text = train_probe(mean_emb, labels, "Mean pooled")
    last_probe, last_metrics, last_text = train_probe(last_emb, labels, "Last token")
    
    class_order = list(mean_probe.classes_)

    # English target space
    english_balanced_df = balance_eval_df(
        english_eval_df,
        class_order,
        random_state=RANDOM_STATE,
    )
    
    english_mean_emb, english_last_emb, english_labels = encode_eval_texts_decoder(
        english_balanced_df,
        tokenizer,
        model,
        device,
        batch_size=batch_size,
        max_length=MAX_LENGTH,
    )
    
    english_mean_centroids = compute_label_centroids(
        english_mean_emb,
        english_labels,
        class_order,
    )
    
    english_last_centroids = compute_label_centroids(
        english_last_emb,
        english_labels,
        class_order,
    )
    
    english_mean_probe, _, english_mean_probe_text = train_probe(
        english_mean_emb,
        english_labels,
        "English target mean pooled",
    )
    
    english_last_probe, _, english_last_probe_text = train_probe(
        english_last_emb,
        english_labels,
        "English target last token",
    )
    
    append_text(log_file, "\n--- English target probes ---\n")
    append_text(log_file, english_mean_probe_text)
    append_text(log_file, english_last_probe_text)

    append_text(log_file, mean_text)
    append_text(log_file, last_text)

    # External evaluation: English
    english_eval_text, english_eval_summary = run_calibrated_evaluation(
        eval_df=english_eval_df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=MAX_LENGTH,
        mean_probe=mean_probe,
        last_probe=last_probe,
        class_order=class_order,
        dataset_name="english_eval",
    )
    append_text(log_file, english_eval_text)

    # External evaluation: Arabic
    arabic_eval_text, arabic_eval_summary = run_calibrated_evaluation(
        eval_df=arabic_eval_df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=MAX_LENGTH,
        mean_probe=mean_probe,
        last_probe=last_probe,
        class_order=class_order,
        dataset_name="arabic_eval",
    )
    append_text(log_file, arabic_eval_text)
    
    # Chinese projected into Arabic space
    chinese_to_arabic_text, chinese_to_arabic_summary = run_crosslingual_projection_eval(
        eval_df=chinese_eval_df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=MAX_LENGTH,
        target_mean_centroids=mean_centroids,
        target_last_centroids=last_centroids,
        target_mean_probe=mean_probe,
        target_last_probe=last_probe,
        class_order=class_order,
        dataset_name="chinese_eval",
        target_name="arabic",
    )
    
    append_text(log_file, chinese_to_arabic_text)

    # Chinese projected into English space
    chinese_to_english_text, chinese_to_english_summary = run_crosslingual_projection_eval(
        eval_df=chinese_eval_df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=MAX_LENGTH,
        target_mean_centroids=english_mean_centroids,
        target_last_centroids=english_last_centroids,
        target_mean_probe=english_mean_probe,
        target_last_probe=english_last_probe,
        class_order=class_order,
        dataset_name="chinese_eval",
        target_name="english",
    )
    
    append_text(log_file, chinese_to_english_text)
    

    # Estonian projected into Arabic space
    estonian_to_arabic_text, estonian_to_arabic_summary = run_crosslingual_projection_eval(
        eval_df=estonian_eval_df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=MAX_LENGTH,
        target_mean_centroids=mean_centroids,
        target_last_centroids=last_centroids,
        target_mean_probe=mean_probe,
        target_last_probe=last_probe,
        class_order=class_order,
        dataset_name="estonian_eval",
        target_name="arabic",
    )
    
    append_text(log_file, estonian_to_arabic_text)
    

    # Estonian projected into English space
    estonian_to_english_text, estonian_to_english_summary = run_crosslingual_projection_eval(
        eval_df=estonian_eval_df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=MAX_LENGTH,
        target_mean_centroids=english_mean_centroids,
        target_last_centroids=english_last_centroids,
        target_mean_probe=english_mean_probe,
        target_last_probe=english_last_probe,
        class_order=class_order,
        dataset_name="estonian_eval",
        target_name="english",
    )
    
    append_text(log_file, estonian_to_english_text)
    

    # Russian projected into Arabic space
    russian_to_arabic_text, russian_to_arabic_summary = run_crosslingual_projection_eval(
        eval_df=russian_eval_df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=MAX_LENGTH,
        target_mean_centroids=mean_centroids,
        target_last_centroids=last_centroids,
        target_mean_probe=mean_probe,
        target_last_probe=last_probe,
        class_order=class_order,
        dataset_name="russian_eval",
        target_name="arabic",
    )
    
    append_text(log_file, russian_to_arabic_text)
    

    # Russian projected into English space
    russian_to_english_text, russian_to_english_summary = run_crosslingual_projection_eval(
        eval_df=russian_eval_df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=MAX_LENGTH,
        target_mean_centroids=english_mean_centroids,
        target_last_centroids=english_last_centroids,
        target_mean_probe=english_mean_probe,
        target_last_probe=english_last_probe,
        class_order=class_order,
        dataset_name="russian_eval",
        target_name="english",
    )
    
    append_text(log_file, russian_to_english_text)

    
    # German projected into Arabic space
    german_to_arabic_text, german_to_arabic_summary = run_crosslingual_projection_eval(
        eval_df=german_eval_df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=MAX_LENGTH,
        target_mean_centroids=mean_centroids,
        target_last_centroids=last_centroids,
        target_mean_probe=mean_probe,
        target_last_probe=last_probe,
        class_order=class_order,
        dataset_name="german_eval",
        target_name="arabic",
    )
    
    append_text(log_file, german_to_arabic_text)
    

    # German projected into English space
    german_to_english_text, german_to_english_summary = run_crosslingual_projection_eval(
        eval_df=german_eval_df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=MAX_LENGTH,
        target_mean_centroids=english_mean_centroids,
        target_last_centroids=english_last_centroids,
        target_mean_probe=english_mean_probe,
        target_last_probe=english_last_probe,
        class_order=class_order,
        dataset_name="german_eval",
        target_name="english",
    )
    
    append_text(log_file, german_to_english_text)


    # Swahili projected into Arabic space
    swahili_to_arabic_text, swahili_to_arabic_summary = run_crosslingual_projection_eval(
        eval_df=swahili_eval_df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=MAX_LENGTH,
        target_mean_centroids=mean_centroids,
        target_last_centroids=last_centroids,
        target_mean_probe=mean_probe,
        target_last_probe=last_probe,
        class_order=class_order,
        dataset_name="swahili_eval",
        target_name="arabic",
    )
    
    append_text(log_file, swahili_to_arabic_text)
    

    # Swahili projected into English space
    swahili_to_english_text, swahili_to_english_summary = run_crosslingual_projection_eval(
        eval_df=swahili_eval_df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=MAX_LENGTH,
        target_mean_centroids=english_mean_centroids,
        target_last_centroids=english_last_centroids,
        target_mean_probe=english_mean_probe,
        target_last_probe=english_last_probe,
        class_order=class_order,
        dataset_name="swahili_eval",
        target_name="english",
    )
    
    append_text(log_file, swahili_to_english_text)
    

    # Ukrainian projected into Arabic space
    ukrainian_to_arabic_text, ukrainian_to_arabic_summary = run_crosslingual_projection_eval(
        eval_df=ukrainian_eval_df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=MAX_LENGTH,
        target_mean_centroids=mean_centroids,
        target_last_centroids=last_centroids,
        target_mean_probe=mean_probe,
        target_last_probe=last_probe,
        class_order=class_order,
        dataset_name="ukrainian_eval",
        target_name="arabic",
    )
    
    append_text(log_file, ukrainian_to_arabic_text)
    

    # Ukrainian projected into English space
    ukrainian_to_english_text, ukrainian_to_english_summary = run_crosslingual_projection_eval(
        eval_df=ukrainian_eval_df,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=MAX_LENGTH,
        target_mean_centroids=english_mean_centroids,
        target_last_centroids=english_last_centroids,
        target_mean_probe=english_mean_probe,
        target_last_probe=english_last_probe,
        class_order=class_order,
        dataset_name="ukrainian_eval",
        target_name="english",
    )
    
    append_text(log_file, ukrainian_to_english_text)
    

    # Summary
    summary_row = {
        "model": model_id,
        "mean_macro_f1_train": mean_metrics["macro_f1"],
        "last_macro_f1_train": last_metrics["macro_f1"],
        "mean_bal_acc_train": mean_metrics["balanced_accuracy"],
        "last_bal_acc_train": last_metrics["balanced_accuracy"],
    }

    summary_row.update(english_eval_summary)
    summary_row.update(arabic_eval_summary)
    summary_row.update(chinese_to_arabic_summary)
    summary_row.update(chinese_to_english_summary)
    
    summary_row.update(estonian_to_arabic_summary)
    summary_row.update(estonian_to_english_summary)
    
    summary_row.update(german_to_arabic_summary)
    summary_row.update(german_to_english_summary)
    
    summary_row.update(russian_to_arabic_summary)
    summary_row.update(russian_to_english_summary)
    
    summary_row.update(swahili_to_arabic_summary)
    summary_row.update(swahili_to_english_summary)
    
    summary_row.update(ukrainian_to_arabic_summary)
    summary_row.update(ukrainian_to_english_summary)

    summary_rows.append(summary_row)


    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()


# Save global summary

summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(BASE_OUTPUT_DIR, "summary.csv")
summary_df.to_csv(summary_path, index=False)

print("\nSaved summary to:", summary_path)
print(summary_df)