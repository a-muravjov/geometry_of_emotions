import os, gc, warnings
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report

# =========================
# CONFIG
# =========================

ROOT = Path("thesis")
TRANSLATED_DIR = ROOT / "translated_nllb"

ENGLISH_FILES = [
    ROOT / "xed_english.csv",
    ROOT / "brighter_english.csv",
]

ARABIC_FILES = [
    ROOT / "xed_arabic.csv",
    ROOT / "brighter_arabic.csv",
    ROOT / "combined_arabic_balanced_1200_new_bestest_no_@.csv",
]

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

TARGET_EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

MAX_LENGTH = 128
RANDOM_STATE = 42
OUTPUT_DIR = Path("outputs_translation_baseline_decoder")
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", device)


# =========================
# HELPERS
# =========================

def safe_model_id(model_name):
    return model_name.split("/")[-1].replace(" ", "_")


def load_and_balance(files, target_emotions=TARGET_EMOTIONS):
    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    df["text"] = df["text"].astype(str)
    df["emotion"] = df["emotion"].astype(str)
    df = df[df["emotion"].isin(target_emotions)].copy()

    target_n = df["emotion"].value_counts().min()

    parts = []
    for emotion, group in df.groupby("emotion"):
        parts.append(group.sample(n=target_n, random_state=RANDOM_STATE))

    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )


def mean_pooling(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    summed = torch.sum(hidden_states * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def last_token_pooling(hidden_states, attention_mask):
    last_token_indices = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_indices, last_token_indices, :]


def encode_texts(texts, tokenizer, model, batch_size):
    mean_embs = []
    last_embs = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), leave=False):
            batch = texts[i:i + batch_size]

            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            outputs = model(
                **encoded,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden = outputs.hidden_states[-1]

            mean_embs.append(mean_pooling(hidden, encoded["attention_mask"]).float().cpu().numpy())
            last_embs.append(last_token_pooling(hidden, encoded["attention_mask"]).float().cpu().numpy())

    mean_embs = normalize(np.vstack(mean_embs), norm="l2")
    last_embs = normalize(np.vstack(last_embs), norm="l2")

    return mean_embs, last_embs


def train_probe(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    clf = LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    clf.fit(X_train, y_train)
    return clf


def evaluate_direct_probe(X, y, probe):
    preds = probe.predict(X)

    return {
        "accuracy": round(accuracy_score(y, preds), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y, preds), 4),
        "macro_f1": round(f1_score(y, preds, average="macro"), 4),
        "weighted_f1": round(f1_score(y, preds, average="weighted"), 4),
        "report": classification_report(y, preds, digits=3),
    }


def translated_files_for_target(target):
    return sorted(TRANSLATED_DIR.glob(f"*to_{target}.csv"))


# =========================
# LOAD TARGET TRAINING DATA
# =========================

english_df = load_and_balance(ENGLISH_FILES)
arabic_df = load_and_balance(ARABIC_FILES)

print("English probe rows:", len(english_df))
print(english_df["emotion"].value_counts())

print("Arabic probe rows:", len(arabic_df))
print(arabic_df["emotion"].value_counts())


# =========================
# MAIN LOOP
# =========================

summary_rows = []

for model_name in MODELS:
    print(f"\n\n===== MODEL: {model_name} =====")

    model_id = safe_model_id(model_name)
    model_output_dir = OUTPUT_DIR / model_id
    model_output_dir.mkdir(exist_ok=True)

    batch_size = MODEL_BATCH_SIZES.get(model_name, 4)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
    )

    model.to(device)
    model.eval()

    # -------------------------
    # Train English probes
    # -------------------------
    english_texts = english_df["text"].tolist()
    english_labels = english_df["emotion"].tolist()

    english_mean_X, english_last_X = encode_texts(
        english_texts,
        tokenizer,
        model,
        batch_size,
    )

    english_mean_probe = train_probe(english_mean_X, english_labels)
    english_last_probe = train_probe(english_last_X, english_labels)

    # -------------------------
    # Train Arabic probes
    # -------------------------
    arabic_texts = arabic_df["text"].tolist()
    arabic_labels = arabic_df["emotion"].tolist()

    arabic_mean_X, arabic_last_X = encode_texts(
        arabic_texts,
        tokenizer,
        model,
        batch_size,
    )

    arabic_mean_probe = train_probe(arabic_mean_X, arabic_labels)
    arabic_last_probe = train_probe(arabic_last_X, arabic_labels)

    summary_row = {"model": model_id}

    log_lines = [f"MODEL: {model_name}\n"]

    # -------------------------
    # Evaluate translations into English
    # -------------------------
    for file_path in translated_files_for_target("english"):
        df_eval = load_and_balance([file_path])
        eval_texts = df_eval["text"].tolist()
        eval_labels = df_eval["emotion"].tolist()

        mean_X, last_X = encode_texts(
            eval_texts,
            tokenizer,
            model,
            batch_size,
        )

        dataset_name = file_path.stem.replace("_to_english", "")

        mean_results = evaluate_direct_probe(mean_X, eval_labels, english_mean_probe)
        last_results = evaluate_direct_probe(last_X, eval_labels, english_last_probe)

        summary_row[f"{dataset_name}_to_english_mean_macro_f1"] = mean_results["macro_f1"]
        summary_row[f"{dataset_name}_to_english_last_macro_f1"] = last_results["macro_f1"]

        log_lines.append(f"\n===== {dataset_name} → English =====")
        log_lines.append(f"Rows after balancing: {len(df_eval)}")
        log_lines.append(f"Mean macro F1: {mean_results['macro_f1']}")
        log_lines.append(f"Last macro F1: {last_results['macro_f1']}")

    # -------------------------
    # Evaluate translations into Arabic
    # -------------------------
    for file_path in translated_files_for_target("arabic"):
        df_eval = load_and_balance([file_path])
        eval_texts = df_eval["text"].tolist()
        eval_labels = df_eval["emotion"].tolist()

        mean_X, last_X = encode_texts(
            eval_texts,
            tokenizer,
            model,
            batch_size,
        )

        dataset_name = file_path.stem.replace("_to_arabic", "")

        mean_results = evaluate_direct_probe(mean_X, eval_labels, arabic_mean_probe)
        last_results = evaluate_direct_probe(last_X, eval_labels, arabic_last_probe)

        summary_row[f"{dataset_name}_to_arabic_mean_macro_f1"] = mean_results["macro_f1"]
        summary_row[f"{dataset_name}_to_arabic_last_macro_f1"] = last_results["macro_f1"]

        log_lines.append(f"\n===== {dataset_name} → Arabic =====")
        log_lines.append(f"Rows after balancing: {len(df_eval)}")
        log_lines.append(f"Mean macro F1: {mean_results['macro_f1']}")
        log_lines.append(f"Last macro F1: {last_results['macro_f1']}")

    with open(model_output_dir / "translation_baseline_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    summary_rows.append(summary_row)

    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()


# =========================
# SAVE SUMMARY
# =========================

summary_df = pd.DataFrame(summary_rows)
summary_path = OUTPUT_DIR / "translation_baseline_summary.csv"
summary_df.to_csv(summary_path, index=False)

print("\nSaved summary to:", summary_path)
print(summary_df)