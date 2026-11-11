"""English-pivot translation baseline for multilingual emotion classification.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Model loading !!!

MODEL_NAME = "google-bert/bert-base-multilingual-cased"
# MODEL_NAME = "FacebookAI/xlm-roberta-base"
# MODEL_NAME = "microsoft/mdeberta-v3-base"
DATA_FILE = Path("combined_english_cleaned_surprise_and_fear_balanced.csv")
TRANSLATED_DIR = Path("translated_nllb")
XED_ENGLISH_FILE = Path("xed_english.csv")
MAX_LENGTH = 128
BATCH_SIZE = 32
RANDOM_STATE = 42
TARGET_EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

TRANSLATED_FILES = {
    "Chinese": [
        "xed_chinese_to_english.csv",
        "xed_tw_to_english.csv",
        "brighter_chinese_to_english.csv",
    ],
    "Estonian": ["xed_estonian_to_english.csv"],
    "Russian": ["xed_russian_to_english.csv", "brighter_russian_to_english.csv"],
    "Swahili": ["brighter_swahili_to_english.csv"],
    "Ukrainian": ["brighter_ukrainian_to_english.csv"],
    "German": ["xed_german_to_english.csv", "brighter_german_to_english.csv"],
}

EXAMPLE_TEXTS = [
    "I am so happy and excited about this!",
    "This is disgusting and makes me sick.",
    "I am terrified something bad will happen.",
    "I miss you so much and feel empty.",
    "What a wonderful surprise!",
    "I am furious and cannot stand this anymore.",
]


def encode_texts(texts, tokenizer, model, device):
    mean_embeddings = []
    cls_embeddings = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encoding"):
            batch = texts[start:start + BATCH_SIZE]
            encoded = tokenizer(
                batch, padding=True, truncation=True,
                max_length=MAX_LENGTH, return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            hidden = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            pooled_mean = torch.sum(hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_embeddings.append(pooled_mean.cpu().numpy())
            cls_embeddings.append(hidden[:, 0, :].cpu().numpy())
    return np.vstack(mean_embeddings), np.vstack(cls_embeddings)


def encode_normalized(texts, tokenizer, model, device):
    mean, cls = encode_texts(texts, tokenizer, model, device)
    return normalize(mean, norm="l2"), normalize(cls, norm="l2")


def make_classifier():
    return LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )


def train_probe(embeddings, labels, name):
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2,
        random_state=RANDOM_STATE, stratify=labels,
    )
    classifier = make_classifier()
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    print(f"\n{name} probe")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("Balanced accuracy:", round(balanced_accuracy_score(y_test, predictions), 4))
    print("Macro F1:", round(f1_score(y_test, predictions, average="macro"), 4))
    print("Weighted F1:", round(f1_score(y_test, predictions, average="weighted"), 4))
    print(classification_report(y_test, predictions, digits=3))
    return classifier


def predict_examples(tokenizer, model, device, probe, pooling):
    # The notebook normalized the embeddings before predicting examples.
    mean, cls = encode_normalized(EXAMPLE_TEXTS, tokenizer, model, device)
    embeddings = mean if pooling == "Mean" else cls
    predictions = probe.predict(embeddings)
    probabilities = probe.predict_proba(embeddings)
    print(f"\n{pooling} example predictions")
    for sentence, prediction, probability in zip(EXAMPLE_TEXTS, predictions, probabilities):
        indices = np.argsort(probability)[::-1][:3]
        top_three = [(probe.classes_[i], round(float(probability[i]), 3)) for i in indices]
        print("Text:", sentence)
        print("Predicted:", prediction)
        print("Top three:", top_three)


def balance_data(dataframe):
    # Groupwise samples and final shuffle exactly match the notebook.
    target_count = dataframe["emotion"].value_counts().min()
    groups = [
        group.sample(n=target_count, random_state=RANDOM_STATE)
        for _, group in dataframe.groupby("emotion")
    ]
    return (
        pd.concat(groups, ignore_index=True)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )


def load_csv_data(paths):
    dataframe = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    dataframe["text"] = dataframe["text"].astype(str)
    dataframe["emotion"] = dataframe["emotion"].astype(str)
    return dataframe


def load_brighter(config):
    parts = [
        load_dataset("brighter-dataset/BRIGHTER-emotion-categories", config, split=split).to_pandas()
        for split in ("train", "dev", "test")
    ]
    frame = pd.concat(parts, ignore_index=True)
    frame["text"] = frame["text"].astype(str)
    for emotion in TARGET_EMOTIONS:
        if emotion not in frame.columns:
            frame[emotion] = 0
        frame[emotion] = frame[emotion].fillna(0).astype(int)
    frame = frame[frame[TARGET_EMOTIONS].sum(axis=1) == 1].copy()
    frame["emotion"] = frame[TARGET_EMOTIONS].idxmax(axis=1)
    return frame[["text", "emotion"]]


def split_embeddings(mean, cls, labels):
    mean_calib, mean_test, y_calib, y_test = train_test_split(
        mean, labels, test_size=0.5,
        random_state=RANDOM_STATE, stratify=labels,
    )
    cls_calib, cls_test, _, _ = train_test_split(
        cls, labels, test_size=0.5,
        random_state=RANDOM_STATE, stratify=labels,
    )
    return (mean_calib, mean_test), (cls_calib, cls_test), y_calib, y_test


def report_metrics(y_true, predictions, probabilities, class_order, name, show_top_three=False):
    print(f"\n{name}")
    print("Accuracy:", round(accuracy_score(y_true, predictions), 4))
    print("Balanced accuracy:", round(balanced_accuracy_score(y_true, predictions), 4))
    print("Macro F1:", round(f1_score(y_true, predictions, average="macro"), 4))
    print("Weighted F1:", round(f1_score(y_true, predictions, average="weighted"), 4))
    print("Log loss:", round(log_loss(y_true, probabilities, labels=class_order), 4))

    if show_top_three:
        top_indices = np.argsort(probabilities, axis=1)[:, -3:]
        gold_indices = np.array([class_order.index(label) for label in y_true])
        top_three_accuracy = np.mean([
            gold_indices[i] in top_indices[i] for i in range(len(gold_indices))
        ])
        print("Top-3 accuracy:", round(top_three_accuracy, 4))

    print("Classification report:")
    print(classification_report(y_true, predictions, digits=3))
    matrix = pd.DataFrame(
        confusion_matrix(y_true, predictions, labels=class_order),
        index=class_order, columns=class_order,
    )
    print("Confusion matrix:")
    print(matrix.to_string())
    return matrix


def evaluate_direct(X_test, y_test, probe, class_order, name, show_details=False):
    predictions = probe.predict(X_test)
    probabilities = probe.predict_proba(X_test)
    matrix = report_metrics(
        y_test, predictions, probabilities, class_order, name,
        show_top_three=show_details,
    )
    if show_details:
        confidences = probabilities[np.arange(len(probabilities)), np.argmax(probabilities, axis=1)]
        confidence_frame = pd.DataFrame({"gold": y_test, "confidence": confidences})
        print("Per-class confidence stats:")
        print(confidence_frame.groupby("gold")["confidence"].agg(["mean", "std", "count"]).round(4).to_string())
    return {"preds": predictions, "probs": probabilities, "confusion_matrix": matrix}


def evaluate_6d(X_calib, X_test, y_calib, y_test, probe, class_order, name):
    # No Procrustes rotation is performed in the source notebook's language runs.
    weights = probe.coef_.copy()
    Z_calib = X_calib @ weights.T
    Z_test = X_test @ weights.T
    classifier = make_classifier()
    classifier.fit(Z_calib, y_calib)
    predictions = classifier.predict(Z_test)
    probabilities = classifier.predict_proba(Z_test)
    print("6D train shape:", Z_calib.shape)
    print("6D test shape:", Z_test.shape)
    matrix = report_metrics(y_test, predictions, probabilities, class_order, name)
    return {"model": classifier, "preds": predictions, "probs": probabilities, "confusion_matrix": matrix}


def evaluate_dataset(name, dataframe, tokenizer, model, device, probes, details=False):
    print(f"\n{name}")
    print("Original rows:", len(dataframe))
    print("Original class distribution:")
    print(dataframe["emotion"].value_counts())
    balanced = balance_data(dataframe)
    print("Balanced rows:", len(balanced))
    print("Balanced class distribution:")
    print(balanced["emotion"].value_counts())
    mean, cls = encode_normalized(balanced["text"].tolist(), tokenizer, model, device)
    print("Mean embeddings shape:", mean.shape)
    print("CLS embeddings shape:", cls.shape)
    (mean_calib, mean_test), (cls_calib, cls_test), y_calib, y_test = split_embeddings(
        mean, cls, balanced["emotion"].tolist(),
    )
    print("Calibration set size:", len(y_calib))
    print("Test set size:", len(y_test))
    class_order = list(probes["Mean"].classes_)
    for pooling, calibration_embeddings, test_embeddings in (
        ("Mean", mean_calib, mean_test),
        ("CLS", cls_calib, cls_test),
    ):
        probe = probes[pooling]
        # The English labeled-data block in the notebook evaluated only
        # the mean-pooled direct probe, while translations evaluated both.
        if not details or pooling == "Mean":
            evaluate_direct(
                test_embeddings, y_test, probe, class_order,
                f"{name} {pooling} direct English probe", show_details=details,
            )
        evaluate_6d(
            calibration_embeddings, test_embeddings, y_calib, y_test,
            probe, class_order, f"{name} {pooling} 6D classifier",
        )


def main():
    # The original notebook selected MPS whenever CUDA was unavailable.
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print("Device:", device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    source = load_csv_data([DATA_FILE])
    source_texts = source["text"].tolist()
    source_labels = source["emotion"].tolist()
    print("English probe rows:", len(source))
    print("Emotions:", sorted(source["emotion"].unique()))
    mean, cls = encode_normalized(source_texts, tokenizer, model, device)
    print("Mean embeddings shape:", mean.shape)
    print("CLS embeddings shape:", cls.shape)
    probes = {
        "Mean": train_probe(mean, source_labels, "Mean-pooled"),
        "CLS": train_probe(cls, source_labels, "CLS"),
    }
    for pooling in ("Mean", "CLS"):
        predict_examples(tokenizer, model, device, probes[pooling], pooling)

    brighter = load_brighter("eng")
    xed = load_csv_data([XED_ENGLISH_FILE])
    xed = xed[xed["emotion"].isin(TARGET_EMOTIONS)]
    for label, frame in (("BRIGHTER English", brighter), ("XED English", xed)):
        print(f"\n{label} distribution:")
        print(frame["emotion"].value_counts())
    combined = pd.concat([brighter, xed], ignore_index=True)
    print("\nCombined English distribution before balancing:")
    print(combined["emotion"].value_counts())
    evaluate_dataset("English labeled-data evaluation", combined, tokenizer, model, device, probes, details=True)

    for language, filenames in TRANSLATED_FILES.items():
        translated = load_csv_data([TRANSLATED_DIR / filename for filename in filenames])
        evaluate_dataset(language, translated, tokenizer, model, device, probes)


if __name__ == "__main__":
    main()
