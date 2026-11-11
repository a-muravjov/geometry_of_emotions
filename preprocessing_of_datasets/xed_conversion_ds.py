"""Converts the XED dataset into a single-label emotion classification format."""
import pandas as pd

# Emotion mapping
label_map = {
    "1": "anger",
    "3": "disgust",
    "4": "fear",
    "5": "joy",
    "6": "sadness",
    "7": "surprise"
}

df = pd.read_csv("tw-projections.tsv", sep="\t", header=None, names=["text", "labels"])

# ensure string
df["text"] = df["text"].astype(str)
df["labels"] = df["labels"].astype(str)

# Keep only single-label rows
def extract_single_label(label_str):
    labels = [l.strip() for l in label_str.split(",")]
    if len(labels) == 1:
        return labels[0]
    return None

df["single_label"] = df["labels"].apply(extract_single_label)

df = df[df["single_label"].notna()].copy()

# Map to emotion names
df["emotion"] = df["single_label"].map(label_map)

# drop rows with unknown labels
df = df[df["emotion"].notna()]

# Final format
df_out = df[["emotion", "text"]].copy()
df_out["emojis"] = ""

# reorder columns
df_out = df_out[["emotion", "text", "emojis"]]

df_out.to_csv("xed_tw.csv", index=False)

print("Saved:", len(df_out), "rows")
print(df_out["emotion"].value_counts())