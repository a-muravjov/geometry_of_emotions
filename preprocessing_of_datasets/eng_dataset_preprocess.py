import os
import glob
import re
import pandas as pd
import emoji
from collections import Counter, defaultdict
import html

# -------------------------
# Config
# -------------------------
input_folder = "all_english_datasets_surprise_and_fear"
output_file = "combined_english_cleaned_surprise_and_fear.csv"

min_words = 3
max_words = 80

emoji_to_emotion = {
    # JOY
    "💐": "joy", "🌹": "joy", "💝": "joy", "💞": "joy", "🥂": "joy",
    "🌈": "joy", "🤗": "joy", "😀": "joy", "😃": "joy", "😄": "joy", "😁": "joy", "😆": "joy",
    "😂": "joy", "🤣": "joy", "😊": "joy", "😉": "joy", "😍": "joy", "🥰": "joy", "😘": "joy",
    "😋": "joy", "🤩": "joy", "☺️": "joy", "😚": "joy", "😙": "joy", "😛": "joy", "😜": "joy",
    "🤪": "joy", "😝": "joy", "😌": "joy", "❤️": "joy", "🩷": "joy", "🧡": "joy", "💜": "joy", "❤": "joy", "🙏": "joy",

    # SADNESS
    "💔": "sadness", "😢": "sadness", "😞": "sadness", "😓": "sadness",
    "🙁": "sadness", "😔": "sadness", "🤕": "sadness", "☹️": "sadness",

    # ANGER
    "🤬": "anger", "😡": "anger", "🥊": "anger", "👿": "anger",
    "😠": "anger", "🖕": "anger", "😤": "anger", "💢": "anger", "😾": "anger",

    # FEAR
    "✝️": "fear", "⚠️": "fear", "😥": "fear", "😨": "fear", "😰": "fear", "😱": "fear",
    "🙀": "fear", "😖": "fear", "😣": "fear", "😟": "fear",

    # SURPRISE (updated)
    "🎉": "surprise",
    "🎊": "surprise",
    "🥳": "surprise",
    "🙇": "surprise",
    "😦": "surprise",
    "😲": "surprise",
    "😯": "surprise",
    "😮": "surprise",
    "😳": "surprise",
    "🤯": "surprise",
    "😵": "surprise",

    # DISGUST
    "🤮": "disgust", "🤢": "disgust",
}

# -------------------------
# Regex helpers
# -------------------------
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
QUOTE_LINE_RE = re.compile(r"(^|\n)\s*>\s*")
ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
MULTISPACE_RE = re.compile(r"\s+")

USER_RE = re.compile(r"\bu/[A-Za-z0-9_-]+\b", flags=re.IGNORECASE)
SUBREDDIT_RE = re.compile(r"\br/[A-Za-z0-9_]+\b", flags=re.IGNORECASE)
DELETED_RE = re.compile(r"\[(deleted|removed)\]", flags=re.IGNORECASE)

# markdown-ish cleanup
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")      # [text](url) -> text
MD_BOLDITALIC_RE = re.compile(r"(\*\*\*|___)(.+?)\1")    # ***text*** / ___text___
MD_BOLD_RE = re.compile(r"(\*\*|__)(.+?)\1")             # **text** / __text__
MD_ITALIC_RE = re.compile(r"(\*|_)([^*_]+?)\1")          # *text* / _text_
BACKTICK_RE = re.compile(r"`([^`]+)`")                   # `text` -> text
TILDE_RE = re.compile(r"~~(.+?)~~")                      # ~~text~~ -> text

# optional bare markdown markers leftover
MARKER_RE = re.compile(r"[*_~`]+")

# -------------------------
# Helpers
# -------------------------
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ")
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text

def remove_urls(text):
    if not isinstance(text, str):
        return ""
    return URL_RE.sub(" ", text)

def clean_html_entities(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = ZERO_WIDTH_RE.sub("", text)
    return text

def remove_reddit_quote_markers(text):
    if not isinstance(text, str):
        return ""
    return QUOTE_LINE_RE.sub(" ", text)

def remove_reddit_entities(text):
    if not isinstance(text, str):
        return ""
    text = USER_RE.sub(" ", text)
    text = SUBREDDIT_RE.sub(" ", text)
    text = DELETED_RE.sub(" ", text)
    return text

def clean_markdown(text):
    if not isinstance(text, str):
        return ""
    text = MD_LINK_RE.sub(r"\1", text)
    text = MD_BOLDITALIC_RE.sub(r"\2", text)
    text = MD_BOLD_RE.sub(r"\2", text)
    text = MD_ITALIC_RE.sub(r"\2", text)
    text = BACKTICK_RE.sub(r"\1", text)
    text = TILDE_RE.sub(r"\1", text)
    text = MARKER_RE.sub("", text)
    return text

def remove_emojis_from_text(text):
    if not isinstance(text, str):
        return ""
    text = clean_html_entities(text)
    text = remove_urls(text)
    text = remove_reddit_quote_markers(text)
    text = remove_reddit_entities(text)
    text = clean_markdown(text)
    text = emoji.replace_emoji(text, replace="")
    text = normalize_text(text)
    return text

def word_count(text):
    return len(text.split())

def valid_length(text):
    wc = word_count(text)
    return min_words <= wc <= max_words

def extract_emojis_any(text):
    if not isinstance(text, str):
        return []
    return [e["emoji"] for e in emoji.emoji_list(text)]

def get_row_emojis(row):
    """
    Use emojis from the actual text for conflict detection.
    Fall back to the 'emojis' column only if the text contains none.
    """
    text_emojis = extract_emojis_any(row.get("text", ""))
    if text_emojis:
        return text_emojis

    emojis_col = row.get("emojis", "")
    if isinstance(emojis_col, str) and emojis_col.strip():
        found = extract_emojis_any(emojis_col)
        if found:
            return found
        return emojis_col.split()

    return []

def check_conflict_and_assign(emojis_found):
    mapped = [emoji_to_emotion[e] for e in emojis_found if e in emoji_to_emotion]
    if not mapped:
        return None, [], False
    unique_emotions = set(mapped)
    if len(unique_emotions) == 1:
        return mapped[0], [e for e in emojis_found if e in emoji_to_emotion], False
    return None, [e for e in emojis_found if e in emoji_to_emotion], True

# -------------------------
# Load all CSVs
# -------------------------
csv_files = sorted(glob.glob(os.path.join(input_folder, "*.csv")))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {input_folder}")

dfs = []
for file in csv_files:
    df_part = pd.read_csv(file)
    df_part["source_file"] = os.path.basename(file)
    dfs.append(df_part)

df = pd.concat(dfs, ignore_index=True)

print("Loaded files:", len(csv_files))
print("Original total rows:", len(df))
print("\nFiles:")
for f in csv_files:
    print("-", os.path.basename(f))

# -------------------------
# Basic cleanup
# -------------------------
if "text" not in df.columns:
    raise ValueError("Expected a 'text' column in the input files.")

df["text"] = df["text"].astype(str).apply(normalize_text)

if "emotion" in df.columns:
    df["emotion"] = df["emotion"].astype(str)

if "emojis" not in df.columns:
    df["emojis"] = ""

# -------------------------
# Re-check emoji consistency
# -------------------------
assigned_emotions = []
mapped_emojis_joined = []
conflict_flags = []
has_mapped_flags = []

for _, row in df.iterrows():
    emojis_found = get_row_emojis(row)
    assigned_emotion, mapped_emojis, has_conflict = check_conflict_and_assign(emojis_found)

    assigned_emotions.append(assigned_emotion)
    mapped_emojis_joined.append(" ".join(mapped_emojis))
    conflict_flags.append(has_conflict)
    has_mapped_flags.append(len(mapped_emojis) > 0)

df["assigned_emotion"] = assigned_emotions
df["mapped_emojis"] = mapped_emojis_joined
df["has_conflict"] = conflict_flags
df["has_mapped_emoji"] = has_mapped_flags

num_conflicting = int(df["has_conflict"].sum())
num_no_mapped = int((~df["has_mapped_emoji"]).sum())

print("\nRows with conflicting mapped emojis:", num_conflicting)
print("Rows with no mapped emojis:", num_no_mapped)

before_conflict_filter = len(df)
df = df[(df["assigned_emotion"].notna()) & (~df["has_conflict"])].copy()

print("Rows removed due to conflict / no mapped emotion:", before_conflict_filter - len(df))

# -------------------------
# Clean text
# -------------------------
df["text_clean"] = df["text"].apply(remove_emojis_from_text)

# -------------------------
# Length filtering
# -------------------------
before_length_filter = len(df)
df = df[df["text_clean"].str.len() > 0].copy()
df = df[df["text_clean"].apply(valid_length)].copy()

print("Rows removed due to short/long cleaned text:", before_length_filter - len(df))

# -------------------------
# Duplicate filtering
# -------------------------
before_dupes = len(df)
df = df.drop_duplicates(subset=["text_clean"]).copy()
print("Duplicates removed after cleaning:", before_dupes - len(df))

# -------------------------
# Finalize columns
# -------------------------
df["text"] = df["text_clean"]
df["emotion"] = df["assigned_emotion"]

final_df = df[["emotion", "text", "mapped_emojis", "source_file"]].copy()
final_df = final_df.rename(columns={"mapped_emojis": "emojis"})
final_df = final_df.reset_index(drop=True)

# -------------------------
# Final statistics
# -------------------------
print("\nFinal rows:", len(final_df))

print("\nFinal class distribution:")
class_counts = final_df["emotion"].value_counts()
print(class_counts)

print("\nClass percentages:")
print((class_counts / len(final_df) * 100).round(2))

print("\nPer-class average text length (words):")
final_df["word_count"] = final_df["text"].apply(word_count)
print(final_df.groupby("emotion")["word_count"].mean().round(2))

print("\nPer-class min/max text length (words):")
print(final_df.groupby("emotion")["word_count"].agg(["min", "max"]))

emoji_counter = Counter()
for val in final_df["emojis"]:
    for e in extract_emojis_any(val):
        emoji_counter[e] += 1

print("\nMapped emoji distribution:")
for e, count in emoji_counter.most_common():
    print(f"{e}: {count}")

# -------------------------
# Save
# -------------------------
final_df = final_df.drop(columns=["word_count"])
final_df.to_csv(output_file, index=False, encoding="utf-8")

print("\nSaved cleaned combined dataset to:", output_file)