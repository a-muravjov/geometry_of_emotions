import json
import re
import emoji
import os
import csv
from collections import defaultdict, Counter

folder = "tead_dataset"
output_file = "tead_pure_mapped_comments.csv"

# https://drive.google.com/drive/folders/1pfqK21nEKL8xYt3L4e09tMIQSnI_5G-7
# https://github.com/HSMAabdellaoui/TEAD?tab=readme-ov-file

emoji_to_emotion = {
    # JOY
    "💐": "joy", "🌹": "joy", "💝": "joy", "💞": "joy", "🥂": "joy",
    "🌈": "joy", "🤗": "joy", "😀": "joy", "😃": "joy", "😄": "joy", "😁": "joy", "😆": "joy",
    "😂": "joy", "🤣": "joy", "😊": "joy", "😉": "joy", "😍": "joy", "🥰": "joy", "😘": "joy",
    "😋": "joy", "🤩": "joy", "☺️": "joy", "😚": "joy", "😙": "joy", "😛": "joy", "😜": "joy",
    "🤪": "joy", "😝": "joy", "😌": "joy", "❤️": "joy", "🩷": "joy", "🧡": "joy", "💜": "joy", "❤": "joy",

    # SADNESS
    "💔": "sadness", "😢": "sadness", "😞": "sadness", "😓": "sadness",
    "🙁": "sadness", "😔": "sadness", "🤕": "sadness", "☹️": "sadness",

    # ANGER
    "🤬": "anger", "😡": "anger", "🥊": "anger", "👿": "anger",
    "😠": "anger", "🖕": "anger", "😤": "anger", "💢": "anger", "😾": "anger",

    # FEAR
    "✝️": "fear", "⚠️": "fear", "😥": "fear", "😨": "fear", "😰": "fear", "😱": "fear",
    "🙀": "fear", "😖": "fear", "😣": "fear",

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

    # DISGUST
    "🤮": "disgust", "🤢": "disgust",
}

emotion_counts = defaultdict(int)
mapped_emoji_counts = Counter()
all_emoji_counts = Counter()

url_re = re.compile(r"https?://\S+|www\.\S+")

seen = set()

total_rows = 0
unique_tweets = 0
duplicates = 0
emoji_texts = 0
pure_texts = 0
conflicting_texts = 0


def normalize(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def remove_urls(text):
    if not isinstance(text, str):
        return ""
    return url_re.sub("", text).strip()


def extract_emojis(text):
    if not isinstance(text, str):
        return []
    return [e["emoji"] for e in emoji.emoji_list(text)]


with open(output_file, "w", newline="", encoding="utf-8") as out_f:
    writer = csv.writer(out_f)
    writer.writerow(["emotion", "text", "emojis"])

    # loop over JSON files
    for file in os.listdir(folder):

        if not file.endswith(".json"):
            continue

        path = os.path.join(folder, file)
        print("Processing:", file)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for obj in data:
            total_rows += 1

            text = obj.get("text", "")
            text = remove_urls(text)
            text = normalize(text)

            if not text:
                continue

            # duplicate detection
            if text in seen:
                duplicates += 1
                continue

            seen.add(text)
            unique_tweets += 1

            emojis = extract_emojis(text)

            if emojis:
                emoji_texts += 1

            mapped_emotions = []
            mapped_emojis = []

            for e in emojis:
                all_emoji_counts[e] += 1

                if e in emoji_to_emotion:
                    mapped_emoji_counts[e] += 1
                    mapped_emotions.append(emoji_to_emotion[e])
                    mapped_emojis.append(e)

            unique_emotions = set(mapped_emotions)

            # save/count only if all mapped emojis agree
            if len(unique_emotions) == 1:
                emotion = next(iter(unique_emotions))
                emotion_counts[emotion] += 1
                pure_texts += 1

                writer.writerow([
                    emotion,
                    text.replace("\n", " "),
                    " ".join(mapped_emojis)
                ])

            elif len(unique_emotions) > 1:
                conflicting_texts += 1


print("\nSaved to:", output_file)

print("\nDataset statistics")
print("Total rows:", total_rows)
print("Unique tweets:", unique_tweets)
print("Duplicates removed:", duplicates)
print("Texts with emojis:", emoji_texts)
print("Pure mapped texts:", pure_texts)
print("Conflicting mapped texts:", conflicting_texts)

print("\nEmotion distribution (pure only):")
for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{emotion}: {count}")

print("\nMapped emoji distribution:")
for e, count in mapped_emoji_counts.most_common():
    print(f"{e}: {count}")

print("\nAll emoji distribution:")
for e, count in all_emoji_counts.most_common(30):
    print(f"{e}: {count}")