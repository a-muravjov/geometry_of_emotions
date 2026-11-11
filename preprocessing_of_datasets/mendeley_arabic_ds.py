"""Extracts and labels Arabic texts from the Mendeley dataset using emoji-based emotion mappings."""
import emoji
import csv
from collections import defaultdict, Counter
import pandas as pd

# FROM https://data.mendeley.com/datasets/m88gg52wp7/1

dataset = pd.read_csv("mendeley_arabic_ds.csv", sep="\t")
print(dataset.columns)
print(dataset.head())

output_file = "mendeley_arabic_pure_mapped_comments.csv"

emoji_to_emotion = {
    # JOY
    "💐": "joy", "🌹": "joy", "💝": "joy", "💞": "joy",
    "🌈": "joy", "🤗": "joy", "😀": "joy", "😃": "joy", "😁": "joy", "😆": "joy",
    "😂": "joy", "🤣": "joy", "😊": "joy", "😉": "joy", "😍": "joy", "🥰": "joy", "😘": "joy",
    "😋": "joy", "🤩": "joy", "☺️": "joy", "😚": "joy", "😛": "joy", "😜": "joy", "🤪": "joy",
    "😝": "joy", "😌": "joy", "❤️": "joy", "🧡": "joy", "💜": "joy", "❤": "joy",
    "🙏": "joy",

    # SADNESS
    "💔": "sadness", "😢": "sadness", "😞": "sadness", "😓": "sadness",
    "🙁": "sadness", "😔": "sadness", "🤕": "sadness", "☹️": "sadness",

    # ANGER
    "🤬": "anger", "😡": "anger", "🥊": "anger", "👿": "anger",
    "😠": "anger", "🖕": "anger", "😤": "anger", "💢": "anger", "😾": "anger",

    # FEAR
    "⚠️": "fear", "😥": "fear", "😨": "fear", "😰": "fear", "😱": "fear",
    "🙀": "fear", "😖": "fear", "😣": "fear",

    # SURPRISE
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

emotion_counts = defaultdict(int)
mapped_emoji_counts = Counter()
all_emoji_counts = Counter()

conflicting_comments = 0
pure_comments = 0

def extract_emojis(text):
    if not isinstance(text, str):
        return []
    return [e["emoji"] for e in emoji.emoji_list(text)]

checked = 0
emoji_comments = 0

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["emotion", "text", "emojis"])

    for text in dataset["text"]:
        if not isinstance(text, str):
            continue

        emojis_in_text = extract_emojis(text)

        if emojis_in_text:
            emoji_comments += 1

        mapped_emotions_in_text = []
        mapped_emojis_in_text = []

        for e in emojis_in_text:
            # count ALL emojis
            all_emoji_counts[e] += 1

            # count mapped emojis
            if e in emoji_to_emotion:
                mapped_emoji_counts[e] += 1
                mapped_emotions_in_text.append(emoji_to_emotion[e])
                mapped_emojis_in_text.append(e)

        # only save/count text-level emotion if all mapped emojis agree
        unique_mapped_emotions = set(mapped_emotions_in_text)

        if len(unique_mapped_emotions) == 1:
            only_emotion = next(iter(unique_mapped_emotions))
            emotion_counts[only_emotion] += 1
            pure_comments += 1

            writer.writerow([
                only_emotion,
                text,
                " ".join(mapped_emojis_in_text)
            ])

        elif len(unique_mapped_emotions) > 1:
            conflicting_comments += 1

        checked += 1

        if checked % 100000 == 0:
            print("Processed:", checked)

print("\nSaved pure mapped comments to:", output_file)

print("\nChecked:", checked)
print("Comments with emojis:", emoji_comments)
print("Pure mapped comments:", pure_comments)
print("Conflicting mapped comments:", conflicting_comments)

print("\nEmotion distribution:")
for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{emotion}: {count}")

print("\nMapped emoji distribution:")
for e, count in mapped_emoji_counts.most_common():
    print(f"{e}: {count}")

print("\nAll emoji distribution:")
for e, count in all_emoji_counts.most_common(30):
    print(f"{e}: {count}")