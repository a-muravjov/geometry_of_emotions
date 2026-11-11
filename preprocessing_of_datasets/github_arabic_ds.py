"""Processes Arabic tweets from multiple files and assigns emotion labels based on emojis."""
import os
import csv
import emoji
import re
from collections import defaultdict, Counter

# https://github.com/motazsaad/arabic-sentiment-analysis

folder = "github_arabic_ds"
output_file = "github_arabic_pure_mapped_comments.csv"

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

seen = set()

total_rows = 0
unique_tweets = 0
duplicates = 0
emoji_tweets = 0
pure_tweets = 0
conflicting_tweets = 0


def normalize(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_emojis(text):
    if not isinstance(text, str):
        return []
    return [e["emoji"] for e in emoji.emoji_list(text)]


with open(output_file, "w", newline="", encoding="utf-8") as out_f:
    writer = csv.writer(out_f)
    writer.writerow(["emotion", "text", "emojis"])

    for file in os.listdir(folder):

        if not file.endswith(".tsv"):
            continue

        path = os.path.join(folder, file)
        print("Processing:", file)

        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")

                if len(parts) != 2:
                    continue

                label, text = parts

                total_rows += 1
                text = normalize(text)

                if not text:
                    continue

                # duplicate filtering
                if text in seen:
                    duplicates += 1
                    continue

                seen.add(text)
                unique_tweets += 1

                emojis = extract_emojis(text)

                if emojis:
                    emoji_tweets += 1

                mapped_emotions = []
                mapped_emojis = []

                for e in emojis:
                    all_emoji_counts[e] += 1

                    if e in emoji_to_emotion:
                        mapped_emoji_counts[e] += 1
                        mapped_emotions.append(emoji_to_emotion[e])
                        mapped_emojis.append(e)

                unique_emotions = set(mapped_emotions)

                # keep/save tweet only if all mapped emojis agree
                if len(unique_emotions) == 1:
                    emotion = next(iter(unique_emotions))
                    emotion_counts[emotion] += 1
                    pure_tweets += 1

                    writer.writerow([
                        emotion,
                        text,
                        " ".join(mapped_emojis)
                    ])

                elif len(unique_emotions) > 1:
                    conflicting_tweets += 1


print("\nSaved pure mapped tweets to:", output_file)

print("\nDataset statistics")
print("Total rows:", total_rows)
print("Unique tweets:", unique_tweets)
print("Duplicates removed:", duplicates)

print("\nTweets containing emojis:", emoji_tweets)
print("Pure mapped tweets:", pure_tweets)
print("Conflicting mapped tweets:", conflicting_tweets)

print("\nEmotion distribution (pure only):")
for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{emotion}: {count}")

print("\nMapped emoji distribution:")
for e, count in mapped_emoji_counts.most_common():
    print(f"{e}: {count}")

print("\nAll emoji distribution:")
for e, count in all_emoji_counts.most_common(30):
    print(f"{e}: {count}")