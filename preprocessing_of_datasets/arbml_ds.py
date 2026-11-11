"""Extracts and labels arbml tweets using emoji-based emotion mappings."""
from datasets import load_dataset
import emoji
from collections import defaultdict, Counter
import csv

dataset = load_dataset("arbml/Arabic_Sentiment_Twitter_Corpus")

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


# Helper
def extract_emojis(text):
    if not isinstance(text, str):
        return []
    return [e["emoji"] for e in emoji.emoji_list(text)]

# Process function
def process_split(split_name):
    emotion_counts = defaultdict(int)
    emoji_counts = Counter()

    pure_comments = 0
    conflicting_comments = 0
    checked = 0
    emoji_comments = 0

    output_file = f"arabic_sentiment_twitter_{split_name}_pure_mapped_comments.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["emotion", "text", "emojis"])

        for example in dataset[split_name]:
            text = example["tweet"]

            emojis_in_text = extract_emojis(text)

            if emojis_in_text:
                emoji_comments += 1

            mapped_emotions = []
            mapped_emojis = []

            for e in emojis_in_text:
                if e in emoji_to_emotion:
                    emoji_counts[e] += 1
                    mapped_emotions.append(emoji_to_emotion[e])
                    mapped_emojis.append(e)

            unique_emotions = set(mapped_emotions)

            # only save if all mapped emojis agree
            if len(unique_emotions) == 1:
                emotion = next(iter(unique_emotions))
                emotion_counts[emotion] += 1
                pure_comments += 1

                writer.writerow([
                    emotion,
                    text,
                    " ".join(mapped_emojis)
                ])

            elif len(unique_emotions) > 1:
                conflicting_comments += 1

            checked += 1

    print("\n" + "=" * 60)
    print("Split:", split_name)
    print("Saved pure mapped comments to:", output_file)

    print("\nChecked:", checked)
    print("Comments with emojis:", emoji_comments)
    print("Pure mapped comments:", pure_comments)
    print("Conflicting mapped comments:", conflicting_comments)

    print("\nEmotion distribution (pure only):")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{emotion}: {count}")

    print("\nEmoji distribution:")
    for e, count in emoji_counts.most_common():
        print(f"{e}: {count}")

# Run both splits
for split_name in dataset.keys():
    process_split(split_name)
