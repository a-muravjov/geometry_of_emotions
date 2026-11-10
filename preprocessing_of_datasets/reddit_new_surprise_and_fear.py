from datasets import load_dataset
import emoji
from collections import defaultdict, Counter
import csv
import os

subreddits = [
    "socialskills",
    "relationship_advice",
    "sports",
    "technology",
    "travel",
    "femalefashionadvice",
    "malefashionadvice",
    "philosophy",
    "science"
]

max_samples_per_subreddit = 38_800_000
output_dir = "all_english_datasets_surprise_and_fear"
os.makedirs(output_dir, exist_ok=True)

emoji_to_emotion = {
    # JOY
    "💐": "joy", "🌹": "joy", "💝": "joy", "🙇": "joy", "💞": "joy", "🥂": "joy",
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

def extract_emojis(text):
    if not isinstance(text, str):
        return []
    return [e["emoji"] for e in emoji.emoji_list(text)]

for subreddit in subreddits:
    print("\n" + "=" * 70)
    print("Processing subreddit:", subreddit)

    dataset = load_dataset(
        "HuggingFaceGECLM/REDDIT_comments",
        split=subreddit,
        streaming=True
    )

    emotion_counts = defaultdict(int)
    emoji_counts = Counter()

    pure_comments = 0
    conflicting_comments = 0
    checked = 0
    emoji_comments = 0

    output_file = os.path.join(output_dir, f"{subreddit}_pure_mapped_comments_surprise.csv")

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["emotion", "text", "emojis"])

        for example in dataset:
            text = example["body"]

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

            if checked % 100000 == 0:
                print(f"{subreddit}: Processed {checked}")

            if checked >= max_samples_per_subreddit:
                break

    print("\nSaved pure mapped comments to:", output_file)

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