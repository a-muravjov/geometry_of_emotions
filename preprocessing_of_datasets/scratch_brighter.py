from datasets import load_dataset
from collections import Counter

ds = load_dataset("brighter-dataset/BRIGHTER-emotion-categories", "swa")

emotion_cols = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

def analyze_split(split):
    total = 0
    neutral = 0
    single = 0
    multi = 0

    single_emotion_counts = Counter()

    for example in ds[split]:
        total += 1

        # get active emotions
        active = [e for e in emotion_cols if example[e] == 1]
        num_emotions = len(active)

        if num_emotions == 0:
            neutral += 1
        elif num_emotions == 1:
            single += 1
            single_emotion_counts[active[0]] += 1  # ONLY count here
        else:
            multi += 1

    print(f"\n--- {split.upper()} ---")
    print("Total:", total)
    print("Neutral:", neutral)
    print("Single emotion:", single)
    print("Multi emotion:", multi)

    print("\nSingle-emotion distribution:")
    for emotion, count in single_emotion_counts.most_common():
        print(f"{emotion}: {count}")

    return single_emotion_counts


# run
analyze_split("train")
analyze_split("dev")
analyze_split("test")