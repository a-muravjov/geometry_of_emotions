import pandas as pd

target_emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

simp = pd.read_csv("nrc_data_og/Chinese-Simplified-NRC-EmoLex.txt", sep="\t")
trad = pd.read_csv("nrc_data_og/Chinese-Traditional-NRC-EmoLex.txt", sep="\t")

simp = simp.rename(columns={"Chinese-Simplified Word": "word"})
trad = trad.rename(columns={"Chinese-Traditional Word": "word"})

simp = simp[["word"] + target_emotions]
trad = trad[["word"] + target_emotions]

combined = pd.concat([simp, trad], ignore_index=True)

combined["word"] = combined["word"].astype(str).str.strip()

combined = (
    combined
    .groupby("word", as_index=False)[target_emotions]
    .max()
)

combined.to_csv(
    "nrc_data_csv/chinese_nrc_combined.csv",
    index=False,
    encoding="utf-8-sig"
)

print(combined.head())
print("Rows:", len(combined))