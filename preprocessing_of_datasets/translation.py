import os
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------------
# Config
# -------------------------
ROOT = Path("thesis")
OUTPUT_DIR = ROOT / "translated_nllb"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "facebook/nllb-200-3.3B"
TEXT_COLUMN = "text"

BATCH_SIZE = 8
MAX_LENGTH = 256

TARGETS = {
    "english": "eng_Latn",
    "arabic": "arb_Arab",
}

FILES_TO_TRANSLATE = {
    "brighter_chinese.csv": "zho_Hans",
    "xed_chinese.csv": "zho_Hans",
    "xed_tw.csv": "zho_Hant",

    "xed_estonian.csv": "est_Latn",

    "brighter_german.csv": "deu_Latn",
    "xed_german.csv": "deu_Latn",

    "brighter_russian.csv": "rus_Cyrl",
    "xed_russian.csv": "rus_Cyrl",

    "brighter_swahili.csv": "swh_Latn",

    "brighter_ukrainian.csv": "ukr_Cyrl",
}

# -------------------------
# Load model
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
model.to(device)
model.eval()


def translate_texts(texts, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    outputs = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE), leave=False):
        batch = texts[i:i + BATCH_SIZE]

        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=forced_bos_token_id,
                max_length=MAX_LENGTH,
                num_beams=4,
            )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        outputs.extend(decoded)

    return outputs


# -------------------------
# Translate files
# -------------------------
for filename, src_lang in FILES_TO_TRANSLATE.items():
    input_path = ROOT / filename

    if not input_path.exists():
        print(f"Skipping missing file: {input_path}")
        continue

    print(f"\nTranslating {filename} from {src_lang}")

    df = pd.read_csv(input_path)

    if TEXT_COLUMN not in df.columns:
        print(f"Skipping {filename}: no '{TEXT_COLUMN}' column")
        continue

    texts = df[TEXT_COLUMN].fillna("").astype(str).tolist()

    for target_name, tgt_lang in TARGETS.items():
        output_path = OUTPUT_DIR / f"{input_path.stem}_to_{target_name}.csv"

        if output_path.exists():
            print(f"Already exists, skipping: {output_path}")
            continue

        print(f" → Target: {target_name} ({tgt_lang})")

        translated = translate_texts(
            texts=texts,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )

        out_df = df.copy()
        out_df["original_text"] = out_df[TEXT_COLUMN]
        out_df[TEXT_COLUMN] = translated
        out_df["source_language_code"] = src_lang
        out_df["target_language_code"] = tgt_lang

        out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Saved: {output_path}")

print("\nDone.")