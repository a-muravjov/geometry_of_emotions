# Geometry of Emotions

Research code for studying how emotions are represented in multilingual language-model embedding spaces and how those representations transfer across languages.

The experiments focus on six emotions: **anger, disgust, fear, joy, sadness, and surprise**. English and Arabic serve as pivot languages, with multilingual evaluation covering Chinese, Estonian, German, Russian, Swahili, and Ukrainian.

The repository compares embedding geometry and linear classifiers with translation-based and lexicon-based approaches. It contains standalone experiment scripts and notebooks.

## Repository structure

| Folder | Purpose |
| --- | --- |
| [`preprocessing_of_datasets/`](preprocessing_of_datasets/) | Extracts, cleans, and converts datasets; creates emoji-based emotion labels; and translates evaluation data. |
| [`encoder_folder/`](encoder_folder/) | Main geometry experiments using multilingual encoder models, with separate English and Arabic source scripts. |
| [`decoder_folder/`](decoder_folder/) | Main geometry experiments using hidden-state representations from Qwen decoder models. |
| [`encoder_translation/`](encoder_translation/) | English- and Arabic-pivot translation baselines using encoder representations. |
| [`decoder_translation/`](decoder_translation/) | Translation baselines using Qwen decoder representations. |
| [`lexicon_baseline/`](lexicon_baseline/) | Language-specific Jupyter notebooks for NRC emotion-lexicon baselines. |

## Main experiments

### Encoder geometry

[`english_geometry.py`](encoder_folder/english_geometry.py) and [`arabic_geometry.py`](encoder_folder/arabic_geometry.py) extract **mean-pooled** and **CLS-token** embeddings and train logistic-regression emotion probes. They analyze emotion centroids and classifier weight directions through cosine similarities and PCA visualizations.

Cross-language experiments include direct transfer, bias-only calibration, orthogonal Procrustes alignment between emotion centroids, cosine-based classification against source emotion directions or prototypes, and classification in a six-dimensional space defined by the source probe weights. Methods that use labeled target-language calibration data are separate from direct transfer.

Three encoder models are tested:

- `google-bert/bert-base-multilingual-cased`
- `FacebookAI/xlm-roberta-base`
- `microsoft/mdeberta-v3-base`

### Decoder geometry

[`llamas_main_method.py`](decoder_folder/llamas_main_method.py) performs multilingual representation and transfer experiments using **mean pooling** and **last-token pooling** of decoder hidden states. The models are:

- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`

### Translation baselines

[`translation.py`](preprocessing_of_datasets/translation.py) uses `facebook/nllb-200-3.3B` to translate target-language evaluation texts into English and Arabic. The translated CSVs are then used by:

- [`english_translation_baseline.py`](encoder_translation/english_translation_baseline.py)
- [`arabic_translation_baseline.py`](encoder_translation/arabic_translation_baseline.py)
- [`llamas_translation.py`](decoder_translation/llamas_translation.py)

These experiments evaluate classifiers on texts translated into the classifier's source language. The decoder baseline writes results under `outputs_translation_baseline_decoder/`.

### Lexicon baselines

The six lexicon baseline notebooks cover Chinese, German, Estonian, Swahili, Russian, and Ukrainian. They match text against language-specific NRC emotion lexicons and weight a word's contribution inversely by the number of emotions associated with it. The notebooks report classification performance and lexicon coverage.
