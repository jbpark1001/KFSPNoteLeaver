from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project import ROOT, input_path, output_path, load_frame
import os

# Force Transformers to avoid TensorFlow
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd

kfsppororo=load_frame(ROOT / "derived/sectioned_notes.pkl")
sentence_df=load_frame(ROOT / "derived/sentences.pkl")

# Same model as your original code
kw_model = KeyBERT(SentenceTransformer('jhgan/ko-sbert-nli'))

# Section columns created by your current sectioning pipeline
third_cols = [f'third_{i}' for i in range(1, 4)]
quart_cols = [f'quart_{i}' for i in range(1, 5)]
third5_cols = [f'third5_{i}' for i in range(1, 4)]
shuf_cols = [f'shuf_third_{i}' for i in range(1, 4)]

primary_cols = ['1st_section','2nd_section','3rd_section']
all_section_cols = primary_cols + third_cols + quart_cols + third5_cols + shuf_cols

# Create empty keyword columns
for section in all_section_cols:
    kfsppororo[f"kluekeywordsentencetransformer_{section}"] = ""

# Extract keywords for each section
for ind in tqdm(kfsppororo.index, desc="Extracting KeyBERT keywords"):
    for section in all_section_cols:
        text = kfsppororo.at[ind, section]

        if isinstance(text, str) and len(text.strip()) > 0:
            keywords = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 1),
                stop_words=None,
                top_n=5
            )
        else:
            keywords = []

        kfsppororo.at[ind, f"kluekeywordsentencetransformer_{section}"] = keywords


#%%
# ==============================
# 8. TOKENIZE KEYBERT KEYWORDS WITH SCORES
# ==============================



def tokenize_keyword_columns_with_scores(df, keyword_columns, stopwords=None):
    if stopwords is None:
        stopwords = []

    # Load the KLUE tokenizer
    klue_tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")

    for col in keyword_columns:
        tokenized_col = []

        for keyword_list in tqdm(df[col], desc=f"Tokenizing {col}"):
            if not isinstance(keyword_list, list):
                tokenized_col.append([])
                continue

            token_score_list = []

            for word_score in keyword_list:
                if not isinstance(word_score, tuple) or len(word_score) != 2:
                    continue

                word, score = word_score

                try:
                    tokens = klue_tokenizer.tokenize(word)
                    tokens = [
                        t for t in tokens
                        if t not in stopwords and "#" not in t
                    ]

                    token_score_list.extend([(t, score) for t in tokens])

                except:
                    continue

            tokenized_col.append(token_score_list)

        output_col = f"tokenized{col}"
        df[output_col] = tokenized_col

    return df


keyword_cols = [
    f"kluekeywordsentencetransformer_{section}"
    for section in all_section_cols
]

kfsppororo = tokenize_keyword_columns_with_scores(
    kfsppororo,
    keyword_cols
)

print("KeyBERT keyword extraction and KLUE tokenization complete.")


#%%
# ==============================
# 9. KEYBERT KEYWORD EXTRACTION FOR SENTENCE_DF
# ==============================

sentence_df["kluekeywordsentencetransformer_sentence"] = ""

for ind in tqdm(sentence_df.index, desc="Extracting KeyBERT keywords from sentences"):
    text = sentence_df.at[ind, "sentence_text"]

    if isinstance(text, str) and len(text.strip()) > 0:
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 1),
            stop_words=None,
            top_n=5
        )
    else:
        keywords = []

    sentence_df.at[ind, "kluekeywordsentencetransformer_sentence"] = keywords


# ==============================
# 10. TOKENIZE SENTENCE_DF KEYWORDS WITH SCORES
# ==============================

sentence_keyword_cols = [
    "kluekeywordsentencetransformer_sentence"
]

sentence_df = tokenize_keyword_columns_with_scores(
    sentence_df,
    sentence_keyword_cols
)

print("Sentence-level KeyBERT keyword extraction and KLUE tokenization complete.")

kfsppororo.to_pickle(ROOT / "derived/keyword_notes.pkl")
sentence_df.to_pickle(ROOT / "derived/keyword_sentences.pkl")
