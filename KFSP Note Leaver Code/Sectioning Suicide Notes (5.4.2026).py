# -*- coding: utf-8 -*-
"""
Created on Mon May  4 13:42:23 2026

@author: Jae Bin Park
"""

# ==============================
# ROBUST SECTIONING PIPELINE
# This creates three separate types of sections alongside a relative position-based df
# ==============================

import numpy as np
import pandas as pd
import kss
from tqdm import tqdm
import random



print("Loading data...")

kfsppororo=pd.read_excel(r'C:\Users\Jae Bin Park\pororokfspnotes.xlsx')

print("Data loaded successfully!")

print("Running basic preprocessing...")

kfsppororo['NOTE_CONTENTDTL']=kfsppororo['NOTE_CONTENTDTL'].str.replace("엄마아빠", "엄마 아빠", regex=False)
kfsppororo['preprocessednotes']=kfsppororo['NOTE_CONTENTDTL'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)

kfsppororo['preprocessednotes']=kfsppororo['NOTE_CONTENTDTL'].str.replace(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 .!?]', "", regex=True)


print("Running basic preprocessing...Completed")


tqdm.pandas()

# ------------------------------
# 1. Safe sentence splitter
# ------------------------------
def safe_split_sentences(text):
    if not isinstance(text, str) or not text.strip():
        return []
    return [s.strip() for s in kss.split_sentences(text) if isinstance(s, str) and s.strip()]


# ------------------------------
# 2. Balanced sectioning function
# ------------------------------
def sectionize_note(text, n_bins=3, prefix='third', min_sentences=None):
    sents = safe_split_sentences(text)
    n = len(sents)

    out = {}

    for b in range(1, n_bins + 1):
        out[f'{prefix}_{b}'] = ''
        out[f'{prefix}_{b}_n_sent'] = 0

    out[f'{prefix}_total_n_sent'] = n

    if n == 0:
        return pd.Series(out)

    if min_sentences is not None and n < min_sentences:
        return pd.Series(out)

    # Relative position (0–1)
    rel_mid = (np.arange(n) + 0.5) / n

    # Assign bins
    bin_ids = np.floor(rel_mid * n_bins).astype(int)
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    for b in range(n_bins):
        bin_sents = [s for s, bid in zip(sents, bin_ids) if bid == b]
        out[f'{prefix}_{b+1}'] = ' '.join(bin_sents)
        out[f'{prefix}_{b+1}_n_sent'] = len(bin_sents)

    return pd.Series(out)


# ------------------------------
# 3. Generate section datasets
# ------------------------------
print("Generating thirds...")
third_df = kfsppororo['preprocessednotes'].progress_apply(
    lambda x: sectionize_note(x, n_bins=3, prefix='third', min_sentences=3)
)

print("Generating quartiles...")
quart_df = kfsppororo['preprocessednotes'].progress_apply(
    lambda x: sectionize_note(x, n_bins=4, prefix='quart', min_sentences=4)
)

print("Generating long-note thirds (≥5 sentences)...")
third5_df = kfsppororo['preprocessednotes'].progress_apply(
    lambda x: sectionize_note(x, n_bins=3, prefix='third5', min_sentences=5)
)

# Merge all
kfsppororo = pd.concat([kfsppororo, third_df, quart_df, third5_df], axis=1)


# ------------------------------
# 4. Sentence-level dataset (CRITICAL)
# ------------------------------
def build_sentence_level_df(df, text_col='preprocessednotes'):
    rows = []

    for idx, text in tqdm(df[text_col].items(), total=len(df), desc="Building sentence-level data"):
        sents = safe_split_sentences(text)
        n = len(sents)

        if n == 0:
            continue

        for i, sent in enumerate(sents):
            rel_pos = (i + 0.5) / n

            rows.append({
                'note_id': idx,
                'sentence_index': i + 1,
                'n_sent': n,
                'rel_pos': rel_pos,
                'sentence_text': sent,
                'third_bin': min(int(rel_pos * 3) + 1, 3),
                'quart_bin': min(int(rel_pos * 4) + 1, 4)
            })

    return pd.DataFrame(rows)

print("Building sentence-level dataset...")
sentence_df = build_sentence_level_df(kfsppororo)


# ------------------------------
# 5. OPTIONAL: shuffled null model
# ------------------------------
def shuffle_note_sentences(text, seed=None):
    sents = safe_split_sentences(text)
    if len(sents) <= 1:
        return text if isinstance(text, str) else ''
    rng = random.Random(seed)
    shuffled = sents.copy()
    rng.shuffle(shuffled)
    return ' '.join(shuffled)

print("Generating shuffled notes...")
kfsppororo['notes_shuffled'] = [
    shuffle_note_sentences(text, seed=i)
    for i, text in enumerate(kfsppororo['preprocessednotes'])
]

print("Sectioning shuffled notes...")
shuf_df = kfsppororo['notes_shuffled'].progress_apply(
    lambda x: sectionize_note(x, n_bins=3, prefix='shuf_third', min_sentences=3)
)

kfsppororo = pd.concat([kfsppororo, shuf_df], axis=1)


# ------------------------------
# 6. Section column groups
# ------------------------------
third_cols = [f'third_{i}' for i in range(1, 4)]
quart_cols = [f'quart_{i}' for i in range(1, 5)]
third5_cols = [f'third5_{i}' for i in range(1, 4)]
shuf_cols = [f'shuf_third_{i}' for i in range(1, 4)]

all_section_cols = third_cols + quart_cols + third5_cols + shuf_cols

print("Sectioning complete.")


#%%
# ==============================
# 7. KEYBERT KEYWORD EXTRACTION FOR NEW SECTIONS
# If you already ran the code above, you should have the related files. So you can just pd.read
# ==============================
import os

# Force Transformers to avoid TensorFlow
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd

kfsppororo=pd.read_excel(r'C:\Users\Jae Bin Park\kfsp_sectioned.xlsx')
sentence_df=pd.read_excel(r'C:\Users\Jae Bin Park\kfsp_sentence_relativepos.xlsx')

# Same model as your original code
kw_model = KeyBERT(SentenceTransformer('jhgan/ko-sbert-nli'))

# Section columns created by your current sectioning pipeline
third_cols = [f'third_{i}' for i in range(1, 4)]
quart_cols = [f'quart_{i}' for i in range(1, 5)]
third5_cols = [f'third5_{i}' for i in range(1, 4)]
shuf_cols = [f'shuf_third_{i}' for i in range(1, 4)]

all_section_cols = third_cols + quart_cols + third5_cols + shuf_cols

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
#%% Sentiment Extraction
# This requires the sentiment_environment

import pytorch_lightning as pl
import torch.nn as nn
from transformers import ElectraModel, AutoTokenizer
import torch
import pandas as pd

kfsppororo=pd.read_excel(r'C:\Users\Jae Bin Park\kfsp_sectioned.xlsx')
sentence_df=pd.read_excel(r'C:\Users\Jae Bin Park\kfsp_sentence_relativepos.xlsx')

LABELS = ['불평/불만',
 '환영/호의',
 '감동/감탄',
 '지긋지긋',
 '고마움',
 '슬픔',
 '화남/분노',
 '존경',
 '기대감',
 '우쭐댐/무시함',
 '안타까움/실망',
 '비장함',
 '의심/불신',
 '뿌듯함',
 '편안/쾌적',
 '신기함/관심',
 '아껴주는',
 '부끄러움',
 '공포/무서움',
 '절망',
 '한심함',
 '역겨움/징그러움',
 '짜증',
 '어이없음',
 '없음',
 '패배/자기혐오',
 '귀찮음',
 '힘듦/지침',
 '즐거움/신남',
 '깨달음',
 '죄책감',
 '증오/혐오',
 '흐뭇함(귀여움/예쁨)',
 '당황/난처',
 '경악',
 '부담/안_내킴',
 '서러움',
 '재미없음',
 '불쌍함/연민',
 '놀람',
 '행복',
 '불안/걱정',
 '기쁨',
 '안심/신뢰']

device = "cuda" if torch.cuda.is_available() else "cpu"

class KOTEtagger(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.electra = ElectraModel.from_pretrained("beomi/KcELECTRA-base", revision='v2021').to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base", revision='v2021')
        self.classifier = nn.Linear(self.electra.config.hidden_size, 44).to(device)
        
    def forward(self, text:str):
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors='pt',
        ).to(device)
        output = self.electra(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"])
        # Unpack the tuple to get last_hidden_state
        last_hidden_state = output[0]
        output = last_hidden_state[:, 0, :]  # Get the first token's hidden state
        output = self.classifier(output)
        output = torch.sigmoid(output)
        torch.cuda.empty_cache()
        
        return output

trained_model = KOTEtagger()
trained_model.load_state_dict(torch.load(r"C:\Users\Jae Bin Park\kote_pytorch_lightning.bin")) # <All keys matched successfully>라는 결과가 나오는지 확인!



import re
import pickle
from tqdm import tqdm

def clean_text(line):
    # Replace multiple hyphens with a single space
    line = re.sub(r'-+', ' ', line)
    # Replace everything except Korean characters, commas, and spaces with a space
    line = re.sub(r'[^가-힣, ]', ' ', line)
    # Replace multiple spaces with a single space
    line = re.sub(r'\s+', ' ', line).strip()
    line = re.sub(r'\.+', '', line)  # Remove sequences of dots
    line = line.replace(",", "")  # Remove commas
    # Optional: Add a period at the end if it doesn't already exist
    #if line and not line.endswith('.'):
    #    line += '.'
    return line

def split_korean_sentences(text):
    # List of common Korean sentence endings including additional specified endings
    endings = ["다", "요", "죠", "까", "어", "않아", "라", "꺼야", "거야"]
    
    # Create a pattern to match sentences ending with these endings followed by space, punctuation, or end of string
    pattern = re.compile(r'([가-힣]+(?:' + '|'.join(re.escape(ending) for ending in endings) + r'))(?=\s|[.?!]|$)')

    sentences = []
    start = 0

    # Iterate through the text to find matches
    for match in pattern.finditer(text):
        end = match.end()
        sentence = text[start:end].strip()
        sentences.append(sentence)
        start = end

    # Append any remaining text as a sentence
    remaining_text = text[start:].strip()
    if remaining_text:
        sentences.append(remaining_text)
    
    return sentences

def predict(sentence):
    if isinstance(sentence, str):  # Check if the input is a single sentence
        preds = trained_model(sentence)[0].detach().cpu().numpy()

        thresh = 0.7 # Adjust threshold as needed
        labels = [l for l, p in zip(LABELS, preds) if p > thresh]
        scores = [float(p) for l, p in zip(LABELS, preds) if p > thresh]  # Convert tensors to floats
        results = {'labels': labels, 'scores': scores}
        return results
    else:  # Handle list of sentences
        results = []
        for sent in sentence:
            results.append(predict(sent))
        return results




def predict(sentence):
    if isinstance(sentence, str):
        preds = trained_model(sentence)[0].detach().cpu().numpy()

        thresh = 0.7  # Adjust threshold as needed
        labels = [l for l, p in zip(LABELS, preds) if p > thresh]
        scores = [float(p) for l, p in zip(LABELS, preds) if p > thresh]

        return {'labels': labels, 'scores': scores}

    elif isinstance(sentence, list):
        results = []
        for sent in sentence:
            results.append(predict(sent))
        return results

    else:
        print(sentence)
        return {'labels': [], 'scores': []}  # or np.nan if you prefer

# ==============================
# APPLY KOTE SENTIMENT TO NEW SECTIONED NOTES
# ==============================

from tqdm import tqdm
tqdm.pandas()

# Your new section columns
third_cols = [f'third_{i}' for i in range(1, 4)]
quart_cols = [f'quart_{i}' for i in range(1, 5)]
third5_cols = [f'third5_{i}' for i in range(1, 4)]
shuf_cols = [f'shuf_third_{i}' for i in range(1, 4)]

section_cols = third_cols + quart_cols + third5_cols + shuf_cols

# Optional: check that all columns exist
missing_cols = [col for col in section_cols if col not in kfsppororo.columns]
if missing_cols:
    raise ValueError(f"These section columns are missing from kfsppororo: {missing_cols}")

# Run prediction on each section as a whole
for col in section_cols:
    new_col = f"{col}_sentiment"
    print(f"Running sentiment prediction: {col} -> {new_col}")
    kfsppororo[new_col] = kfsppororo[col].progress_apply(predict)

print("Section-level sentiment prediction complete.")

from tqdm import tqdm
tqdm.pandas()

print("Running sentence-level sentiment prediction...")

sentence_df["sentiment"] = sentence_df["sentence_text"].progress_apply(predict)

print("Sentence-level sentiment complete.")

#Ran whole code = 5.8.2026, 6:09 pm

#%%
kfsppororo['too_short'] = kfsppororo['preprocessednotes'].progress_apply(
    lambda x: len(kss.split_sentences(x)) < 3 if isinstance(x, str) else True
)

kfsppororo.to_excel("kfsp_sectioned.xlsx")

#%% Raw vs Summarized Final Version + Stability Checks
# -*- coding: utf-8 -*-
"""
Final Version: Summarized-vs-Raw detector for Korean suicide notes
(weak supervision -> classifier) + stability checks

Features:
- Handles investigator-style summaries:
  요약 markers, reportative endings, page/A4 mentions, role tags,
  "사망 당일", "상세내용 알수없음", bullet headers, numbered lists, etc.
- Handles message summaries:
  문자/카톡/메일 + 인용부호 OR 보냄/전송/발송/송부 verbs
- Handles written summaries:
  달력/수첩 등에 '…' 이라고 써있음/적혀있음/적어놓았음
- Default drops raw length from ML features to reduce length bias
  (rules may still use it as a guard)

Added stability checks:
1. Threshold sensitivity
2. Bootstrap stability of class proportions
3. Counterfactual cue-removal tests
4. Mixture separation diagnostics
5. Weak-label agreement / proxy checks
6. Optional length-bias diagnostics
"""

# =========================
# Imports
# =========================
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, brier_score_loss
from sklearn.mixture import GaussianMixture
from sklearn.utils import resample

from scipy import sparse

# =========================
# Load your data here
# =========================
# Example:
kfsppororo= pd.read_excel(r'C:\Users\Jae Bin Park\kfsp_sectioned.xlsx')
kfsp=pd.read_excel(r'G:\KFSP\Raw Data\KFSPdatacopy.xlsx')
allnotes = kfsp[~kfsp["NOTE_CONTENTDTL"].isna()].copy()

# Uncomment and edit as needed:
# kfsppororo = pd.read_excel(r'C:\Users\Jae Bin Park\kfsppororosplit.xlsx')
# allnotes = kfsp[~kfsp["NOTE_CONTENTDTL"].isna()].copy()

# =========================
# Regex & Lexicons
# =========================

KOREAN_RE = re.compile(r"[가-힣]")
QUOTE_RE = re.compile(r"[\"“”‘’'＂]")

DATE_RE = re.compile(r"(?:19|20)\d{2}\s*[./-]\s*\d{1,2}\s*[./-]\s*\d{1,2}")
TIME_RE = re.compile(r"(\d{1,2}\s*시(\s*\d{1,2}\s*분)?)")

LEADING_DATE_RE = re.compile(
    r"^\s*(?:(?:19|20)\d{2}|\d{2})\s*(?:[.\-/]|년)\s*\d{1,2}\s*(?:[.\-/]|월)?\s*\d{1,2}\s*(?:일|\.)?"
)

POLICE_TERMS = [
    "경찰","수사","조사","진술","보고","기재","확인","확인됨","확인됐다","파악",
    "파악됐다","판단","판단된다","추정","추정된다","발견","발견되","현장","사건",
    "사망자","고인","작성","담당","기록","인용","문구","부검","감식","목격자",
    "변사자","시신","유류품"
]

REPORTED_SPEECH_PATTERNS = [
    r"라고\s*(말|전|진술|하였다|했다)",
    r"이라며\s*(말|했다|전했다)",
    r"라는\s*내용",
    r"라는\s*(메시지|문자)"
]

REPORTED_SUMMARY_END_RE = re.compile(
    r"(?:라?고\s*함|다?고\s*함|고\s*하였음|고\s*하였다|고\s*전했[다음]|고\s*진술했[다음])\b"
)

FORMAL_MEDIA_PASSIVE = [
    "된 것으로","것으로 보","것으로 추정","보인다","파악됐다",
    "확인됐다","으로 보임","으로 판단된다","하였다","했다","한 바 있다","하였음"
]

FIRST_PERSON = ["나","내","난","나는","내가","저","전","제","저는","제가","우리","우린"]

SECOND_PERSON_KIN = [
    "너","당신","엄마","아빠","아들","딸","오빠","언니","형","누나","할머니",
    "할아버지","친구","여보","아이들","애들",
    "사랑","사랑해","미안","죄송","용서","보고싶","보고 싶","울지말","고맙"
]

DIRECT_ADDR_TOKENS = ["엄마","아빠","여보","형","언니","오빠","누나","아이들","애들","딸","아들"]

FOUND_WILL_RE = re.compile(
    r"유서\s*(?:가|를|은|이)?\s*(?:발견|확인)[^.,\n)]{0,12}(?:됨|됐다|되었|되다)"
)
PAREN_ENUM_RE = re.compile(r"(?:^|[,，]\s*)[^,()]{1,20}\([^()]{1,120}\)")
NUMBERED_ENUM_RE = re.compile(r"(?:(?<=\s)|^)\d{1,2}\s*[.)](?=\s|[^0-9])")
ANGLE_HEADER_RE = re.compile(r"^\s*<[^>]{1,40}>\s*", re.M)
REDACTION_RE = re.compile(r"(\(\*{2,}\)|\*{2,}|○○|△△|◇◇|□□|ＸＸ|XX|xx)")
PAGE_TOKEN_RE = re.compile(r"\d+\s*(장|쪽|페이지)")
KOR_PAGE_TOKEN_RE = re.compile(r"(한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*(장|쪽|페이지)")
SUMMARIZER_PHRASES = [
    "와 같은 내용의 유서","같은 내용의 유서","유서 내용은","유서에는","유서 내용("
]
CONTAINER_TERMS = ["수첩","메모장","메모지","가방","주머니","노트","메모","달력"]

YOYAK_MARK_RE = re.compile(r"요약\s*[\-:)\]]")
PAPER_REF_RE = re.compile(r"(?:A\s*4|에이포)\s*용지")

ROLE_TAG_RE = re.compile(
    r"\((?:변사자|아내|남편|배우자|부인|모|부|부모|딸|아들|형|누나|언니|오빠|지인|친구|유족)\)"
)

MSG_TOKENS = ["문자메시지","메시지","문자","카카오톡","카톡","메일","이메일","email","e-mail","임시저장"]
MSG_SEND_VERB_RE = re.compile(r"(보냄|보냈다|보내며|보내고|전송|발송|송부)")

POLICE_INSTR_RE = re.compile(
    r"경찰(?:서)?(?:에|에게)?\s*신고\s*(?:하|해)(?:세요|시오|십시오|셔요|줘|주(?:세|십)요|라)?"
)

DECEASED_DAY_RE = re.compile(r"사망\s*당일")
UNKNOWN_CONTENT_RE = re.compile(r"(상세\s*내용\s*알\s*수\s*없음|내용\s*불상)")

BULLET_LABEL_WORDS = [
    "자살방법","사체처리","사후\s*처리","자살이유","개인적\s*메[시세]지"
]
BULLET_HEADER_RE = re.compile(
    rf"(?m)^\s*[-–—]\s*(?:{'|'.join(BULLET_LABEL_WORDS)})\s*[:：)\]]"
)

WRITE_VERB_RE = re.compile(r"(적(?:었|은|혀|어놓|어 놓)|쓴|써(?:놓|있|서)|쓰여)")
REPORTED_WRITTEN_END_RE = re.compile(
    r"(?:라?고)\s*(?:써\s*있(?:다|음)|적혀\s*있(?:다|음)|적어\s*놓았(?:다|음)?|쓰여\s*있(?:다|음))"
)
WRITTEN_STATE_RE = re.compile(
    r"(?:써\s*있(?:다|음)|적혀\s*있(?:다|음)|적어\s*놓았(?:다|음)?|쓰여\s*있(?:다|음))"
)

SENT_SPLIT_RE = re.compile(r"[.!?…]+|\n")

# =========================
# Helpers
# =========================

def normalize_quotes(s: str) -> str:
    return s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'").replace("＂", "\"")

def tfidf_clean(s: str) -> str:
    s = re.sub(r"\*+", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s

def hangul_ratio(s: str) -> float:
    if not s:
        return 0.0
    total = len(s)
    if total == 0:
        return 0.0
    return len(KOREAN_RE.findall(s)) / total

def count_matches(s: str, patterns: List[str]) -> int:
    return sum(1 for p in patterns if re.search(p, s))

def count_list_tokens(s: str, vocab: List[str]) -> int:
    return sum(s.count(w) for w in vocab)

def declarative_da_ratio(s: str) -> float:
    toks = [t.strip() for t in SENT_SPLIT_RE.split(s) if t and t.strip()]
    if not toks:
        return 0.0
    ends = sum(1 for t in toks if re.search(r"(다|니다)$", t))
    return ends / len(toks)

# =========================
# Signals dataclass
# =========================

@dataclass
class RuleSignals:
    police_terms: int
    reported_speech: int
    formal_passive: int
    quote_count: int
    date_count: int
    time_count: int
    first_person: int
    second_kin: int
    hangul_ratio: float
    length: int
    leading_date: int
    found_will: int
    parenthetical_enum_count: int
    summarizer_phrases_count: int
    yuseo_count: int
    numbered_enum_count: int
    angle_header: int
    redaction_count: int
    page_token_count: int
    container_count: int
    admin_verbs_count: int
    msg_token_count: int
    da_ratio: float
    direct_addr_count: int
    police_instruction: int
    yoyak_marker: int
    reportative_end: int
    kor_page_token_count: int
    paper_ref: int
    role_tag_count: int
    deceased_day: int
    unknown_content: int
    msg_send_verb: int
    bullet_header_count: int
    write_verb: int
    calendar_token: int
    written_end: int
    written_state: int

def extract_rule_signals(text: str) -> RuleSignals:
    if text is None:
        text = ""
    s = normalize_quotes(text)

    return RuleSignals(
        police_terms=count_list_tokens(s, POLICE_TERMS),
        reported_speech=count_matches(s, REPORTED_SPEECH_PATTERNS),
        formal_passive=count_list_tokens(s, FORMAL_MEDIA_PASSIVE),
        quote_count=len(QUOTE_RE.findall(s)),
        date_count=len(DATE_RE.findall(s)),
        time_count=len(TIME_RE.findall(s)),
        first_person=count_list_tokens(s, FIRST_PERSON),
        second_kin=count_list_tokens(s, SECOND_PERSON_KIN),
        hangul_ratio=hangul_ratio(s),
        length=len(s.strip()),
        leading_date=1 if LEADING_DATE_RE.search(s) else 0,
        found_will=1 if FOUND_WILL_RE.search(s) else 0,
        parenthetical_enum_count=len(PAREN_ENUM_RE.findall(s)),
        summarizer_phrases_count=count_list_tokens(s, SUMMARIZER_PHRASES),
        yuseo_count=s.count("유서"),
        numbered_enum_count=len(NUMBERED_ENUM_RE.findall(s)),
        angle_header=1 if ANGLE_HEADER_RE.search(s) else 0,
        redaction_count=len(REDACTION_RE.findall(s)),
        page_token_count=len(PAGE_TOKEN_RE.findall(s)),
        container_count=count_list_tokens(s, CONTAINER_TERMS),
        admin_verbs_count=count_list_tokens(
            s, ["신고","부탁","전달","안내","수거","보관","확보","인계","판매","팔기","정리","처분","회수","화장","수리","태워"]
        ),
        msg_token_count=count_list_tokens(s, MSG_TOKENS),
        da_ratio=declarative_da_ratio(s),
        direct_addr_count=count_list_tokens(s, DIRECT_ADDR_TOKENS),
        police_instruction=1 if POLICE_INSTR_RE.search(s) else 0,
        yoyak_marker=1 if YOYAK_MARK_RE.search(s) else 0,
        reportative_end=1 if REPORTED_SUMMARY_END_RE.search(s) else 0,
        kor_page_token_count=len(KOR_PAGE_TOKEN_RE.findall(s)),
        paper_ref=1 if PAPER_REF_RE.search(s) else 0,
        role_tag_count=len(ROLE_TAG_RE.findall(s)),
        deceased_day=1 if DECEASED_DAY_RE.search(s) else 0,
        unknown_content=1 if UNKNOWN_CONTENT_RE.search(s) else 0,
        msg_send_verb=1 if MSG_SEND_VERB_RE.search(s) else 0,
        bullet_header_count=len(BULLET_HEADER_RE.findall(s)),
        write_verb=1 if WRITE_VERB_RE.search(s) else 0,
        calendar_token=s.count("달력"),
        written_end=1 if REPORTED_WRITTEN_END_RE.search(s) else 0,
        written_state=1 if WRITTEN_STATE_RE.search(s) else 0,
    )

# =========================
# Context helper
# =========================

def has_investigator_ctx(sig: RuleSignals) -> bool:
    if sig.yoyak_marker == 1 or sig.reportative_end == 1:
        return True
    if sig.deceased_day == 1 or sig.unknown_content == 1:
        return True
    if sig.role_tag_count >= 1 or sig.paper_ref == 1:
        return True
    if sig.kor_page_token_count >= 1:
        return True
    if sig.bullet_header_count >= 1:
        return True

    if sig.written_end == 1:
        return True
    if (sig.calendar_token >= 1 and sig.write_verb == 1 and sig.quote_count >= 1):
        return True
    if (sig.written_state == 1) and (sig.quote_count >= 1 or sig.yuseo_count >= 1):
        return True

    if sig.msg_token_count >= 1 and (sig.quote_count >= 1 or sig.msg_send_verb == 1):
        return True

    if sig.yuseo_count >= 1 or sig.angle_header == 1 or sig.found_will == 1 or sig.leading_date == 1:
        return True
    if sig.page_token_count >= 1 or sig.numbered_enum_count >= 3 or sig.summarizer_phrases_count >= 1:
        return True
    if sig.police_terms >= 1 and sig.police_instruction == 0:
        return True
    if sig.msg_token_count >= 1 and sig.reported_speech >= 1:
        return True
    return False

# =========================
# Weak labeling functions
# =========================

def lf_yoyak_marker(sig: RuleSignals) -> int:
    return 1 if sig.yoyak_marker == 1 else -1

def lf_reportative_end(sig: RuleSignals) -> int:
    return 1 if sig.reportative_end == 1 else -1

def lf_role_tag(sig: RuleSignals) -> int:
    return 1 if sig.role_tag_count >= 1 else -1

def lf_deceased_day(sig: RuleSignals) -> int:
    return 1 if sig.deceased_day == 1 else -1

def lf_unknown_content(sig: RuleSignals) -> int:
    return 1 if sig.unknown_content == 1 else -1

def lf_message_quote_or_send(sig: RuleSignals) -> int:
    return 1 if (sig.msg_token_count >= 1 and (sig.quote_count >= 1 or sig.msg_send_verb == 1)) else -1

def lf_bullet_headers(sig: RuleSignals) -> int:
    return 1 if sig.bullet_header_count >= 2 else -1

def lf_written_end(sig: RuleSignals) -> int:
    return 1 if sig.written_end == 1 else -1

def lf_calendar_written_quote(sig: RuleSignals) -> int:
    return 1 if (sig.calendar_token >= 1 and sig.write_verb == 1 and sig.quote_count >= 1) else -1

def lf_found_will(sig: RuleSignals) -> int:
    return 1 if sig.found_will >= 1 else -1

def lf_parenthetical_enum(sig: RuleSignals) -> int:
    return 1 if (sig.yuseo_count >= 1 and sig.parenthetical_enum_count >= 2) else -1

def lf_summarizer_phrase(sig: RuleSignals) -> int:
    return 1 if sig.summarizer_phrases_count >= 1 else -1

def lf_inventory_meta(sig: RuleSignals) -> int:
    cond1 = (sig.angle_header == 1)
    cond2 = (sig.yuseo_count >= 1 and (sig.page_token_count >= 1 or sig.kor_page_token_count >= 1))
    cond3 = (sig.yuseo_count >= 1 and sig.container_count >= 1 and sig.numbered_enum_count >= 1)
    cond4 = (sig.paper_ref == 1 and (sig.kor_page_token_count >= 1 or sig.page_token_count >= 1))
    return 1 if (cond1 or cond2 or cond3 or cond4) else -1

def lf_numbered_will(sig: RuleSignals) -> int:
    return 1 if (sig.yuseo_count >= 1 and sig.numbered_enum_count >= 3) else -1

def lf_redaction(sig: RuleSignals) -> int:
    return 1 if (sig.redaction_count >= 1 and sig.yuseo_count >= 1) else -1

def lf_message_container(sig: RuleSignals) -> int:
    return 1 if (sig.msg_token_count >= 1 and sig.reported_speech >= 1) else -1

def lf_leading_date_summary(sig: RuleSignals) -> int:
    return 1 if (sig.leading_date == 1 and has_investigator_ctx(sig)) else -1

def lf_police_narrative(sig: RuleSignals) -> int:
    score = 0
    score += (sig.police_terms >= 2 and sig.police_instruction == 0)
    score += (sig.reported_speech >= 1)
    score += (sig.date_count + sig.time_count >= 1)
    score += (sig.formal_passive >= 1)
    return 1 if score >= 2 else -1

def lf_quotes_reported(sig: RuleSignals) -> int:
    return 1 if (sig.quote_count >= 2 and sig.reported_speech >= 1) else -1

def lf_formal_passive(sig: RuleSignals) -> int:
    return 1 if (sig.formal_passive >= 2 and has_investigator_ctx(sig)) else -1

def lf_raw_admin_direct(sig: RuleSignals) -> int:
    no_ctx = (not has_investigator_ctx(sig))
    has_admin = (sig.admin_verbs_count >= 2)
    has_direct = (sig.direct_addr_count >= 1)
    firstp = (sig.first_person >= 1)
    not_listy = (sig.numbered_enum_count <= 1 and sig.parenthetical_enum_count <= 1)
    return 0 if (no_ctx and has_admin and has_direct and firstp and not_listy) else -1

def lf_diary_raw(sig: RuleSignals) -> int:
    no_ctx = (not has_investigator_ctx(sig))
    diaryish = (sig.da_ratio >= 0.6)
    rich_affect = (sig.first_person >= 2 and sig.second_kin >= 1)
    not_listy = (sig.numbered_enum_count <= 1 and sig.parenthetical_enum_count <= 1)
    return 0 if (no_ctx and diaryish and rich_affect and not_listy) else -1

def lf_strong_raw_monologue(sig: RuleSignals) -> int:
    no_ctx = (not has_investigator_ctx(sig))
    affect_heavy = (sig.first_person >= 2 and sig.second_kin >= 2)
    not_listy = (sig.numbered_enum_count <= 1 and sig.parenthetical_enum_count <= 1)
    return 0 if (affect_heavy and no_ctx and not_listy) else -1

def lf_firstperson_affect(sig: RuleSignals) -> int:
    if sig.first_person >= 1 and sig.second_kin >= 1 and not has_investigator_ctx(sig):
        return 0
    return -1

def lf_length_guard(sig: RuleSignals) -> int:
    if sig.length <= 15 and sig.first_person >= 1:
        return 0
    return -1

def lf_korean_ratio(sig: RuleSignals) -> int:
    return -1

LABELING_FUNCTIONS = [
    lf_yoyak_marker,
    lf_reportative_end,
    lf_role_tag,
    lf_deceased_day,
    lf_unknown_content,
    lf_message_quote_or_send,
    lf_bullet_headers,
    lf_written_end,
    lf_calendar_written_quote,
    lf_leading_date_summary,
    lf_found_will,
    lf_parenthetical_enum,
    lf_summarizer_phrase,
    lf_inventory_meta,
    lf_numbered_will,
    lf_redaction,
    lf_message_container,
    lf_police_narrative,
    lf_quotes_reported,
    lf_formal_passive,
    lf_raw_admin_direct,
    lf_diary_raw,
    lf_strong_raw_monologue,
    lf_firstperson_affect,
    lf_length_guard,
    lf_korean_ratio,
]

def apply_weak_labels(sig: RuleSignals) -> Tuple[int, float, Dict[str, int]]:
    votes, per_rule = [], {}
    for f in LABELING_FUNCTIONS:
        v = f(sig)
        per_rule[f.__name__] = v
        if v != -1:
            votes.append(v)
    if not votes:
        return -1, 0.0, per_rule
    ones = sum(v == 1 for v in votes)
    zeros = sum(v == 0 for v in votes)
    if ones == zeros:
        return -1, 0.0, per_rule
    label = 1 if ones > zeros else 0
    conf = abs(ones - zeros) / len(votes)
    return label, conf, per_rule

# =========================
# Sklearn transformers
# =========================

class SignalExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, text_col: str = "text", use_length_feature: bool = False):
        self.text_col = text_col
        self.use_length_feature = use_length_feature

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            texts = X[self.text_col].fillna("").astype(str).tolist()
        elif isinstance(X, pd.Series):
            texts = X.fillna("").astype(str).tolist()
        else:
            arr = np.asarray(X)
            if arr.ndim == 2 and arr.shape[1] == 1:
                texts = pd.Series(arr[:, 0]).fillna("").astype(str).tolist()
            else:
                texts = pd.Series(arr).fillna("").astype(str).tolist()

        rows = []
        for txt in texts:
            s = extract_rule_signals(txt)
            feats = [
                s.police_terms, s.reported_speech, s.formal_passive, s.quote_count,
                s.date_count, s.time_count, s.first_person, s.second_kin,
                s.hangul_ratio,
                s.leading_date, s.found_will, s.parenthetical_enum_count, s.summarizer_phrases_count, s.yuseo_count,
                s.numbered_enum_count, s.angle_header, s.redaction_count, s.page_token_count, s.container_count,
                s.admin_verbs_count, s.msg_token_count, s.da_ratio, s.direct_addr_count, s.police_instruction,
                s.yoyak_marker, s.reportative_end, s.kor_page_token_count, s.paper_ref,
                s.role_tag_count, s.deceased_day, s.unknown_content,
                s.msg_send_verb, s.bullet_header_count,
                s.write_verb, s.calendar_token, s.written_end, s.written_state
            ]
            if self.use_length_feature:
                feats.insert(9, s.length)
            rows.append(feats)
        return np.array(rows)

def select_col(X, col: str):
    if isinstance(X, pd.DataFrame):
        return X[col]
    if isinstance(X, dict):
        return X[col]
    return X

def to_1d(x):
    return np.asarray(x).ravel()

def to_csr(X):
    return sparse.csr_matrix(X)

# =========================
# Main detector
# =========================

class SummarizationDetector:
    def __init__(
        self,
        text_col: str = "text",
        random_state: int = 42,
        min_rule_conf: float = 0.34,
        lr_solver: str = "saga",
        lr_penalty: str = "l2",
        lr_C: float = 0.5,
        lr_max_iter: int = 5000,
        lr_tol: float = 1e-3,
        calibrate_default: bool = True,
        use_length_feature: bool = False,
    ):
        self.text_col = text_col
        self.random_state = random_state
        self.min_rule_conf = min_rule_conf
        self.pipe = None
        self.label_map_ = {0: "raw", 1: "summarized"}
        self.lr_solver = lr_solver
        self.lr_penalty = lr_penalty
        self.lr_C = lr_C
        self.lr_max_iter = lr_max_iter
        self.lr_tol = lr_tol
        self.calibrate_default = calibrate_default
        self.use_length_feature = use_length_feature

    def _build_pipeline(self, calibrated: bool, cv: Optional[int] = None):
        text_branch = Pipeline([
            ("select", FunctionTransformer(select_col, validate=False, kw_args={"col": self.text_col})),
            ("to1d", FunctionTransformer(to_1d, validate=False)),
            ("tfidf", TfidfVectorizer(
                analyzer="char", ngram_range=(3, 5), min_df=2, max_features=200000,
                preprocessor=tfidf_clean
            )),
        ])

        signals_branch = Pipeline([
            ("select", FunctionTransformer(select_col, validate=False, kw_args={"col": self.text_col})),
            ("signals", SignalExtractor(text_col=self.text_col, use_length_feature=self.use_length_feature)),
            ("scale", StandardScaler(with_mean=False)),
            ("to_csr", FunctionTransformer(to_csr, validate=False)),
        ])

        features = FeatureUnion([
            ("text", text_branch),
            ("signals", signals_branch),
        ])

        base = LogisticRegression(
            solver=self.lr_solver,
            penalty=self.lr_penalty,
            C=self.lr_C,
            max_iter=self.lr_max_iter,
            tol=self.lr_tol,
            class_weight="balanced",
            random_state=self.random_state,
        )

        clf = CalibratedClassifierCV(base, cv=cv, method="sigmoid") if calibrated else base
        self.pipe = Pipeline([("features", features), ("clf", clf)])

    def _weak_label_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        recs = []
        for i, txt in enumerate(df[self.text_col].fillna("")):
            sig = extract_rule_signals(txt)
            y, conf, per_rule = apply_weak_labels(sig)
            recs.append({
                "idx": i,
                "weak_label": y,
                "rule_conf": conf,
                "signals": sig,
                **{f"rule_{k}": v for k, v in per_rule.items()},
            })
        w = pd.DataFrame(recs)
        keep = (w["weak_label"] != -1) & (w["rule_conf"] >= self.min_rule_conf)
        return w.loc[keep].reset_index(drop=True)

    def fit(self, df: pd.DataFrame):
        if self.text_col not in df.columns:
            raise KeyError(f"Column '{self.text_col}' not found in DataFrame.")
        df = df.reset_index(drop=True)

        wd = self._weak_label_dataframe(df)
        n = len(wd)
        if n < 50:
            print(f"[WARN] Only {n} confident weak labels; consider lowering min_rule_conf or adding a few gold labels.")
        if n == 0:
            raise ValueError("No confident weak labels to train on. Adjust rules or min_rule_conf.")

        X = df.iloc[wd["idx"].to_numpy()][[self.text_col]].reset_index(drop=True)
        y = wd["weak_label"].values

        classes, counts = np.unique(y, return_counts=True)
        min_class_count = counts.min()
        n_classes = len(classes)

        if self.calibrate_default and min_class_count >= 2:
            cv = int(min(3, min_class_count))
            calibrated = True
        else:
            if self.calibrate_default and min_class_count < 2:
                print("[Info] Skipping probability calibration (too few samples per class).")
            calibrated = False
            cv = None

        self._build_pipeline(calibrated=calibrated, cv=cv)
        self.pipe.fit(X, y)

        if n >= (n_classes * 2):
            test_size_int = max(int(np.ceil(0.2 * n)), n_classes)
            test_size_int = min(test_size_int, n - 1)
            try:
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=test_size_int, random_state=self.random_state, stratify=y
                )
                acc = (self.pipe.predict(Xte) == yte).mean()
                print(f"[Info] Sanity accuracy vs. weak labels: {acc:.3f} (interpret cautiously)")
            except ValueError as e:
                print(f"[Info] Skipping sanity split: {e}")
        else:
            print("[Info] Skipping sanity split; too few weak labels for a stratified holdout.")

        self._weak_df_ = wd
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.pipe is None:
            raise RuntimeError("Call fit() first.")
        pm = self.pipe.predict_proba(df[[self.text_col]])[:, 1]

        rule_p = []
        ctxless_flags = []
        for txt in df[self.text_col].fillna(""):
            sig = extract_rule_signals(txt)
            y, conf, _ = apply_weak_labels(sig)
            if y == -1:
                rp = 0.5
            elif y == 1:
                rp = 0.5 + 0.5 * conf
            else:
                rp = 0.5 - 0.5 * conf
            rule_p.append(rp)
            ctxless_flags.append(not has_investigator_ctx(sig))

        rule_p = np.array(rule_p)
        ctxless_flags = np.array(ctxless_flags, dtype=bool)

        blended = 0.7 * pm + 0.3 * rule_p
        blended[ctxless_flags] = 0.35 * pm[ctxless_flags] + 0.65 * rule_p[ctxless_flags]

        return np.vstack([1 - blended, blended]).T

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(df)[:, 1]
        return (proba >= threshold).astype(int)

    def score_dataframe(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        proba = self.predict_proba(df)[:, 1]
        yhat = (proba >= threshold).astype(int)
        rules_fired = []
        signals_list = []

        for txt in df[self.text_col].fillna(""):
            sig = extract_rule_signals(txt)
            _, _, per_rule = apply_weak_labels(sig)
            fired = [k for k, v in per_rule.items() if v in (0, 1)]
            rules_fired.append(", ".join(fired))
            signals_list.append(sig.__dict__)

        out = df.copy()
        out["p_summarized"] = proba
        out["pred_label"] = np.where(yhat == 1, "summarized", "raw")
        out["rule_signals"] = signals_list
        out["rules_fired"] = rules_fired
        return out

    def propose_for_review(self, df: pd.DataFrame, k: int = 100, low: float = 0.4, high: float = 0.6) -> pd.DataFrame:
        proba = self.predict_proba(df)[:, 1]
        mask = (proba > low) & (proba < high)
        cand = df.loc[mask].copy()
        cand["p_summarized"] = proba[mask]
        return cand.iloc[np.argsort(np.abs(cand["p_summarized"] - 0.5))[:k]]

# =========================
# Stability check functions
# =========================

def threshold_sensitivity(det, df, text_col="NOTE_CONTENTDTL", thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.45, 0.61, 0.01)

    proba = det.predict_proba(df[[text_col]])[:, 1]
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        n_sum = int(pred.sum())
        n_raw = int((1 - pred).sum())
        pct_raw = float((1 - pred).mean())
        rows.append({
            "threshold": round(float(t), 3),
            "n_summarized": n_sum,
            "n_raw": n_raw,
            "pct_raw": pct_raw
        })
    return pd.DataFrame(rows)

def bootstrap_class_proportion(det, df, text_col="NOTE_CONTENTDTL", B=200, frac=1.0, seed=42, threshold=0.5):
    rng = np.random.default_rng(seed)
    n = len(df)
    vals = []

    for b in range(B):
        idx = rng.integers(0, n, int(n * frac))
        sub = df.iloc[idx].reset_index(drop=True)
        p = det.predict_proba(sub[[text_col]])[:, 1]
        pred = (p >= threshold).astype(int)
        pct_raw = float((1 - pred).mean())
        vals.append(pct_raw)

    vals = np.array(vals)
    return {
        "bootstrap_B": B,
        "mean_pct_raw": float(vals.mean()),
        "sd_pct_raw": float(vals.std(ddof=1)),
        "ci95_low": float(np.percentile(vals, 2.5)),
        "ci95_high": float(np.percentile(vals, 97.5)),
    }

def _edit_text(s, rules):
    for pat, rep in rules:
        s = pat.sub(rep, s)
    return s

_SUMMARY_PAT = [
    (re.compile(r"<[^>]{1,40}>"), ""),
    (re.compile(r"유서\s*(?:내용|에는|내용\(|가|을|이)?"), ""),
    (re.compile(r"(발견|확인)\s*(?:됨|됐다|되었|되다)"), ""),
    (re.compile(r"(?m)^\s*\d{1,2}\s*[.)]\s*"), ""),
    (re.compile(r"(?m)^\s*(?:19|20)\d{2}[.\-/년]\s*\d{1,2}"), ""),
    (re.compile(r"요약\s*[\-:)\]]"), ""),
]

_RAW_PAT = [
    (re.compile(r"(엄마|아빠|여보|형|언니|오빠|누나|아이들|애들|딸|아들)"), ""),
    (re.compile(r"(사랑해|미안|울지마|용서|고맙)"), ""),
]

def counterfactual_tests(det, df, text_col="NOTE_CONTENTDTL"):
    base_p = det.predict_proba(df[[text_col]])[:, 1]

    df_sumless = df.copy()
    df_sumless[text_col] = df_sumless[text_col].fillna("").map(lambda t: _edit_text(t, _SUMMARY_PAT))

    df_rawless = df.copy()
    df_rawless[text_col] = df_rawless[text_col].fillna("").map(lambda t: _edit_text(t, _RAW_PAT))

    p_sumless = det.predict_proba(df_sumless[[text_col]])[:, 1]
    p_rawless = det.predict_proba(df_rawless[[text_col]])[:, 1]

    labeled = (base_p >= 0.5).astype(int)

    drop_on_sumless = (base_p[labeled == 1] - p_sumless[labeled == 1]).mean() if (labeled == 1).any() else np.nan
    rise_on_rawless = (p_rawless[labeled == 0] - base_p[labeled == 0]).mean() if (labeled == 0).any() else np.nan

    return {
        "mean_delta_p_on_summarized_after_stripping_summary_cues": float(drop_on_sumless),
        "mean_delta_p_on_raw_after_stripping_raw_cues": float(rise_on_rawless)
    }

def _signals_df(scored):
    if "rule_signals" not in scored.columns:
        raise ValueError("score_dataframe output with 'rule_signals' required.")
    sig = pd.DataFrame(list(scored["rule_signals"]))
    sig.index = scored.index
    return sig

def _lf_label(sig_row):
    y, conf, _ = apply_weak_labels(RuleSignals(**sig_row.to_dict()))
    return y

def _has_investigator_ctx(sig_row):
    return has_investigator_ctx(RuleSignals(**sig_row.to_dict()))

def silver_metrics(scored, text_col="NOTE_CONTENTDTL"):
    sig = _signals_df(scored)

    S = (
        (sig["found_will"] == 1) |
        (sig["angle_header"] == 1) |
        ((sig["leading_date"] == 1) & sig.apply(_has_investigator_ctx, axis=1)) |
        (sig["numbered_enum_count"] >= 3) |
        (sig["page_token_count"] >= 1) |
        (sig["yoyak_marker"] == 1) |
        (sig["reportative_end"] == 1)
    )

    no_ctx = ~sig.apply(_has_investigator_ctx, axis=1)
    R = (
        no_ctx &
        (sig["admin_verbs_count"] >= 2) &
        (sig["direct_addr_count"] >= 1) &
        (sig["first_person"] >= 1) &
        (sig["numbered_enum_count"] <= 1) &
        (sig["parenthetical_enum_count"] <= 1)
    )

    yhat = (scored["pred_label"] == "summarized").astype(int)
    prec_S = float(yhat[S].mean()) if S.any() else np.nan
    prec_R = float((1 - yhat[R]).mean()) if R.any() else np.nan

    return {
        "n_seeds_summarized": int(S.sum()),
        "n_seeds_raw": int(R.sum()),
        "proxy_precision_on_summarized_seeds": prec_S,
        "proxy_precision_on_raw_seeds": prec_R
    }

def weaklabel_consistency(scored):
    sig = _signals_df(scored)
    lf_labels = sig.apply(_lf_label, axis=1)
    mask = lf_labels != -1
    yhat = (scored["pred_label"] == "summarized").astype(int)
    agree = float((yhat[mask].to_numpy() == lf_labels[mask].to_numpy()).mean()) if mask.any() else np.nan
    return {"n_nonabstain": int(mask.sum()), "agreement_with_LFs": agree}

def stability_bootstrap_retrain(det, df, B=5, frac=0.8, text_col="NOTE_CONTENTDTL"):
    preds = []
    for b in range(B):
        sub = resample(df, replace=True, n_samples=int(len(df) * frac), random_state=42 + b)
        det_b = SummarizationDetector(
            text_col=det.text_col,
            min_rule_conf=det.min_rule_conf,
            lr_solver=det.lr_solver,
            lr_penalty=det.lr_penalty,
            lr_C=det.lr_C,
            lr_max_iter=det.lr_max_iter,
            lr_tol=det.lr_tol,
            calibrate_default=det.calibrate_default,
            use_length_feature=det.use_length_feature
        )
        det_b.fit(sub[[text_col]].rename(columns={text_col: det.text_col}))
        preds.append(det_b.predict(df[[text_col]].rename(columns={text_col: det.text_col})))

    preds = np.stack(preds, axis=1)
    maj = (preds.mean(axis=1) >= 0.5).astype(int)
    agree_rate = (preds == maj[:, None]).mean()
    unanimous = (np.all(preds == preds[:, [0]], axis=1)).mean()
    return {
        "bootstrap_models": B,
        "pairwise_agreement_to_majority": float(agree_rate),
        "fraction_unanimous": float(unanimous)
    }

def mixture_separation(scored):
    p = np.clip(scored["p_summarized"].to_numpy(), 1e-4, 1 - 1e-4)
    logit = np.log(p / (1 - p))[:, None]
    gm = GaussianMixture(n_components=2, random_state=0).fit(logit)
    resp = gm.predict_proba(logit)[:, gm.means_.argmax()]
    overlap = float(np.mean(np.minimum(resp, 1 - resp)))
    return {"mixture_overlap_(0=good,0.5=bad)": overlap}

def calibration_with_weak(scored):
    sig = _signals_df(scored)
    lf_labels = sig.apply(_lf_label, axis=1)
    keep = lf_labels != -1
    if not keep.any():
        return {"kept_for_calibration": 0, "brier": np.nan}
    y = lf_labels[keep].to_numpy()
    p = scored.loc[keep, "p_summarized"].to_numpy()
    brier = float(brier_score_loss(y, p))
    return {"kept_for_calibration": int(keep.sum()), "brier_vs_weak": brier}

def analyze_length_vs_summary(det, notes, text_col="NOTE_CONTENTDTL", n_bins=10, seed=42, make_plot=True):
    if getattr(det, "pipe", None) is None:
        det.fit(notes)

    scored = det.score_dataframe(notes)
    scored = scored.reset_index(drop=True)

    def _get_len(d):
        if isinstance(d, dict) and "length" in d:
            return int(d["length"])
        return len(str(d))

    scored["length"] = scored["rule_signals"].apply(_get_len).astype(int)

    pearson = scored[["length", "p_summarized"]].corr(method="pearson").iloc[0, 1]
    spearman = scored[["length", "p_summarized"]].corr(method="spearman").iloc[0, 1]

    rng = np.random.default_rng(seed)

    def _boot_corr(x, y, method="spearman", B=1000):
        n = len(x)
        vals = []
        for _ in range(B):
            idx = rng.integers(0, n, n)
            vals.append(pd.Series(x[idx]).corr(pd.Series(y[idx]), method=method))
        lo, hi = np.nanpercentile(vals, [2.5, 97.5])
        return float(np.nanmean(vals)), float(lo), float(hi)

    x = scored["length"].to_numpy()
    y = scored["p_summarized"].to_numpy()
    pearson_b, p_lo, p_hi = _boot_corr(x, y, method="pearson")
    spearman_b, s_lo, s_hi = _boot_corr(x, y, method="spearman")

    scored["len_bin"] = pd.qcut(scored["length"].rank(method="first"), q=n_bins, labels=False)
    bin_stats = scored.groupby("len_bin").agg(
        mean_p=("p_summarized", "mean"),
        n=("p_summarized", "size"),
        len_min=("length", "min"),
        len_max=("length", "max")
    ).reset_index(drop=True)

    if make_plot:
        plt.figure(figsize=(6, 4))
        plt.plot(np.arange(n_bins), bin_stats["mean_p"], marker="o")
        plt.xlabel("Length decile (short → long)")
        plt.ylabel("Mean P(summarized)")
        plt.title("P(summarized) vs. note length")
        plt.tight_layout()
        plt.show()

    short_mask = scored["length"] <= 15
    long_thresh = np.nanpercentile(scored["length"], 90)
    long_mask = scored["length"] >= long_thresh

    summary = {
        "pearson": float(pearson),
        "spearman": float(spearman),
        "pearson_bootstrap_CI": (p_lo, p_hi),
        "spearman_bootstrap_CI": (s_lo, s_hi),
        "meanP_short_len<=15": float(scored.loc[short_mask, "p_summarized"].mean()) if short_mask.any() else np.nan,
        "meanP_long_len>=p90": float(scored.loc[long_mask, "p_summarized"].mean()) if long_mask.any() else np.nan,
        "n_short": int(short_mask.sum()),
        "n_long": int(long_mask.sum()),
        "n_total": int(len(scored)),
    }

    return scored, summary, bin_stats

def run_all_stability_checks(det, df, text_col="NOTE_CONTENTDTL", threshold=0.5, run_retrain_bootstrap=True):
    print("\n[1] Scoring full dataset...")
    scored = det.score_dataframe(df, threshold=threshold)

    print("[2] Threshold sensitivity...")
    threshold_df = threshold_sensitivity(det, df, text_col=text_col)

    print("[3] Bootstrap class proportion stability...")
    boot_prop = bootstrap_class_proportion(det, df, text_col=text_col, B=200, frac=1.0, threshold=threshold)

    print("[4] Counterfactual cue-removal tests...")
    counter = counterfactual_tests(det, df[[text_col]].copy(), text_col=text_col)

    print("[5] Weak-label agreement / proxy precision...")
    silver = silver_metrics(scored, text_col=text_col)
    weak_cons = weaklabel_consistency(scored)

    print("[6] Mixture separation diagnostics...")
    mix = mixture_separation(scored)

    print("[7] Weak-label calibration diagnostics...")
    calib = calibration_with_weak(scored)

    print("[8] Length-bias diagnostics...")
    _, length_summary, length_bins = analyze_length_vs_summary(det, df[[text_col]].copy(), text_col=text_col, make_plot=True)

    retrain_boot = None
    if run_retrain_bootstrap:
        print("[9] Retrain bootstrap stability (slower)...")
        df_small = df[[text_col]].sample(min(8000, len(df)), random_state=7) if len(df) > 8000 else df[[text_col]].copy()
        retrain_boot = stability_bootstrap_retrain(det, df_small, B=5, frac=0.8, text_col=text_col)

    report = {}
    report.update(boot_prop)
    report.update(counter)
    report.update(silver)
    report.update(weak_cons)
    report.update(mix)
    report.update(calib)
    report.update(length_summary)
    if retrain_boot is not None:
        report.update(retrain_boot)

    report_df = pd.DataFrame([report])

    return {
        "scored": scored,
        "threshold_sensitivity": threshold_df,
        "length_bins": length_bins,
        "summary_report": report_df
    }

# =========================
# Optional gold-label threshold sweep
# =========================

def sweep_for_precision(y_true, p_raw, target_prec=0.95):
    thresholds = np.linspace(0.50, 0.99, 100)
    rows = []
    for t in thresholds:
        y_hat = (p_raw >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_hat, average='binary', zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
        rows.append((t, prec, rec, f1, tp, fp, fn, tn))
    df = pd.DataFrame(rows, columns=["threshold", "precision", "recall", "f1", "tp", "fp", "fn", "tn"])
    ok = df[df["precision"] >= target_prec]
    return (ok.iloc[0] if not ok.empty else df.iloc[df["precision"].idxmax()]), df

# =========================
# Main run block
# =========================
if __name__ == "__main__":
    # --------------------------------------
    # 1) Choose your dataframe
    # --------------------------------------
    # Example:
    # notes_df = allnotes.copy()
    # or
    # notes_df = kfsppororo.copy()

    notes_df = kfsppororo.copy()   # <-- edit this if needed
    text_col = "NOTE_CONTENTDTL"

    # Keep only rows with text
    notes_df = notes_df[~notes_df[text_col].isna()].copy().reset_index(drop=True)

    # --------------------------------------
    # 2) Fit detector
    # --------------------------------------
    det = SummarizationDetector(
        text_col=text_col,
        min_rule_conf=0.34,
        use_length_feature=False
    )
    det.fit(notes_df)

    # --------------------------------------
    # 3) Score full data
    # --------------------------------------
    scored = det.score_dataframe(notes_df, threshold=0.5)
    print(scored[[text_col, "pred_label", "p_summarized"]].head())

    summarized = scored[scored["pred_label"] == "summarized"].copy()
    raw = scored[scored["pred_label"] == "raw"].copy()

    summarizednotes = summarized[text_col]
    rawnotes = raw[text_col]

    print("\n=== Classification counts at threshold 0.50 ===")
    print(scored["pred_label"].value_counts(dropna=False))

    # --------------------------------------
    # 4) Run stability checks
    # --------------------------------------
    results = run_all_stability_checks(
        det,
        notes_df,
        text_col=text_col,
        threshold=0.5,
        run_retrain_bootstrap=True
    )

    stability_report = results["summary_report"]
    threshold_report = results["threshold_sensitivity"]
    length_bins = results["length_bins"]

    print("\n=== Stability summary report ===")
    print(stability_report.T)

    print("\n=== Threshold sensitivity ===")
    print(threshold_report)

    print("\n=== Length bins ===")
    print(length_bins)

    # --------------------------------------
    # 5) Save outputs
    # --------------------------------------
    scored.to_csv("scored_notes_with_probs.csv", index=False, encoding="utf-8-sig")
    stability_report.to_csv("stability_summary_report.csv", index=False, encoding="utf-8-sig")
    threshold_report.to_csv("threshold_sensitivity_report.csv", index=False, encoding="utf-8-sig")
    length_bins.to_csv("length_bias_bins.csv", index=False, encoding="utf-8-sig")

    print("\nSaved:")
    print("- scored_notes_with_probs.csv")
    print("- stability_summary_report.csv")
    print("- threshold_sensitivity_report.csv")
    print("- length_bias_bins.csv")
    
    
    #%%
# -*- coding: utf-8 -*-
"""
MULTI-SCHEME DOWNSTREAM ANALYSIS AFTER RAW-VS-SUMMARIZED DETECTOR

Run this AFTER:
    1. Sectioning pipeline
    2. KeyBERT keyword extraction/tokenization for all section columns
    3. KOTE sentiment extraction for all section columns
    4. Raw-vs-summarized detector, producing:
           scored = det.score_dataframe(notes_df, threshold=0.5)

This code analyzes all section schemes:
    - third
    - quart
    - third5
    - shuf_third

For each scheme, it runs:
    A. Theme presence extraction
    B. Theme age one-vs-rest logistic regression controlling for gender
    C. Theme gender logistic regression controlling for age
    D. Sentiment presence extraction
    E. Sentiment age one-vs-rest logistic regression controlling for gender
    F. Sentiment gender logistic regression controlling for age
    G. First-version line plots:
           x-axis = Age Group
           lines = Section
           panels = Theme/Sentiment
    H. Heatmaps
    I. Saves CSV outputs
"""

# ============================================================
# 0. IMPORTS
# ============================================================

import ast
import os
import re
import warnings

import numpy as np
import pandas as pd

from tqdm import tqdm

import statsmodels.api as sm

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

warnings.filterwarnings("ignore")
tqdm.pandas()


# ============================================================
# 1. GLOBAL OPTIONS
# ============================================================

SAVE_OUTPUTS = True
OUTPUT_DIR = r"C:\Users\Jae Bin Park\multi_scheme_outputs"

if SAVE_OUTPUTS:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. GLOBAL FIGURE FORMATTING
# ============================================================

FONT_CANDIDATES = [
    r"C:\Users\Jae Bin Park\AppData\Local\Microsoft\Windows\Fonts\Roboto-Bold.ttf",
    r"C:\Windows\Fonts\Arialbd.ttf",
    r"C:\Windows\Fonts\malgunbd.ttf",
    r"C:\Windows\Fonts\malgun.ttf",
]

font_prop = None
custom_font = None

for fp in FONT_CANDIDATES:
    try:
        if os.path.exists(fp):
            font_prop = fm.FontProperties(fname=fp)
            custom_font = font_prop.get_name()
            break
    except Exception:
        continue

if custom_font is not None:
    plt.rcParams["font.family"] = custom_font


FIG_STYLE = {
    "cmap": "coolwarm",
    "center": 0,
    "line_width": 2,
    "marker": "o",
    "zero_line_style": "--",
    "zero_line_color": "gray",
    "zero_line_width": 1,
    "heatmap_linewidths": 0.5,
    "heatmap_linecolor": "gray",
    "title_fontsize": 14,
    "axis_label_fontsize": 12,
    "tick_fontsize": 11,
    "annot_fontsize": 10,
    "star_fontsize": 11,
    "legend_fontsize": 11,
    "legend_title_fontsize": 12,
}

AGE_ORDER = [1, 2, 3, 4, 5]

AGE_LABELS = {
    1: "≤18",
    2: "19–34",
    3: "35–49",
    4: "50–64",
    5: "65+",
}


def set_constant_plot_style():
    sns.set(style="white", context="notebook", font_scale=1.2)

    plt.rcParams.update({
        "axes.linewidth": 1.2,
        "xtick.major.width": 1,
        "ytick.major.width": 1,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.dpi": 100,
    })

    if custom_font is not None:
        plt.rcParams["font.family"] = custom_font


def apply_axis_font(ax):
    if font_prop is None:
        return

    for label in ax.get_xticklabels():
        label.set_fontproperties(font_prop)
        label.set_fontsize(FIG_STYLE["tick_fontsize"])

    for label in ax.get_yticklabels():
        label.set_fontproperties(font_prop)
        label.set_fontsize(FIG_STYLE["tick_fontsize"])

    ax.xaxis.label.set_fontproperties(font_prop)
    ax.yaxis.label.set_fontproperties(font_prop)


def get_sig_star(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


def safe_log_or(x):
    x = pd.to_numeric(x, errors="coerce")
    x = np.clip(x, np.finfo(float).tiny, np.inf)
    return np.log(x)


# ============================================================
# 3. PARSING HELPERS
# ============================================================

def parse_listlike(x):
    """
    Converts Excel-saved stringified lists back into Python lists.

    Examples:
        "[('미안', 0.51), ('사랑', 0.44)]"
        "['문장1', '문장2']"
    """
    if isinstance(x, list):
        return x

    if pd.isna(x):
        return []

    if isinstance(x, str):
        x = x.strip()

        if x == "":
            return []

        if x.startswith("[") and x.endswith("]"):
            try:
                parsed = ast.literal_eval(x)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                return []

    return []


def parse_dictlike(x):
    """
    Converts Excel-saved stringified dictionaries back into Python dicts.

    Example:
        "{'labels': ['슬픔'], 'scores': [0.91]}"
    """
    if isinstance(x, dict):
        if "labels" not in x:
            x["labels"] = []
        if "scores" not in x:
            x["scores"] = []
        return x

    if pd.isna(x):
        return {"labels": [], "scores": []}

    if isinstance(x, str):
        x = x.strip()

        if x == "":
            return {"labels": [], "scores": []}

        if x.startswith("{") and x.endswith("}"):
            try:
                parsed = ast.literal_eval(x)
                if isinstance(parsed, dict):
                    if "labels" not in parsed:
                        parsed["labels"] = []
                    if "scores" not in parsed:
                        parsed["scores"] = []
                    return parsed
            except Exception:
                return {"labels": [], "scores": []}

    return {"labels": [], "scores": []}


def ensure_listn(x, n):
    """
    Makes sure a presence vector is an n-item list.
    """
    x = parse_listlike(x) if not isinstance(x, list) else x

    if not isinstance(x, list):
        return [0] * n

    x = list(x)

    if len(x) < n:
        x = x + [0] * (n - len(x))

    if len(x) > n:
        x = x[:n]

    return [int(v) if not pd.isna(v) else 0 for v in x]


def extract_tokens_from_keyword_list(section):
    """
    Expected forms:
        [('미안', 0.51), ('사랑', 0.44)]
        [['미안', 0.51], ['사랑', 0.44]]
        ['미안', '사랑']
    """
    tokens = []

    if not isinstance(section, list):
        return tokens

    for item in section:
        if isinstance(item, tuple) and len(item) >= 1:
            tokens.append(str(item[0]))
        elif isinstance(item, list) and len(item) >= 1:
            tokens.append(str(item[0]))
        elif isinstance(item, str):
            tokens.append(item)

    return tokens


# ============================================================
# 4. SECTION SCHEMES
# ============================================================

SECTION_SCHEMES = {
    "third": {
        "section_cols": ["third_1", "third_2", "third_3"],
        "tokenized_keyword_cols": [
            "tokenizedkluekeywordsentencetransformer_third_1",
            "tokenizedkluekeywordsentencetransformer_third_2",
            "tokenizedkluekeywordsentencetransformer_third_3",
        ],
        "sentiment_cols": [
            "third_1_sentiment",
            "third_2_sentiment",
            "third_3_sentiment",
        ],
        "section_labels": ["Section 1", "Section 2", "Section 3"],
        "section_display": {
            "Section 1": "Introduction",
            "Section 2": "Body",
            "Section 3": "Conclusion",
        },
        "min_sent_col": "third_total_n_sent",
        "min_sentences": 3,
    },

    "quart": {
        "section_cols": ["quart_1", "quart_2", "quart_3", "quart_4"],
        "tokenized_keyword_cols": [
            "tokenizedkluekeywordsentencetransformer_quart_1",
            "tokenizedkluekeywordsentencetransformer_quart_2",
            "tokenizedkluekeywordsentencetransformer_quart_3",
            "tokenizedkluekeywordsentencetransformer_quart_4",
        ],
        "sentiment_cols": [
            "quart_1_sentiment",
            "quart_2_sentiment",
            "quart_3_sentiment",
            "quart_4_sentiment",
        ],
        "section_labels": ["Section 1", "Section 2", "Section 3", "Section 4"],
        "section_display": {
            "Section 1": "Q1",
            "Section 2": "Q2",
            "Section 3": "Q3",
            "Section 4": "Q4",
        },
        "min_sent_col": "quart_total_n_sent",
        "min_sentences": 4,
    },

    "third5": {
        "section_cols": ["third5_1", "third5_2", "third5_3"],
        "tokenized_keyword_cols": [
            "tokenizedkluekeywordsentencetransformer_third5_1",
            "tokenizedkluekeywordsentencetransformer_third5_2",
            "tokenizedkluekeywordsentencetransformer_third5_3",
        ],
        "sentiment_cols": [
            "third5_1_sentiment",
            "third5_2_sentiment",
            "third5_3_sentiment",
        ],
        "section_labels": ["Section 1", "Section 2", "Section 3"],
        "section_display": {
            "Section 1": "Introduction",
            "Section 2": "Body",
            "Section 3": "Conclusion",
        },
        "min_sent_col": "third5_total_n_sent",
        "min_sentences": 5,
    },

    "shuf_third": {
        "section_cols": ["shuf_third_1", "shuf_third_2", "shuf_third_3"],
        "tokenized_keyword_cols": [
            "tokenizedkluekeywordsentencetransformer_shuf_third_1",
            "tokenizedkluekeywordsentencetransformer_shuf_third_2",
            "tokenizedkluekeywordsentencetransformer_shuf_third_3",
        ],
        "sentiment_cols": [
            "shuf_third_1_sentiment",
            "shuf_third_2_sentiment",
            "shuf_third_3_sentiment",
        ],
        "section_labels": ["Section 1", "Section 2", "Section 3"],
        "section_display": {
            "Section 1": "Shuffled 1",
            "Section 2": "Shuffled 2",
            "Section 3": "Shuffled 3",
        },
        "min_sent_col": "shuf_third_total_n_sent",
        "min_sentences": 3,
    },
}


# ============================================================
# 5. THEME AND SENTIMENT DICTIONARIES
# ============================================================

sorry = [
    "미안", "죄송", "용서", "잘못", "후회", "죄", "책임", "민폐", "반성"
]

loveandgratitude = [
    "사랑", "고맙", "감사", "고생", "수고", "보고싶", "애틋", "소중", "그리"
]

burdensome = [
    "버겁", "부담", "짐", "감당", "힘들", "무겁", "지치", "헷갈"
]

despair = [
    "포기", "좌절", "죽", "끝", "아무것", "헛되", "무의미", "잊히", "없어지", "그만두"
]

pmaffairs = [
    "부탁", "정리", "남기", "처리", "보험", "은행", "장례", "통장", "유서"
]


THEME_TOKEN_DICT = {
    "Sorry and Shame": sorry,
    "Love and Gratitude": loveandgratitude,
    "Burden": burdensome,
    "Despair": despair,
    "Post-mortem Affairs": pmaffairs,
}

THEMES = list(THEME_TOKEN_DICT.keys())


sentiment_label_map = {
    "Despair": "절망",
    "Defeat": "패배/자기혐오",
    "Exhaustion": "힘듦/지침",
    "Anxious": "불안/걱정",
    "Disappointment": "안타까움/실망",
    "Anger": "화남/분노",
    "Hatred": "증오/혐오",
    "Resentment": "어이없음",
    "Sadness": "슬픔",
    "Sorrow": "서러움",
    "Gratitude": "고마움",
    "Affection": "흐뭇함(귀여움/예쁨)",
    "Happiness": "행복",
    "Relief": "안심/신뢰",
    "Neutral": "없음",
}

SENTIMENTS = list(sentiment_label_map.keys())

LINEPLOT_SENTIMENTS = [
    "Defeat",
    "Exhaustion",
    "Neutral",
    "Sadness",
    "Happiness",
    "Disappointment",
]

import pandas as pd

scored= pd.read_excel(r'C:\Users\Jae Bin Park\kfsp_sectioned_ver2.xlsx')

#raw=scored[scored["pred_label"].isin(["raw"])]

#raw=pd.read_excel(r'C:\Users\Jae Bin Park\oldrawnotes.xlsx')


# ============================================================
# 6. PREPARE RAW NOTES FROM SCORED
# ============================================================

if "scored" not in globals():
    raise NameError(
        "`scored` is not in memory. "
        "Run the raw-vs-summarized detector first:\n"
        "    scored = det.score_dataframe(notes_df, threshold=0.5)"
    )

if "pred_label" not in scored.columns:
    raise KeyError("`scored` must contain `pred_label`.")

raw = scored[scored["pred_label"].eq("raw")].copy()
print("\n==============================")
print("RAW FILTERING")
print("==============================")
print("Total scored rows:", scored.shape[0])
print("Raw rows:", raw.shape[0])
print(raw["pred_label"].value_counts(dropna=False))

#raw=pd.read_excel(r'C:\Users\Jae Bin Park\oldrawnotes1.xlsx')

# ------------------------------------------------------------
# Define a general too_short marker if possible
# ------------------------------------------------------------
if "sentencesplit" in raw.columns:
    raw["sentencesplit"] = raw["sentencesplit"].apply(parse_listlike)
    raw["too_short"] = raw["sentencesplit"].apply(
        lambda x: len(x) < 3 if isinstance(x, list) else True
    )

elif "third_total_n_sent" in raw.columns:
    raw["too_short"] = raw["third_total_n_sent"].fillna(0).astype(int) < 3

else:
    raw["too_short"] = False

morethan3 = raw[raw["too_short"].eq(False)].copy()

print("Raw rows with >=3 sentences using general filter:", morethan3.shape[0])


# ============================================================
# 7. CHECK MISSING COLUMNS FOR ALL SCHEMES
# ============================================================

def check_scheme_columns(df, section_schemes):
    rows = []

    for scheme_name, scheme_config in section_schemes.items():
        for group_name in ["section_cols", "tokenized_keyword_cols", "sentiment_cols"]:
            for col in scheme_config[group_name]:
                rows.append({
                    "scheme": scheme_name,
                    "column_group": group_name,
                    "column": col,
                    "exists": col in df.columns,
                })

    out = pd.DataFrame(rows)
    return out


scheme_column_check = check_scheme_columns(raw, SECTION_SCHEMES)

print("\n==============================")
print("SCHEME COLUMN CHECK")
print("==============================")
print(scheme_column_check.groupby(["scheme", "column_group"])["exists"].mean())

missing_scheme_cols = scheme_column_check[scheme_column_check["exists"].eq(False)]

if not missing_scheme_cols.empty:
    print("\nWARNING: Some scheme columns are missing.")
    print(missing_scheme_cols.to_string(index=False))
    print(
        "\nIf sentiment columns are missing for quart/third5/shuf_third, "
        "rerun KOTE sentiment extraction for all section columns."
    )


# ============================================================
# 8. FEATURE BUILDERS
# ============================================================

def filter_scheme_df(df, scheme_name, scheme_config):
    """
    Applies scheme-specific minimum sentence filter.
    """
    out = df.copy()

    min_sent_col = scheme_config.get("min_sent_col")
    min_sentences = scheme_config.get("min_sentences")

    if min_sent_col in out.columns:
        out = out[out[min_sent_col].fillna(0).astype(int) >= min_sentences].copy()

    return out


def build_theme_features_for_scheme(df, scheme_name, scheme_config):
    """
    Builds theme presence features for one section scheme.

    Feature names:
        third Section 1 Sorry and Shame
        quart Section 4 Despair
        third5 Section 2 Burden
    """
    out = df.copy()

    keyword_cols = scheme_config["tokenized_keyword_cols"]
    section_labels = scheme_config["section_labels"]
    n_sections = len(section_labels)

    missing = [c for c in keyword_cols if c not in out.columns]
    if missing:
        raise KeyError(f"[{scheme_name}] Missing keyword columns:\n{missing}")

    for col in keyword_cols:
        out[col] = out[col].apply(parse_listlike)

    def create_presence_vector(row, target_tokens):
        presence = []

        for col in keyword_cols:
            tokens = extract_tokens_from_keyword_list(row[col])

            has_token = any(
                any(target in token for token in tokens)
                for target in target_tokens
            )

            presence.append(int(has_token))

        return presence

    for theme, target_tokens in THEME_TOKEN_DICT.items():
        presence_col = f"{scheme_name}_{theme}_presence"

        tqdm.pandas(desc=f"{scheme_name}: {theme}")

        out[presence_col] = out.progress_apply(
            lambda row: create_presence_vector(row, target_tokens),
            axis=1
        )

        out[presence_col] = out[presence_col].apply(
            lambda x: ensure_listn(x, n_sections)
        )

    feature_names = []

    for sec_idx, section_label in enumerate(section_labels):
        for theme in THEMES:
            feature_col = f"{scheme_name} {section_label} {theme}"
            presence_col = f"{scheme_name}_{theme}_presence"

            out[feature_col] = out[presence_col].apply(
                lambda v: ensure_listn(v, n_sections)[sec_idx]
            )

            feature_names.append(feature_col)

    return out, feature_names


def build_sentiment_features_for_scheme(df, scheme_name, scheme_config):
    """
    Builds sentiment presence features for one section scheme.

    Feature names:
        third Section 1 Sadness
        quart Section 4 Defeat
        third5 Section 2 Neutral
    """
    out = df.copy()

    sentiment_cols = scheme_config["sentiment_cols"]
    section_labels = scheme_config["section_labels"]
    n_sections = len(section_labels)

    missing = [c for c in sentiment_cols if c not in out.columns]
    if missing:
        raise KeyError(
            f"[{scheme_name}] Missing sentiment columns:\n{missing}\n"
            "You probably need to run KOTE sentiment prediction for all section columns."
        )

    for col in sentiment_cols:
        out[col] = out[col].apply(parse_dictlike)

    def create_sentiment_presence(row, korean_label):
        presence = []

        for col in sentiment_cols:
            entry = row[col]

            if isinstance(entry, dict):
                section_labels_found = entry.get("labels", [])
                has_label = int(korean_label in section_labels_found)
            else:
                has_label = 0

            presence.append(has_label)

        return presence

    for english_label, korean_label in sentiment_label_map.items():
        presence_col = f"{scheme_name}_{english_label}_presence"

        tqdm.pandas(desc=f"{scheme_name}: {english_label}")

        out[presence_col] = out.progress_apply(
            lambda row: create_sentiment_presence(row, korean_label),
            axis=1
        )

        out[presence_col] = out[presence_col].apply(
            lambda x: ensure_listn(x, n_sections)
        )

    feature_names = []

    for sec_idx, section_label in enumerate(section_labels):
        for sentiment in SENTIMENTS:
            feature_col = f"{scheme_name} {section_label} {sentiment}"
            presence_col = f"{scheme_name}_{sentiment}_presence"

            out[feature_col] = out[presence_col].apply(
                lambda v: ensure_listn(v, n_sections)[sec_idx]
            )

            feature_names.append(feature_col)

    return out, feature_names


# ============================================================
# 9. MODEL HELPERS
# ============================================================

def fit_age_ovr_logit(
    df,
    feature_names,
    outcome_col="AGE2",
    sex_col="SEX",
    gender_covariate_name="Gender",
    maxiter=200
):
    """
    One-vs-rest logistic regression:
        AGE2 group membership ~ features + Gender

    Gender coding:
        Male=1, Female=0
    """
    work = df[
        df[outcome_col].isin([1, 2, 3, 4, 5])
        & df[sex_col].isin([1, 2])
    ].copy()

    work[gender_covariate_name] = work[sex_col].replace({1: 1, 2: 0}).astype(int)

    X = work[feature_names + [gender_covariate_name]].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = sm.add_constant(X, has_constant="add")
    X = X.astype(float)

    group_results = {}

    for group in AGE_ORDER:
        y = (work[outcome_col] == group).astype(int)

        mask = ~(X.isna().any(axis=1) | y.isna())
        Xi = X.loc[mask].copy()
        yi = y.loc[mask].copy()

        try:
            model = sm.Logit(yi, Xi)
            result = model.fit(disp=False, maxiter=maxiter)

            odds_ratios = np.exp(result.params)
            conf = np.exp(result.conf_int())
            conf.columns = ["CI Lower", "CI Upper"]

            summary_df = pd.DataFrame({
                "Feature": result.params.index,
                "Coefficient": result.params.values,
                "Odds Ratio": odds_ratios.values,
                "p-value": result.pvalues.values,
                "CI Lower": conf["CI Lower"].values,
                "CI Upper": conf["CI Upper"].values,
                "Age Group": group,
                "n_model": len(yi),
            })

        except Exception as e:
            print(f"[WARN] Age model failed for group {group}: {e}")

            summary_df = pd.DataFrame({
                "Feature": feature_names,
                "Coefficient": np.nan,
                "Odds Ratio": np.nan,
                "p-value": np.nan,
                "CI Lower": np.nan,
                "CI Upper": np.nan,
                "Age Group": group,
                "n_model": len(yi),
            })

        group_results[group] = summary_df

    combined = pd.concat(group_results.values(), ignore_index=True)

    combined = combined[combined["Feature"].isin(feature_names)].copy()
    combined["log_odds_ratio"] = safe_log_or(combined["Odds Ratio"])
    combined["sig_star"] = combined["p-value"].apply(get_sig_star)
    combined["annot"] = (
        combined["log_odds_ratio"].round(2).astype(str)
        + combined["sig_star"]
    )
    combined["Demographic"] = combined["Age Group"].apply(lambda x: f"Age{x}")

    return combined


def fit_gender_logit_controlling_age(
    df,
    feature_names,
    sex_col="SEX",
    age_col="AGE2",
    maxiter=200
):
    """
    Logistic regression:
        SEX_BINARY ~ features + AGE2 categorical dummies

    Outcome:
        Male=1, Female=0

    AGE2 reference group:
        AGE2 == 1
    """
    work = df[
        df[sex_col].isin([1, 2])
        & df[age_col].isin([1, 2, 3, 4, 5])
    ].copy()

    work["SEX_BINARY"] = work[sex_col].replace({1: 1, 2: 0}).astype(int)

    age_dummies = pd.get_dummies(
        work[age_col],
        prefix="AGE2",
        drop_first=True,
        dtype=int
    )

    for col in ["AGE2_2", "AGE2_3", "AGE2_4", "AGE2_5"]:
        if col not in age_dummies.columns:
            age_dummies[col] = 0

    age_dummies = age_dummies[["AGE2_2", "AGE2_3", "AGE2_4", "AGE2_5"]]

    X = pd.concat(
        [
            work[feature_names],
            age_dummies
        ],
        axis=1
    )

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = sm.add_constant(X, has_constant="add")
    X = X.astype(float)

    y = work["SEX_BINARY"].astype(float)

    mask = ~(X.isna().any(axis=1) | y.isna())
    X_clean = X.loc[mask].copy()
    y_clean = y.loc[mask].copy()

    print("Gender model diagnostics:")
    print("  X_clean shape:", X_clean.shape)
    print("  y_clean shape:", y_clean.shape)
    print("  Class balance:", y_clean.value_counts().to_dict())

    try:
        model = sm.Logit(y_clean, X_clean)
        result = model.fit(disp=False, maxiter=maxiter)

        odds_ratios = np.exp(result.params)
        conf = np.exp(result.conf_int())
        conf.columns = ["CI Lower", "CI Upper"]

        summary_df = pd.DataFrame({
            "Feature": result.params.index,
            "Coefficient": result.params.values,
            "Odds Ratio": odds_ratios.values,
            "p-value": result.pvalues.values,
            "CI Lower": conf["CI Lower"].values,
            "CI Upper": conf["CI Upper"].values,
            "n_model": len(y_clean),
        })

    except Exception as e:
        print(f"[WARN] Gender model failed: {e}")

        result = None

        summary_df = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": np.nan,
            "Odds Ratio": np.nan,
            "p-value": np.nan,
            "CI Lower": np.nan,
            "CI Upper": np.nan,
            "n_model": len(y_clean),
        })

    summary_df = summary_df[summary_df["Feature"].isin(feature_names)].copy()
    summary_df["Demographic"] = "Gender (Male)"
    summary_df["log_odds_ratio"] = safe_log_or(summary_df["Odds Ratio"])
    summary_df["sig_star"] = summary_df["p-value"].apply(get_sig_star)
    summary_df["annot"] = (
        summary_df["log_odds_ratio"].round(2).astype(str)
        + summary_df["sig_star"]
    )

    return summary_df, result


def split_scheme_section_and_label(df, label_col_name):
    """
    Splits feature names like:
        third Section 1 Sorry and Shame
        quart Section 4 Despair

    into:
        scheme
        section
        theme/sentiment
    """
    out = df.copy()

    extracted = out["Feature"].str.extract(
        r"^(?P<scheme>\S+)\s+(?P<section>Section\s+\d+)\s+(?P<label>.+)$"
    )

    out["scheme"] = extracted["scheme"]
    out["section"] = extracted["section"]
    out[label_col_name] = extracted["label"]

    return out


# ============================================================
# 10. PLOT HELPERS
# ============================================================

def plot_age_heatmaps_by_feature(
    combined_df,
    label_col,
    label_order,
    section_order,
    section_display,
    title_prefix,
    figsize=(8, 5)
):
    set_constant_plot_style()

    for group in sorted(combined_df["Age Group"].dropna().unique()):
        sub_df = combined_df[combined_df["Age Group"] == group].copy()

        heatmap_data = sub_df.pivot(
            index=label_col,
            columns="section",
            values="log_odds_ratio"
        ).reindex(index=label_order, columns=section_order)

        annot_data = sub_df.pivot(
            index=label_col,
            columns="section",
            values="annot"
        ).reindex(index=label_order, columns=section_order)

        plt.figure(figsize=figsize)

        ax = sns.heatmap(
            heatmap_data,
            annot=annot_data,
            fmt="",
            center=FIG_STYLE["center"],
            cmap=FIG_STYLE["cmap"],
            linewidths=FIG_STYLE["heatmap_linewidths"],
            linecolor=FIG_STYLE["heatmap_linecolor"],
            cbar_kws={"label": "Log-Odds"}
        )

        for text in ax.texts:
            text.set_fontsize(FIG_STYLE["annot_fontsize"])
            if font_prop is not None:
                text.set_fontproperties(font_prop)

        ax.set_title(
            f"{title_prefix} by Age Group {group}",
            fontsize=FIG_STYLE["title_fontsize"],
            fontproperties=font_prop
        )
        ax.set_ylabel(label_col.capitalize(), fontsize=FIG_STYLE["axis_label_fontsize"])
        ax.set_xlabel("Section", fontsize=FIG_STYLE["axis_label_fontsize"])

        current_xticks = [x.get_text() for x in ax.get_xticklabels()]
        ax.set_xticklabels(
            [section_display.get(x, x) for x in current_xticks],
            rotation=0
        )

        apply_axis_font(ax)

        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel("Log-Odds", fontsize=FIG_STYLE["axis_label_fontsize"])
        if font_prop is not None:
            cbar.ax.yaxis.label.set_fontproperties(font_prop)

        plt.tight_layout()
        plt.show()


def plot_first_version_lineplot_by_age(
    combined_df,
    label_col,
    selected_labels,
    section_order,
    section_display,
    title,
    n_rows=2,
    n_cols=3,
    figsize=(15, 8)
):
    """
    FIRST-VERSION LINE PLOT.

    This matches the original structure:
        - one panel per theme/sentiment
        - x-axis = Age Group
        - lines = Section
        - y-axis = log_odds_ratio
    """
    set_constant_plot_style()

    line_df = combined_df.copy()

    line_df = line_df[line_df[label_col].isin(selected_labels)].copy()
    line_df = line_df[line_df["section"].isin(section_order)].copy()

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharey=True
    )

    axes = axes.flatten()
    fig.subplots_adjust(right=0.82)

    for idx, label in enumerate(selected_labels):
        if idx >= len(axes):
            break

        ax = axes[idx]

        label_data = line_df[line_df[label_col] == label].copy()

        for section in section_order:
            line_data = (
                label_data[label_data["section"] == section]
                .sort_values("Age Group")
            )

            if line_data.empty:
                continue

            ax.plot(
                line_data["Age Group"],
                line_data["log_odds_ratio"],
                marker="o",
                label=section_display.get(section, section),
                linewidth=2
            )

            for _, row_ in line_data.iterrows():
                x = row_["Age Group"]
                y = row_["log_odds_ratio"]
                star = get_sig_star(row_["p-value"])

                if star and pd.notna(y):
                    ax.text(
                        x,
                        y,
                        star,
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        weight="bold"
                    )

        ax.set_title(
            f"{label}",
            fontsize=13,
            fontproperties=font_prop
        )

        tick_positions = [1, 2, 3, 4, 5]
        tick_labels = ["≤18", "19–34", "35–49", "50–64", "65+"]

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            tick_labels,
            fontsize=10,
            fontproperties=font_prop
        )

        ax.axhline(
            0,
            linestyle="--",
            color="gray",
            linewidth=1
        )

        ax.set_xlabel(
            "Age Group",
            fontsize=11,
            fontproperties=font_prop
        )

        ax.tick_params(bottom=True, left=True)

        if idx % n_cols == 0:
            ax.set_ylabel(
                "Log-Odds",
                fontsize=11,
                fontproperties=font_prop
            )

        if font_prop is not None:
            for tick_label in ax.get_yticklabels():
                tick_label.set_fontproperties(font_prop)
                tick_label.set_fontsize(12)

            for tick_label in ax.get_xticklabels():
                tick_label.set_fontproperties(font_prop)
                tick_label.set_fontsize(12)

    # Remove extra subplot if present
    for idx in range(len(selected_labels), len(axes)):
        fig.delaxes(axes[idx])

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()

    if handles:
        fig.legend(
            handles,
            labels,
            title="Section",
            loc="center left",
            bbox_to_anchor=(0.87, 0.5),
            frameon=False,
            fontsize=11,
            title_fontsize=12
        )

    plt.suptitle(
        title,
        fontsize=14,
        weight="bold",
        y=0.95,
        fontproperties=font_prop
    )

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.show()

    plt.rcdefaults()


def plot_combined_demographic_heatmap(
    age_df,
    gender_df,
    feature_order,
    row_label_kind,
    title,
    figsize=(12, 10),
    include_ci=True,
    section_display_map=None
):
    """
    Combined heatmap:
        columns = Age1..Age5 + Gender(Male)
        rows = Section x Theme/Sentiment features

    If include_ci=True:
        first line = log OR + stars
        second line = 95% CI on log scale
    """
    set_constant_plot_style()

    if section_display_map is None:
        section_display_map = {
            "Section 1": "Opening",
            "Section 2": "Middle",
            "Section 3": "Ending",
            "Section 4": "Q4",
        }

    common_cols = [
        "Feature",
        "Coefficient",
        "Odds Ratio",
        "p-value",
        "CI Lower",
        "CI Upper",
        "Demographic",
        "log_odds_ratio",
        "sig_star",
        "annot",
    ]

    age_tbl = age_df.copy()
    gender_tbl = gender_df.copy()

    if "Demographic" not in age_tbl.columns:
        age_tbl["Demographic"] = age_tbl["Age Group"].apply(lambda x: f"Age{x}")

    if "Demographic" not in gender_tbl.columns:
        gender_tbl["Demographic"] = "Gender (Male)"

    for df_ in [age_tbl, gender_tbl]:
        if "annot" not in df_.columns:
            df_["annot"] = df_["log_odds_ratio"].round(2).astype(str) + df_["sig_star"]

    age_tbl = age_tbl[[c for c in common_cols if c in age_tbl.columns]].copy()
    gender_tbl = gender_tbl[[c for c in common_cols if c in gender_tbl.columns]].copy()

    merged_df = pd.concat([age_tbl, gender_tbl], ignore_index=True)

    demographic_order = [f"Age{i}" for i in AGE_ORDER] + ["Gender (Male)"]
    xtick_labels = [AGE_LABELS[i] for i in AGE_ORDER] + ["Gender (Male)"]

    heatmap_data = pd.pivot_table(
        merged_df,
        index="Feature",
        columns="Demographic",
        values="log_odds_ratio",
        aggfunc="first"
    ).reindex(index=feature_order, columns=demographic_order)

    if include_ci:
        eps = np.finfo(float).tiny

        merged_df["CI Lower"] = pd.to_numeric(
            merged_df["CI Lower"],
            errors="coerce"
        ).clip(lower=eps)

        merged_df["CI Upper"] = pd.to_numeric(
            merged_df["CI Upper"],
            errors="coerce"
        ).clip(lower=eps)

        merged_df["ci_lo_log"] = np.log(merged_df["CI Lower"])
        merged_df["ci_hi_log"] = np.log(merged_df["CI Upper"])

        merged_df["annot_main"] = (
            merged_df["log_odds_ratio"].round(2).astype(str)
            + merged_df["sig_star"].astype(str)
        )

        merged_df["annot_ci_only"] = (
            "["
            + merged_df["ci_lo_log"].round(2).astype(str)
            + ", "
            + merged_df["ci_hi_log"].round(2).astype(str)
            + "]"
        )

        annot_main_tbl = pd.pivot_table(
            merged_df,
            index="Feature",
            columns="Demographic",
            values="annot_main",
            aggfunc="first"
        ).reindex(index=feature_order, columns=demographic_order)

        annot_ci_tbl = pd.pivot_table(
            merged_df,
            index="Feature",
            columns="Demographic",
            values="annot_ci_only",
            aggfunc="first"
        ).reindex(index=feature_order, columns=demographic_order)

    else:
        annot_tbl = pd.pivot_table(
            merged_df,
            index="Feature",
            columns="Demographic",
            values="annot",
            aggfunc="first"
        ).reindex(index=feature_order, columns=demographic_order)

    plt.figure(figsize=figsize)

    ax = sns.heatmap(
        heatmap_data,
        annot=False if include_ci else annot_tbl,
        fmt="",
        cmap=FIG_STYLE["cmap"],
        center=FIG_STYLE["center"],
        linewidths=FIG_STYLE["heatmap_linewidths"],
        linecolor=FIG_STYLE["heatmap_linecolor"],
        cbar_kws={"label": "Log-Odds"}
    )

    if include_ci:
        mesh = ax.collections[0]

        def pick_text_color_from_value(val, mappable, thresh=0.53):
            rgba = mappable.cmap(mappable.norm(val if pd.notna(val) else 0.0))
            r, g, b, _ = rgba
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            return "black" if lum > thresh else "white"

        for i, row_key in enumerate(heatmap_data.index):
            for j, col_key in enumerate(heatmap_data.columns):
                val = heatmap_data.loc[row_key, col_key]
                main_txt = annot_main_tbl.loc[row_key, col_key]
                ci_txt = annot_ci_tbl.loc[row_key, col_key]

                if pd.isna(main_txt) and pd.isna(ci_txt):
                    continue

                color = pick_text_color_from_value(val, mesh)

                if pd.notna(main_txt):
                    ax.text(
                        j + 0.5,
                        i + 0.42,
                        str(main_txt),
                        ha="center",
                        va="center",
                        fontsize=11,
                        color=color,
                        fontproperties=font_prop
                    )

                if pd.notna(ci_txt):
                    ax.text(
                        j + 0.5,
                        i + 0.68,
                        str(ci_txt),
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=color,
                        fontproperties=font_prop
                    )

    else:
        for text in ax.texts:
            text.set_fontsize(FIG_STYLE["annot_fontsize"])
            if font_prop is not None:
                text.set_fontproperties(font_prop)

    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel("Log-Odds", fontsize=FIG_STYLE["axis_label_fontsize"])
    if font_prop is not None:
        cbar.ax.yaxis.label.set_fontproperties(font_prop)

    ax.set_xticks(np.arange(len(demographic_order)) + 0.5)
    ax.set_xticklabels(
        xtick_labels,
        rotation=0,
        fontsize=FIG_STYLE["tick_fontsize"],
        fontproperties=font_prop
    )

    ytick_labels = [
        re.sub(r"^\S+\s+Section\s+\d+\s+", "", str(lbl))
        for lbl in heatmap_data.index
    ]

    ax.set_yticklabels(
        ytick_labels,
        rotation=0,
        fontsize=FIG_STYLE["tick_fontsize"],
        fontproperties=font_prop
    )

    ax.set_title(
        title,
        fontsize=FIG_STYLE["title_fontsize"],
        fontproperties=font_prop
    )

    ax.set_xlabel(
        "Demographic",
        fontsize=FIG_STYLE["axis_label_fontsize"],
        fontproperties=font_prop
    )

    ax.set_ylabel(
        row_label_kind,
        fontsize=FIG_STYLE["axis_label_fontsize"],
        fontproperties=font_prop
    )

    ax.yaxis.set_label_coords(-0.35, 0.5)

    x_section_label = -1.4 if len(feature_order) > 20 else -1.2
    row_labels = list(heatmap_data.index)

    for section_prefix, display_name in section_display_map.items():
        rows_for_section = [
            i for i, lbl in enumerate(row_labels)
            if re.search(fr"\b{re.escape(section_prefix)}\b", str(lbl))
        ]

        if not rows_for_section:
            continue

        y_middle = (rows_for_section[0] + rows_for_section[-1]) / 2.0

        ax.text(
            x=x_section_label,
            y=y_middle,
            s=display_name,
            va="center",
            ha="center",
            rotation=90,
            fontsize=14,
            weight="bold",
            color="black",
            transform=ax.transData,
            clip_on=False,
            fontproperties=font_prop
        )

    ax.tick_params(bottom=True, left=True)

    plt.tight_layout()
    plt.show()


# ============================================================
# 11. RUN THEME ANALYSES FOR ALL SECTION SCHEMES
# ============================================================

theme_scheme_outputs = {}

for scheme_name, scheme_config in SECTION_SCHEMES.items():
    print("\n" + "=" * 80)
    print(f"THEME ANALYSIS FOR SECTION SCHEME: {scheme_name}")
    print("=" * 80)

    try:
        scheme_df = filter_scheme_df(morethan3, scheme_name, scheme_config)

        print(f"[{scheme_name}] N after scheme-specific sentence filter:", len(scheme_df))

        scheme_df, scheme_theme_features = build_theme_features_for_scheme(
            scheme_df,
            scheme_name,
            scheme_config
        )

        scheme_theme_age = fit_age_ovr_logit(
            scheme_df,
            feature_names=scheme_theme_features,
            outcome_col="AGE2",
            sex_col="SEX"
        )

        scheme_theme_age = split_scheme_section_and_label(
            scheme_theme_age,
            label_col_name="theme"
        )

        scheme_theme_gender, scheme_theme_gender_model = fit_gender_logit_controlling_age(
            scheme_df,
            feature_names=scheme_theme_features,
            sex_col="SEX",
            age_col="AGE2"
        )

        scheme_theme_gender = split_scheme_section_and_label(
            scheme_theme_gender,
            label_col_name="theme"
        )

        theme_scheme_outputs[scheme_name] = {
            "df": scheme_df,
            "feature_names": scheme_theme_features,
            "age_results": scheme_theme_age,
            "gender_results": scheme_theme_gender,
            "gender_model": scheme_theme_gender_model,
        }

        if SAVE_OUTPUTS:
            scheme_theme_age.to_csv(
                os.path.join(OUTPUT_DIR, f"{scheme_name}_theme_age_ovr_results.csv"),
                index=False,
                encoding="utf-8-sig"
            )

            scheme_theme_gender.to_csv(
                os.path.join(OUTPUT_DIR, f"{scheme_name}_theme_gender_results.csv"),
                index=False,
                encoding="utf-8-sig"
            )

            scheme_df.to_csv(
                os.path.join(OUTPUT_DIR, f"{scheme_name}_theme_features_df.csv"),
                index=False,
                encoding="utf-8-sig"
            )

    except Exception as e:
        print(f"[ERROR] Theme analysis failed for scheme {scheme_name}: {e}")


# ============================================================
# 12. RUN SENTIMENT ANALYSES FOR ALL SECTION SCHEMES
# ============================================================

sentiment_scheme_outputs = {}

for scheme_name, scheme_config in SECTION_SCHEMES.items():
    print("\n" + "=" * 80)
    print(f"SENTIMENT ANALYSIS FOR SECTION SCHEME: {scheme_name}")
    print("=" * 80)

    try:
        scheme_df = filter_scheme_df(morethan3, scheme_name, scheme_config)

        print(f"[{scheme_name}] N after scheme-specific sentence filter:", len(scheme_df))

        scheme_df, scheme_sentiment_features = build_sentiment_features_for_scheme(
            scheme_df,
            scheme_name,
            scheme_config
        )

        scheme_sentiment_age = fit_age_ovr_logit(
            scheme_df,
            feature_names=scheme_sentiment_features,
            outcome_col="AGE2",
            sex_col="SEX"
        )

        scheme_sentiment_age = split_scheme_section_and_label(
            scheme_sentiment_age,
            label_col_name="sentiment"
        )

        scheme_sentiment_gender, scheme_sentiment_gender_model = fit_gender_logit_controlling_age(
            scheme_df,
            feature_names=scheme_sentiment_features,
            sex_col="SEX",
            age_col="AGE2"
        )

        scheme_sentiment_gender = split_scheme_section_and_label(
            scheme_sentiment_gender,
            label_col_name="sentiment"
        )

        sentiment_scheme_outputs[scheme_name] = {
            "df": scheme_df,
            "feature_names": scheme_sentiment_features,
            "age_results": scheme_sentiment_age,
            "gender_results": scheme_sentiment_gender,
            "gender_model": scheme_sentiment_gender_model,
        }

        if SAVE_OUTPUTS:
            scheme_sentiment_age.to_csv(
                os.path.join(OUTPUT_DIR, f"{scheme_name}_sentiment_age_ovr_results.csv"),
                index=False,
                encoding="utf-8-sig"
            )

            scheme_sentiment_gender.to_csv(
                os.path.join(OUTPUT_DIR, f"{scheme_name}_sentiment_gender_results.csv"),
                index=False,
                encoding="utf-8-sig"
            )

            scheme_df.to_csv(
                os.path.join(OUTPUT_DIR, f"{scheme_name}_sentiment_features_df.csv"),
                index=False,
                encoding="utf-8-sig"
            )

    except Exception as e:
        print(f"[ERROR] Sentiment analysis failed for scheme {scheme_name}: {e}")


# ============================================================
# 13. PLOT ALL THEME RESULTS FOR ALL SECTION SCHEMES
#     FIRST LINE-PLOT VERSION ONLY
# ============================================================

for scheme_name, output in theme_scheme_outputs.items():
    print("\n" + "=" * 80)
    print(f"PLOTTING THEME RESULTS FOR: {scheme_name}")
    print("=" * 80)

    scheme_config = SECTION_SCHEMES[scheme_name]
    section_order = scheme_config["section_labels"]
    section_display = scheme_config["section_display"]

    theme_age_df = output["age_results"]
    theme_gender_df = output["gender_results"]
    theme_feature_names = output["feature_names"]

    # --------------------------------------------------------
    # Age heatmaps by section and theme
    # --------------------------------------------------------
    plot_age_heatmaps_by_feature(
        combined_df=theme_age_df,
        label_col="theme",
        label_order=THEMES,
        section_order=section_order,
        section_display=section_display,
        title_prefix=f"{scheme_name}: Theme × Section Log-Odds",
        figsize=(max(8, len(section_order) * 2.2), 5)
    )

    # --------------------------------------------------------
    # FIRST-VERSION LINE PLOT ONLY:
    # x-axis = Age Group
    # lines = Section
    # panels = Theme
    # --------------------------------------------------------
    plot_first_version_lineplot_by_age(
        combined_df=theme_age_df,
        label_col="theme",
        selected_labels=THEMES,
        section_order=section_order,
        section_display=section_display,
        title=(
            f"{scheme_name}: Theme-Section Effects on Age Group Membership\n"
            "(Controlling for Gender)"
        ),
        n_rows=2,
        n_cols=3,
        figsize=(15, 8)
    )

    # --------------------------------------------------------
    # Combined age + gender heatmap
    # --------------------------------------------------------
    plot_combined_demographic_heatmap(
        age_df=theme_age_df,
        gender_df=theme_gender_df,
        feature_order=theme_feature_names,
        row_label_kind="Thematic Feature",
        title=(
            f"{scheme_name}: Log(Odds Ratio) by Thematic Feature across Demographics\n"
            "Top: log(OR) with stars, Bottom: 95% CI on log scale"
        ),
        figsize=(12, max(10, 0.45 * len(theme_feature_names))),
        include_ci=True,
        section_display_map=section_display
    )


# ============================================================
# 14. PLOT ALL SENTIMENT RESULTS FOR ALL SECTION SCHEMES
#     FIRST LINE-PLOT VERSION ONLY
# ============================================================

for scheme_name, output in sentiment_scheme_outputs.items():
    print("\n" + "=" * 80)
    print(f"PLOTTING SENTIMENT RESULTS FOR: {scheme_name}")
    print("=" * 80)

    scheme_config = SECTION_SCHEMES[scheme_name]
    section_order = scheme_config["section_labels"]
    section_display = scheme_config["section_display"]

    sentiment_age_df = output["age_results"]
    sentiment_gender_df = output["gender_results"]
    sentiment_feature_names = output["feature_names"]

    # --------------------------------------------------------
    # Sentiment heatmaps by age group
    # --------------------------------------------------------
    plot_age_heatmaps_by_feature(
        combined_df=sentiment_age_df,
        label_col="sentiment",
        label_order=SENTIMENTS,
        section_order=section_order,
        section_display=section_display,
        title_prefix=f"{scheme_name}: Sentiment × Section Log-Odds",
        figsize=(max(10, len(section_order) * 2.6), 6)
    )

    # --------------------------------------------------------
    # FIRST-VERSION LINE PLOT ONLY:
    # x-axis = Age Group
    # lines = Section
    # panels = Sentiment
    # --------------------------------------------------------
    plot_first_version_lineplot_by_age(
        combined_df=sentiment_age_df,
        label_col="sentiment",
        selected_labels=LINEPLOT_SENTIMENTS,
        section_order=section_order,
        section_display=section_display,
        title=(
            f"{scheme_name}: Sentiment-Section Effects on Age Group Membership\n"
            "(Controlling for Gender)"
        ),
        n_rows=2,
        n_cols=3,
        figsize=(15, 8)
    )

    # --------------------------------------------------------
    # Combined age + gender sentiment heatmap
    # --------------------------------------------------------
    plot_combined_demographic_heatmap(
        age_df=sentiment_age_df,
        gender_df=sentiment_gender_df,
        feature_order=sentiment_feature_names,
        row_label_kind="Sentiment Feature",
        title=(
            f"{scheme_name}: Log(Odds Ratio) by Sentiment Feature across Demographics\n"
            "Top: log(OR) with stars, Bottom: 95% CI on log scale"
        ),
        figsize=(12, max(14, 0.38 * len(sentiment_feature_names))),
        include_ci=True,
        section_display_map=section_display
    )


# ============================================================
# 15. COMBINE AND SAVE ALL RESULTS
# ============================================================

all_theme_age_results = []
all_theme_gender_results = []

for scheme_name, output in theme_scheme_outputs.items():
    age_df = output["age_results"].copy()
    gender_df = output["gender_results"].copy()

    age_df["scheme"] = scheme_name
    gender_df["scheme"] = scheme_name

    all_theme_age_results.append(age_df)
    all_theme_gender_results.append(gender_df)

if all_theme_age_results:
    all_theme_age_results = pd.concat(all_theme_age_results, ignore_index=True)
else:
    all_theme_age_results = pd.DataFrame()

if all_theme_gender_results:
    all_theme_gender_results = pd.concat(all_theme_gender_results, ignore_index=True)
else:
    all_theme_gender_results = pd.DataFrame()


all_sentiment_age_results = []
all_sentiment_gender_results = []

for scheme_name, output in sentiment_scheme_outputs.items():
    age_df = output["age_results"].copy()
    gender_df = output["gender_results"].copy()

    age_df["scheme"] = scheme_name
    gender_df["scheme"] = scheme_name

    all_sentiment_age_results.append(age_df)
    all_sentiment_gender_results.append(gender_df)

if all_sentiment_age_results:
    all_sentiment_age_results = pd.concat(all_sentiment_age_results, ignore_index=True)
else:
    all_sentiment_age_results = pd.DataFrame()

if all_sentiment_gender_results:
    all_sentiment_gender_results = pd.concat(all_sentiment_gender_results, ignore_index=True)
else:
    all_sentiment_gender_results = pd.DataFrame()


if SAVE_OUTPUTS:
    all_theme_age_results.to_csv(
        os.path.join(OUTPUT_DIR, "ALL_SCHEMES_theme_age_ovr_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    all_theme_gender_results.to_csv(
        os.path.join(OUTPUT_DIR, "ALL_SCHEMES_theme_gender_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    all_sentiment_age_results.to_csv(
        os.path.join(OUTPUT_DIR, "ALL_SCHEMES_sentiment_age_ovr_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    all_sentiment_gender_results.to_csv(
        os.path.join(OUTPUT_DIR, "ALL_SCHEMES_sentiment_gender_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    scheme_column_check.to_csv(
        os.path.join(OUTPUT_DIR, "scheme_column_check.csv"),
        index=False,
        encoding="utf-8-sig"
    )


print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

print("\nTheme schemes completed:")
print(list(theme_scheme_outputs.keys()))

print("\nSentiment schemes completed:")
print(list(sentiment_scheme_outputs.keys()))

if SAVE_OUTPUTS:
    print("\nSaved outputs to:")
    print(OUTPUT_DIR)
    #%%
# ============================================================
# 16. REAL VS SHUFFLED NULL-MODEL COMPARISON FIGURES
# ============================================================
# Add this AFTER theme_scheme_outputs and sentiment_scheme_outputs are created.
#
# Purpose:
#   Compare real third sections against shuffled third sections.
#
# Main outputs:
#   1. Real-minus-shuffled delta dataframe
#   2. Delta heatmaps
#   3. Positional-signal attenuation heatmaps
#   4. Publication-resolution PDF/PNG files
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------------------
# Output directory for null-model figures
# ------------------------------------------------------------
NULL_FIG_DIR = os.path.join(OUTPUT_DIR, "real_vs_shuffled_figures")

if SAVE_OUTPUTS:
    os.makedirs(NULL_FIG_DIR, exist_ok=True)


# ------------------------------------------------------------
# Helper: save figures in publication-friendly formats
# ------------------------------------------------------------
def save_publication_figure(fig, filename_base, outdir=NULL_FIG_DIR, dpi=600):
    """
    Saves both PDF and PNG versions.
    PDF is preferred for vector graphics in manuscripts.
    PNG is useful for quick inspection.
    """
    if not SAVE_OUTPUTS:
        return

    pdf_path = os.path.join(outdir, f"{filename_base}.pdf")
    png_path = os.path.join(outdir, f"{filename_base}.png")

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


# ------------------------------------------------------------
# Helper: build real-vs-shuffled comparison table
# ------------------------------------------------------------
def make_real_vs_shuffle_comparison(
    real_df,
    shuf_df,
    label_col,
    real_scheme_name="third",
    shuf_scheme_name="shuf_third",
    section_order=None
):
    """
    Creates a tidy comparison dataframe.

    Expected input:
        real_df = theme_scheme_outputs["third"]["age_results"]
        shuf_df = theme_scheme_outputs["shuf_third"]["age_results"]

    Output includes:
        log_odds_real
        log_odds_shuf
        delta_real_minus_shuf
        p_real
        p_shuf
    """
    if section_order is None:
        section_order = ["Section 1", "Section 2", "Section 3"]

    real_keep = real_df[
        [
            label_col,
            "section",
            "Age Group",
            "Feature",
            "log_odds_ratio",
            "p-value",
            "Odds Ratio",
            "CI Lower",
            "CI Upper",
        ]
    ].copy()

    shuf_keep = shuf_df[
        [
            label_col,
            "section",
            "Age Group",
            "Feature",
            "log_odds_ratio",
            "p-value",
            "Odds Ratio",
            "CI Lower",
            "CI Upper",
        ]
    ].copy()

    real_keep = real_keep.rename(
        columns={
            "Feature": "Feature_real",
            "log_odds_ratio": "log_odds_real",
            "p-value": "p_real",
            "Odds Ratio": "OR_real",
            "CI Lower": "CI_lower_real",
            "CI Upper": "CI_upper_real",
        }
    )

    shuf_keep = shuf_keep.rename(
        columns={
            "Feature": "Feature_shuf",
            "log_odds_ratio": "log_odds_shuf",
            "p-value": "p_shuf",
            "Odds Ratio": "OR_shuf",
            "CI Lower": "CI_lower_shuf",
            "CI Upper": "CI_upper_shuf",
        }
    )

    compare = real_keep.merge(
        shuf_keep,
        on=[label_col, "section", "Age Group"],
        how="inner"
    )

    compare["real_scheme"] = real_scheme_name
    compare["shuf_scheme"] = shuf_scheme_name

    compare["delta_real_minus_shuf"] = (
        compare["log_odds_real"] - compare["log_odds_shuf"]
    )

    compare["abs_delta"] = compare["delta_real_minus_shuf"].abs()

    compare["real_sig"] = compare["p_real"].apply(get_sig_star)
    compare["shuf_sig"] = compare["p_shuf"].apply(get_sig_star)

    compare["section"] = pd.Categorical(
        compare["section"],
        categories=section_order,
        ordered=True
    )

    compare["Age Group"] = pd.Categorical(
        compare["Age Group"],
        categories=AGE_ORDER,
        ordered=True
    )

    return compare


# ------------------------------------------------------------
# Helper: positional-signal attenuation table
# ------------------------------------------------------------
def make_positional_signal_table(compare_df, label_col):
    """
    Computes how much section differentiation is reduced after shuffling.

    For each label × age group:
        real_range = max(real section logOR) - min(real section logOR)
        shuf_range = max(shuffled section logOR) - min(shuffled section logOR)
        attenuation = real_range - shuf_range

    Positive attenuation means:
        real ordering has stronger section-specific structure than shuffled order.
    """
    rows = []

    for (label, age_group), g in compare_df.groupby([label_col, "Age Group"], observed=True):
        real_vals = pd.to_numeric(g["log_odds_real"], errors="coerce")
        shuf_vals = pd.to_numeric(g["log_odds_shuf"], errors="coerce")

        real_range = real_vals.max() - real_vals.min()
        shuf_range = shuf_vals.max() - shuf_vals.min()

        attenuation = real_range - shuf_range

        if pd.notna(real_range) and real_range != 0:
            attenuation_ratio = attenuation / real_range
        else:
            attenuation_ratio = np.nan

        rows.append({
            label_col: label,
            "Age Group": age_group,
            "real_sectional_range": real_range,
            "shuf_sectional_range": shuf_range,
            "attenuation": attenuation,
            "attenuation_ratio": attenuation_ratio,
        })

    out = pd.DataFrame(rows)

    out["Age Group"] = pd.Categorical(
        out["Age Group"],
        categories=AGE_ORDER,
        ordered=True
    )

    return out


# ------------------------------------------------------------
# Plot 1: Delta heatmaps, faceted by theme/sentiment
# ------------------------------------------------------------
def plot_delta_heatmap_facets(
    compare_df,
    label_col,
    label_order,
    section_order,
    section_display,
    title,
    filename_base=None,
    n_cols=3,
    figsize_per_panel=(4.2, 3.1),
    center=0
):
    """
    Faceted heatmap:
        rows = Section
        columns = Age Group
        values = real logOR - shuffled logOR

    This is the clearest null-model figure.
    """
    set_constant_plot_style()

    plot_df = compare_df.copy()
    plot_df = plot_df[plot_df[label_col].isin(label_order)].copy()

    n_labels = len(label_order)
    n_rows = int(np.ceil(n_labels / n_cols))

    fig_width = figsize_per_panel[0] * n_cols
    fig_height = figsize_per_panel[1] * n_rows

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False
    )

    axes_flat = axes.flatten()

    vmax = np.nanmax(np.abs(plot_df["delta_real_minus_shuf"]))
    if pd.isna(vmax) or vmax == 0:
        vmax = 1.0

    for idx, label in enumerate(label_order):
        ax = axes_flat[idx]

        sub = plot_df[plot_df[label_col] == label].copy()

        heatmap_data = sub.pivot(
            index="section",
            columns="Age Group",
            values="delta_real_minus_shuf"
        ).reindex(index=section_order, columns=AGE_ORDER)

        annot_data = heatmap_data.round(2).astype(str)

        sns.heatmap(
            heatmap_data,
            ax=ax,
            cmap=FIG_STYLE["cmap"],
            center=center,
            vmin=-vmax,
            vmax=vmax,
            annot=annot_data,
            fmt="",
            linewidths=0.5,
            linecolor="gray",
            cbar=(idx == n_labels - 1),
            cbar_kws={"label": "Real − shuffled log-odds"}
        )

        ax.set_title(label, fontsize=13, fontproperties=font_prop)

        ax.set_xlabel("Age group", fontsize=11, fontproperties=font_prop)
        ax.set_ylabel("Section", fontsize=11, fontproperties=font_prop)

        ax.set_xticklabels(
            [AGE_LABELS.get(int(x.get_text()), x.get_text()) for x in ax.get_xticklabels()],
            rotation=0,
            fontsize=10,
            fontproperties=font_prop
        )

        ax.set_yticklabels(
            [section_display.get(x.get_text(), x.get_text()) for x in ax.get_yticklabels()],
            rotation=0,
            fontsize=10,
            fontproperties=font_prop
        )

    for idx in range(n_labels, len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    fig.suptitle(
        title,
        fontsize=15,
        weight="bold",
        y=1.02,
        fontproperties=font_prop
    )

    plt.tight_layout()

    if filename_base is not None:
        save_publication_figure(fig, filename_base)

    plt.show()


# ------------------------------------------------------------
# Plot 2: Positional-signal attenuation heatmap
# ------------------------------------------------------------
def plot_positional_attenuation_heatmap(
    signal_df,
    label_col,
    label_order,
    title,
    filename_base=None,
    figsize=(8.5, 4.8)
):
    """
    Heatmap:
        rows = theme/sentiment
        columns = age group
        values = real sectional spread - shuffled sectional spread

    Positive values mean real sentence order produces stronger positional structure.
    """
    set_constant_plot_style()

    plot_df = signal_df.copy()
    plot_df = plot_df[plot_df[label_col].isin(label_order)].copy()

    heatmap_data = plot_df.pivot(
        index=label_col,
        columns="Age Group",
        values="attenuation"
    ).reindex(index=label_order, columns=AGE_ORDER)

    annot_data = heatmap_data.round(2).astype(str)

    vmax = np.nanmax(np.abs(heatmap_data.to_numpy()))
    if pd.isna(vmax) or vmax == 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap=FIG_STYLE["cmap"],
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=annot_data,
        fmt="",
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"label": "Real spread − shuffled spread"}
    )

    ax.set_title(
        title,
        fontsize=14,
        weight="bold",
        fontproperties=font_prop
    )

    ax.set_xlabel("Age group", fontsize=12, fontproperties=font_prop)
    ax.set_ylabel("", fontsize=12, fontproperties=font_prop)

    ax.set_xticklabels(
        [AGE_LABELS.get(int(x.get_text()), x.get_text()) for x in ax.get_xticklabels()],
        rotation=0,
        fontsize=11,
        fontproperties=font_prop
    )

    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=11,
        fontproperties=font_prop
    )

    plt.tight_layout()

    if filename_base is not None:
        save_publication_figure(fig, filename_base)

    plt.show()


# ------------------------------------------------------------
# Plot 3: Real vs shuffled overlay line plot
# ------------------------------------------------------------
def plot_real_vs_shuffled_overlay_lines(
    compare_df,
    label_col,
    selected_labels,
    section_order,
    section_display,
    title,
    filename_base=None,
    n_rows=2,
    n_cols=3,
    figsize=(15, 8)
):
    """
    Overlay plot:
        x-axis = Age group
        color/line group = Section
        solid line = real
        dashed line = shuffled

    This is useful as a supplementary figure.
    """
    set_constant_plot_style()

    plot_df = compare_df.copy()
    plot_df = plot_df[plot_df[label_col].isin(selected_labels)].copy()

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharey=True
    )

    axes = axes.flatten()
    fig.subplots_adjust(right=0.82)

    for idx, label in enumerate(selected_labels):
        if idx >= len(axes):
            break

        ax = axes[idx]
        sub_label = plot_df[plot_df[label_col] == label].copy()

        for section in section_order:
            sub_sec = sub_label[sub_label["section"] == section].sort_values("Age Group")

            if sub_sec.empty:
                continue

            section_label = section_display.get(section, section)

            ax.plot(
                sub_sec["Age Group"].astype(int),
                sub_sec["log_odds_real"],
                marker="o",
                linewidth=2,
                linestyle="-",
                label=f"{section_label} real"
            )

            ax.plot(
                sub_sec["Age Group"].astype(int),
                sub_sec["log_odds_shuf"],
                marker="o",
                linewidth=1.6,
                linestyle="--",
                alpha=0.75,
                label=f"{section_label} shuffled"
            )

        ax.axhline(0, linestyle="--", color="gray", linewidth=1)

        ax.set_title(label, fontsize=13, fontproperties=font_prop)

        ax.set_xticks(AGE_ORDER)
        ax.set_xticklabels(
            [AGE_LABELS[i] for i in AGE_ORDER],
            fontsize=10,
            fontproperties=font_prop
        )

        ax.set_xlabel("Age group", fontsize=11, fontproperties=font_prop)

        if idx % n_cols == 0:
            ax.set_ylabel("Log-odds", fontsize=11, fontproperties=font_prop)

        ax.tick_params(bottom=True, left=True)

    for idx in range(len(selected_labels), len(axes)):
        fig.delaxes(axes[idx])

    handles, labels = axes[0].get_legend_handles_labels()

    if handles:
        fig.legend(
            handles,
            labels,
            title="Section / null model",
            loc="center left",
            bbox_to_anchor=(0.87, 0.5),
            frameon=False,
            fontsize=9,
            title_fontsize=10
        )

    fig.suptitle(
        title,
        fontsize=14,
        weight="bold",
        y=0.96,
        fontproperties=font_prop
    )

    plt.tight_layout(rect=[0, 0, 0.85, 0.94])

    if filename_base is not None:
        save_publication_figure(fig, filename_base)

    plt.show()


# ============================================================
# 17. RUN REAL VS SHUFFLED COMPARISONS: THEMES
# ============================================================

if "third" in theme_scheme_outputs and "shuf_third" in theme_scheme_outputs:

    theme_real_shuf_compare = make_real_vs_shuffle_comparison(
        real_df=theme_scheme_outputs["third"]["age_results"],
        shuf_df=theme_scheme_outputs["shuf_third"]["age_results"],
        label_col="theme",
        real_scheme_name="third",
        shuf_scheme_name="shuf_third",
        section_order=["Section 1", "Section 2", "Section 3"]
    )

    theme_positional_signal = make_positional_signal_table(
        theme_real_shuf_compare,
        label_col="theme"
    )

    if SAVE_OUTPUTS:
        theme_real_shuf_compare.to_csv(
            os.path.join(NULL_FIG_DIR, "theme_real_vs_shuffled_delta_table.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        theme_positional_signal.to_csv(
            os.path.join(NULL_FIG_DIR, "theme_positional_signal_attenuation_table.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    # Main-paper candidate: delta heatmap
    plot_delta_heatmap_facets(
        compare_df=theme_real_shuf_compare,
        label_col="theme",
        label_order=THEMES,
        section_order=["Section 1", "Section 2", "Section 3"],
        section_display={
            "Section 1": "Introduction",
            "Section 2": "Body",
            "Section 3": "Conclusion",
        },
        title=(
            "Theme positional effects beyond shuffled-order null model\n"
            "Delta = real third-section log-odds − shuffled third-section log-odds"
        ),
        filename_base="FIG_theme_real_minus_shuffled_delta_heatmaps",
        n_cols=3,
        figsize_per_panel=(4.2, 3.1)
    )

    # Main-paper or supplement: positional attenuation
    plot_positional_attenuation_heatmap(
        signal_df=theme_positional_signal,
        label_col="theme",
        label_order=THEMES,
        title=(
            "Theme positional signal attenuated by sentence-order shuffling\n"
            "Positive values indicate stronger section differentiation in real note order"
        ),
        filename_base="FIG_theme_positional_signal_attenuation",
        figsize=(8.5, 4.8)
    )

    # Supplement: direct overlay
    plot_real_vs_shuffled_overlay_lines(
        compare_df=theme_real_shuf_compare,
        label_col="theme",
        selected_labels=THEMES,
        section_order=["Section 1", "Section 2", "Section 3"],
        section_display={
            "Section 1": "Introduction",
            "Section 2": "Body",
            "Section 3": "Conclusion",
        },
        title=(
            "Theme effects in real versus shuffled section order\n"
            "Solid = real order, dashed = shuffled-order null"
        ),
        filename_base="SUPP_theme_real_vs_shuffled_overlay_lines",
        n_rows=2,
        n_cols=3,
        figsize=(16, 8)
    )

else:
    print(
        "Theme real-vs-shuffled comparison skipped: "
        "need theme_scheme_outputs['third'] and theme_scheme_outputs['shuf_third']."
    )


# ============================================================
# 18. RUN REAL VS SHUFFLED COMPARISONS: SENTIMENTS
# ============================================================

if "third" in sentiment_scheme_outputs and "shuf_third" in sentiment_scheme_outputs:

    sentiment_real_shuf_compare = make_real_vs_shuffle_comparison(
        real_df=sentiment_scheme_outputs["third"]["age_results"],
        shuf_df=sentiment_scheme_outputs["shuf_third"]["age_results"],
        label_col="sentiment",
        real_scheme_name="third",
        shuf_scheme_name="shuf_third",
        section_order=["Section 1", "Section 2", "Section 3"]
    )

    sentiment_positional_signal = make_positional_signal_table(
        sentiment_real_shuf_compare,
        label_col="sentiment"
    )

    if SAVE_OUTPUTS:
        sentiment_real_shuf_compare.to_csv(
            os.path.join(NULL_FIG_DIR, "sentiment_real_vs_shuffled_delta_table.csv"),
            index=False,
            encoding="utf-8-sig"
        )

        sentiment_positional_signal.to_csv(
            os.path.join(NULL_FIG_DIR, "sentiment_positional_signal_attenuation_table.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    # Main-paper candidate for selected sentiments
    plot_delta_heatmap_facets(
        compare_df=sentiment_real_shuf_compare,
        label_col="sentiment",
        label_order=LINEPLOT_SENTIMENTS,
        section_order=["Section 1", "Section 2", "Section 3"],
        section_display={
            "Section 1": "Introduction",
            "Section 2": "Body",
            "Section 3": "Conclusion",
        },
        title=(
            "Sentiment positional effects beyond shuffled-order null model\n"
            "Delta = real third-section log-odds − shuffled third-section log-odds"
        ),
        filename_base="FIG_sentiment_real_minus_shuffled_delta_heatmaps_selected",
        n_cols=3,
        figsize_per_panel=(4.2, 3.1)
    )

    # Full sentiment attenuation heatmap
    plot_positional_attenuation_heatmap(
        signal_df=sentiment_positional_signal,
        label_col="sentiment",
        label_order=SENTIMENTS,
        title=(
            "Sentiment positional signal attenuated by sentence-order shuffling\n"
            "Positive values indicate stronger section differentiation in real note order"
        ),
        filename_base="FIG_sentiment_positional_signal_attenuation_all",
        figsize=(8.5, 8.5)
    )

    # Supplement overlay for selected sentiments
    plot_real_vs_shuffled_overlay_lines(
        compare_df=sentiment_real_shuf_compare,
        label_col="sentiment",
        selected_labels=LINEPLOT_SENTIMENTS,
        section_order=["Section 1", "Section 2", "Section 3"],
        section_display={
            "Section 1": "Introduction",
            "Section 2": "Body",
            "Section 3": "Conclusion",
        },
        title=(
            "Sentiment effects in real versus shuffled section order\n"
            "Solid = real order, dashed = shuffled-order null"
        ),
        filename_base="SUPP_sentiment_real_vs_shuffled_overlay_lines_selected",
        n_rows=2,
        n_cols=3,
        figsize=(16, 8)
    )

else:
    print(
        "Sentiment real-vs-shuffled comparison skipped: "
        "need sentiment_scheme_outputs['third'] and sentiment_scheme_outputs['shuf_third']."
    )
    
#%%

# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:01:24 2026

@author: Jae Bin Park
"""

# -*- coding: utf-8 -*-
"""
SENTENCE_DF VERSION OF MULTI-SCHEME DOWNSTREAM ANALYSIS

Use this when you already have sentence_df with columns like:

    note_id
    sentence_index
    n_sent
    rel_pos
    sentence_text
    third_bin
    quart_bin
    tokenizedkluekeywordsentencetransformer_sentence
    sentiment

And a note-level dataframe with demographics:
    kfsppororo or scored
    SEX
    AGE2
    pred_label, optional

This code:
    1. Creates shuffled sentence positions within each note
    2. Builds sentence-level theme flags
    3. Builds sentence-level sentiment flags
    4. Aggregates sentence flags into note-level section features
    5. Runs age one-vs-rest logistic regression controlling for gender
    6. Runs gender logistic regression controlling for age
    7. Makes the first-version line plots:
           x-axis = Age Group
           lines = Section
           panels = Theme/Sentiment
    8. Makes combined demographic heatmaps
"""

# ============================================================
# 0. IMPORTS
# ============================================================

import ast
import os
import re
import warnings

import numpy as np
import pandas as pd

from tqdm import tqdm

import statsmodels.api as sm

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

warnings.filterwarnings("ignore")
tqdm.pandas()


# ============================================================
# 1. OPTIONS
# ============================================================

SAVE_OUTPUTS = True
OUTPUT_DIR = r"C:\Users\Jae Bin Park\sentence_df_outputs"

if SAVE_OUTPUTS:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. FIGURE FORMATTING
# ============================================================

FONT_CANDIDATES = [
    r"C:\Users\Jae Bin Park\AppData\Local\Microsoft\Windows\Fonts\Roboto-Bold.ttf",
    r"C:\Windows\Fonts\Arialbd.ttf",
    r"C:\Windows\Fonts\malgunbd.ttf",
    r"C:\Windows\Fonts\malgun.ttf",
]

font_prop = None
custom_font = None

for fp in FONT_CANDIDATES:
    try:
        if os.path.exists(fp):
            font_prop = fm.FontProperties(fname=fp)
            custom_font = font_prop.get_name()
            break
    except Exception:
        continue

if custom_font is not None:
    plt.rcParams["font.family"] = custom_font


FIG_STYLE = {
    "cmap": "coolwarm",
    "center": 0,
    "line_width": 2,
    "marker": "o",
    "zero_line_style": "--",
    "zero_line_color": "gray",
    "zero_line_width": 1,
    "heatmap_linewidths": 0.5,
    "heatmap_linecolor": "gray",
    "title_fontsize": 14,
    "axis_label_fontsize": 12,
    "tick_fontsize": 11,
    "annot_fontsize": 10,
    "star_fontsize": 11,
    "legend_fontsize": 11,
    "legend_title_fontsize": 12,
}

AGE_ORDER = [1, 2, 3, 4, 5]

AGE_LABELS = {
    1: "≤18",
    2: "19–34",
    3: "35–49",
    4: "50–64",
    5: "65+",
}


def set_constant_plot_style():
    sns.set(style="white", context="notebook", font_scale=1.2)

    plt.rcParams.update({
        "axes.linewidth": 1.2,
        "xtick.major.width": 1,
        "ytick.major.width": 1,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.dpi": 100,
    })

    if custom_font is not None:
        plt.rcParams["font.family"] = custom_font


def get_sig_star(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


def safe_log_or(x):
    x = pd.to_numeric(x, errors="coerce")
    x = np.clip(x, np.finfo(float).tiny, np.inf)
    return np.log(x)


# ============================================================
# 3. PARSING HELPERS
# ============================================================

def parse_listlike(x):
    if isinstance(x, list):
        return x

    if pd.isna(x):
        return []

    if isinstance(x, str):
        x = x.strip()

        if x == "":
            return []

        if x.startswith("[") and x.endswith("]"):
            try:
                parsed = ast.literal_eval(x)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                return []

    return []


def parse_dictlike(x):
    if isinstance(x, dict):
        if "labels" not in x:
            x["labels"] = []
        if "scores" not in x:
            x["scores"] = []
        return x

    if pd.isna(x):
        return {"labels": [], "scores": []}

    if isinstance(x, str):
        x = x.strip()

        if x == "":
            return {"labels": [], "scores": []}

        if x.startswith("{") and x.endswith("}"):
            try:
                parsed = ast.literal_eval(x)
                if isinstance(parsed, dict):
                    if "labels" not in parsed:
                        parsed["labels"] = []
                    if "scores" not in parsed:
                        parsed["scores"] = []
                    return parsed
            except Exception:
                return {"labels": [], "scores": []}

    return {"labels": [], "scores": []}


def extract_tokens_from_keyword_list(keyword_list):
    tokens = []

    keyword_list = parse_listlike(keyword_list)

    for item in keyword_list:
        if isinstance(item, tuple) and len(item) >= 1:
            tokens.append(str(item[0]))
        elif isinstance(item, list) and len(item) >= 1:
            tokens.append(str(item[0]))
        elif isinstance(item, str):
            tokens.append(item)

    return tokens


# ============================================================
# 4. THEME AND SENTIMENT DICTIONARIES
# ============================================================

THEME_TOKEN_DICT = {
    "Sorry and Shame": [
        "미안", "죄송", "용서", "잘못", "후회", "죄", "책임", "민폐", "반성"
    ],
    "Love and Gratitude": [
        "사랑", "고맙", "감사", "고생", "수고", "보고싶", "애틋", "소중", "그리"
    ],
    "Burden": [
        "버겁", "부담", "짐", "감당", "힘들", "무겁", "지치", "헷갈"
    ],
    "Despair": [
        "포기", "좌절", "죽", "끝", "아무것", "헛되", "무의미", "잊히", "없어지", "그만두"
    ],
    "Post-mortem Affairs": [
        "부탁", "정리", "남기", "처리", "보험", "은행", "장례", "통장", "유서"
    ],
}

THEMES = list(THEME_TOKEN_DICT.keys())


sentiment_label_map = {
    "Despair": "절망",
    "Defeat": "패배/자기혐오",
    "Exhaustion": "힘듦/지침",
    "Anxious": "불안/걱정",
    "Disappointment": "안타까움/실망",
    "Anger": "화남/분노",
    "Hatred": "증오/혐오",
    "Resentment": "어이없음",
    "Sadness": "슬픔",
    "Sorrow": "서러움",
    "Gratitude": "고마움",
    "Affection": "흐뭇함(귀여움/예쁨)",
    "Happiness": "행복",
    "Relief": "안심/신뢰",
    "Neutral": "없음",
}

SENTIMENTS = list(sentiment_label_map.keys())

LINEPLOT_SENTIMENTS = [
    "Defeat",
    "Exhaustion",
    "Neutral",
    "Sadness",
    "Happiness",
    "Disappointment",
]

sentence_df=pd.read_excel(r'C:\Users\Jae Bin Park\kfsp_sentence_relativepos.xlsx')

# ============================================================
# 5. PREPARE NOTE-LEVEL DEMOGRAPHIC DATA
# ============================================================

# You can use scored if it exists, otherwise kfsppororo.
# Important: note_id in sentence_df should correspond to the row index
# from the note-level dataframe.

if "scored" in globals():
    note_df = scored.copy()
elif "kfsppororo" in globals():
    note_df = kfsppororo.copy()
else:
    raise NameError("Need either `scored` or `kfsppororo` in memory.")

if "sentence_df" not in globals():
    raise NameError("`sentence_df` is not in memory.")

if "note_id" not in sentence_df.columns:
    raise KeyError("sentence_df must contain `note_id`.")

if "SEX" not in note_df.columns or "AGE2" not in note_df.columns:
    raise KeyError("note-level dataframe must contain SEX and AGE2.")

# Keep only raw notes if pred_label exists.
if "pred_label" in note_df.columns:
    note_df = note_df[note_df["pred_label"].eq("raw")].copy()

note_demo = note_df[["SEX", "AGE2"]].copy()
note_demo = note_demo.reset_index().rename(columns={"index": "note_id"})

valid_note_ids = set(note_demo["note_id"])

sentence_work = sentence_df[sentence_df["note_id"].isin(valid_note_ids)].copy()

print("Sentence rows kept:", sentence_work.shape[0])
print("Unique notes kept:", sentence_work["note_id"].nunique())


# ============================================================
# 6. CREATE SHUFFLED SENTENCE POSITIONS
# ============================================================

def add_shuffled_sentence_positions(sentence_df_in, seed=42):
    """
    Keeps sentence content fixed but randomly reassigns sentence positions
    within each note.

    Adds:
        shuf_sentence_index
        shuf_rel_pos
        shuf_third_bin
        shuf_quart_bin
    """
    rng = np.random.default_rng(seed)
    parts = []

    for note_id, group in tqdm(
        sentence_df_in.groupby("note_id", sort=False),
        desc="Creating shuffled sentence positions"
    ):
        g = group.copy()
        n = len(g)

        if n == 0:
            continue

        shuffled_positions = rng.permutation(np.arange(1, n + 1))

        g["shuf_sentence_index"] = shuffled_positions
        g["shuf_rel_pos"] = (g["shuf_sentence_index"] - 0.5) / n

        g["shuf_third_bin"] = np.minimum(
            np.floor(g["shuf_rel_pos"] * 3).astype(int) + 1,
            3
        )

        g["shuf_quart_bin"] = np.minimum(
            np.floor(g["shuf_rel_pos"] * 4).astype(int) + 1,
            4
        )

        parts.append(g)

    return pd.concat(parts, ignore_index=True)


sentence_work = add_shuffled_sentence_positions(sentence_work, seed=42)


# ============================================================
# 7. CREATE SENTENCE-LEVEL THEME FLAGS
# ============================================================

keyword_col = "tokenizedkluekeywordsentencetransformer_sentence"

if keyword_col not in sentence_work.columns:
    raise KeyError(f"sentence_df must contain `{keyword_col}`.")

sentence_work[keyword_col] = sentence_work[keyword_col].apply(parse_listlike)


def sentence_has_theme(keyword_list, target_tokens):
    tokens = extract_tokens_from_keyword_list(keyword_list)

    return int(
        any(
            any(target in token for token in tokens)
            for target in target_tokens
        )
    )


for theme, target_tokens in THEME_TOKEN_DICT.items():
    flag_col = f"theme_{theme}"
    sentence_work[flag_col] = sentence_work[keyword_col].apply(
        lambda x: sentence_has_theme(x, target_tokens)
    )

theme_flag_cols = [f"theme_{theme}" for theme in THEMES]


# ============================================================
# 8. CREATE SENTENCE-LEVEL SENTIMENT FLAGS
# ============================================================

sentiment_col = "sentiment"

if sentiment_col not in sentence_work.columns:
    raise KeyError("sentence_df must contain `sentiment`.")

sentence_work[sentiment_col] = sentence_work[sentiment_col].apply(parse_dictlike)


for english_label, korean_label in sentiment_label_map.items():
    flag_col = f"sentiment_{english_label}"
    sentence_work[flag_col] = sentence_work[sentiment_col].apply(
        lambda d: int(korean_label in d.get("labels", []))
    )

sentiment_flag_cols = [f"sentiment_{sentiment}" for sentiment in SENTIMENTS]


# ============================================================
# 9. SENTENCE-LEVEL SECTION SCHEMES
# ============================================================

SENTENCE_SCHEMES = {
    "sent_real_third": {
        "bin_col": "third_bin",
        "n_bins": 3,
        "section_labels": ["Section 1", "Section 2", "Section 3"],
        "section_display": {
            "Section 1": "Introduction",
            "Section 2": "Body",
            "Section 3": "Conclusion",
        },
        "min_sentences": 3,
    },
    "sent_real_quart": {
        "bin_col": "quart_bin",
        "n_bins": 4,
        "section_labels": ["Section 1", "Section 2", "Section 3", "Section 4"],
        "section_display": {
            "Section 1": "Q1",
            "Section 2": "Q2",
            "Section 3": "Q3",
            "Section 4": "Q4",
        },
        "min_sentences": 4,
    },
    "sent_shuf_third": {
        "bin_col": "shuf_third_bin",
        "n_bins": 3,
        "section_labels": ["Section 1", "Section 2", "Section 3"],
        "section_display": {
            "Section 1": "Shuffled 1",
            "Section 2": "Shuffled 2",
            "Section 3": "Shuffled 3",
        },
        "min_sentences": 3,
    },
    "sent_shuf_quart": {
        "bin_col": "shuf_quart_bin",
        "n_bins": 4,
        "section_labels": ["Section 1", "Section 2", "Section 3", "Section 4"],
        "section_display": {
            "Section 1": "Shuffled Q1",
            "Section 2": "Shuffled Q2",
            "Section 3": "Shuffled Q3",
            "Section 4": "Shuffled Q4",
        },
        "min_sentences": 4,
    },
}


# ============================================================
# 10. AGGREGATE SENTENCE FLAGS TO NOTE-LEVEL FEATURES
# ============================================================

def aggregate_sentence_flags_to_note_features(
    sentence_df_in,
    note_demo_df,
    flag_cols,
    label_names,
    bin_col,
    n_bins,
    scheme_name,
    min_sentences
):
    """
    Aggregates sentence-level binary flags into note-level section features.

    If any sentence in a note-section has the flag, the note-section feature = 1.

    Feature name format:
        sent_real_third Section 1 Sorry and Shame
        sent_shuf_quart Section 4 Despair
    """
    rows = []

    if bin_col not in sentence_df_in.columns:
        raise KeyError(f"Missing bin column: {bin_col}")

    for note_id, group in tqdm(
        sentence_df_in.groupby("note_id", sort=False),
        desc=f"Aggregating {scheme_name}"
    ):
        n_sent = int(group["n_sent"].iloc[0]) if "n_sent" in group.columns else len(group)

        if n_sent < min_sentences:
            continue

        row = {"note_id": note_id}

        for b in range(1, n_bins + 1):
            section_group = group[group[bin_col] == b]
            section_label = f"Section {b}"

            for flag_col, clean_label in zip(flag_cols, label_names):
                feature_name = f"{scheme_name} {section_label} {clean_label}"

                if len(section_group) == 0:
                    row[feature_name] = 0
                else:
                    row[feature_name] = int(section_group[flag_col].max())

        rows.append(row)

    feature_df = pd.DataFrame(rows)

    out = note_demo_df.merge(feature_df, on="note_id", how="inner")

    feature_names = [
        col for col in out.columns
        if col.startswith(f"{scheme_name} Section")
    ]

    return out, feature_names


# ============================================================
# 11. MODEL HELPERS
# ============================================================

def fit_age_ovr_logit(
    df,
    feature_names,
    outcome_col="AGE2",
    sex_col="SEX",
    gender_covariate_name="Gender",
    maxiter=200
):
    work = df[
        df[outcome_col].isin([1, 2, 3, 4, 5])
        & df[sex_col].isin([1, 2])
    ].copy()

    work[gender_covariate_name] = work[sex_col].replace({1: 1, 2: 0}).astype(int)

    X = work[feature_names + [gender_covariate_name]].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = sm.add_constant(X, has_constant="add")
    X = X.astype(float)

    group_results = {}

    for group in AGE_ORDER:
        y = (work[outcome_col] == group).astype(int)

        mask = ~(X.isna().any(axis=1) | y.isna())
        Xi = X.loc[mask].copy()
        yi = y.loc[mask].copy()

        try:
            model = sm.Logit(yi, Xi)
            result = model.fit(disp=False, maxiter=maxiter)

            odds_ratios = np.exp(result.params)
            conf = np.exp(result.conf_int())
            conf.columns = ["CI Lower", "CI Upper"]

            summary_df = pd.DataFrame({
                "Feature": result.params.index,
                "Coefficient": result.params.values,
                "Odds Ratio": odds_ratios.values,
                "p-value": result.pvalues.values,
                "CI Lower": conf["CI Lower"].values,
                "CI Upper": conf["CI Upper"].values,
                "Age Group": group,
                "n_model": len(yi),
            })

        except Exception as e:
            print(f"[WARN] Age model failed for group {group}: {e}")

            summary_df = pd.DataFrame({
                "Feature": feature_names,
                "Coefficient": np.nan,
                "Odds Ratio": np.nan,
                "p-value": np.nan,
                "CI Lower": np.nan,
                "CI Upper": np.nan,
                "Age Group": group,
                "n_model": len(yi),
            })

        group_results[group] = summary_df

    combined = pd.concat(group_results.values(), ignore_index=True)

    combined = combined[combined["Feature"].isin(feature_names)].copy()
    combined["log_odds_ratio"] = safe_log_or(combined["Odds Ratio"])
    combined["sig_star"] = combined["p-value"].apply(get_sig_star)
    combined["annot"] = (
        combined["log_odds_ratio"].round(2).astype(str)
        + combined["sig_star"]
    )
    combined["Demographic"] = combined["Age Group"].apply(lambda x: f"Age{x}")

    return combined


def fit_gender_logit_controlling_age(
    df,
    feature_names,
    sex_col="SEX",
    age_col="AGE2",
    maxiter=200
):
    work = df[
        df[sex_col].isin([1, 2])
        & df[age_col].isin([1, 2, 3, 4, 5])
    ].copy()

    work["SEX_BINARY"] = work[sex_col].replace({1: 1, 2: 0}).astype(int)

    age_dummies = pd.get_dummies(
        work[age_col],
        prefix="AGE2",
        drop_first=True,
        dtype=int
    )

    for col in ["AGE2_2", "AGE2_3", "AGE2_4", "AGE2_5"]:
        if col not in age_dummies.columns:
            age_dummies[col] = 0

    age_dummies = age_dummies[["AGE2_2", "AGE2_3", "AGE2_4", "AGE2_5"]]

    X = pd.concat([work[feature_names], age_dummies], axis=1)

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = sm.add_constant(X, has_constant="add")
    X = X.astype(float)

    y = work["SEX_BINARY"].astype(float)

    mask = ~(X.isna().any(axis=1) | y.isna())
    X_clean = X.loc[mask].copy()
    y_clean = y.loc[mask].copy()

    print("Gender model diagnostics:")
    print("  X_clean shape:", X_clean.shape)
    print("  y_clean shape:", y_clean.shape)
    print("  Class balance:", y_clean.value_counts().to_dict())

    try:
        model = sm.Logit(y_clean, X_clean)
        result = model.fit(disp=False, maxiter=maxiter)

        odds_ratios = np.exp(result.params)
        conf = np.exp(result.conf_int())
        conf.columns = ["CI Lower", "CI Upper"]

        summary_df = pd.DataFrame({
            "Feature": result.params.index,
            "Coefficient": result.params.values,
            "Odds Ratio": odds_ratios.values,
            "p-value": result.pvalues.values,
            "CI Lower": conf["CI Lower"].values,
            "CI Upper": conf["CI Upper"].values,
            "n_model": len(y_clean),
        })

    except Exception as e:
        print(f"[WARN] Gender model failed: {e}")

        result = None

        summary_df = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": np.nan,
            "Odds Ratio": np.nan,
            "p-value": np.nan,
            "CI Lower": np.nan,
            "CI Upper": np.nan,
            "n_model": len(y_clean),
        })

    summary_df = summary_df[summary_df["Feature"].isin(feature_names)].copy()
    summary_df["Demographic"] = "Gender (Male)"
    summary_df["log_odds_ratio"] = safe_log_or(summary_df["Odds Ratio"])
    summary_df["sig_star"] = summary_df["p-value"].apply(get_sig_star)
    summary_df["annot"] = (
        summary_df["log_odds_ratio"].round(2).astype(str)
        + summary_df["sig_star"]
    )

    return summary_df, result


def split_sentence_scheme_section_and_label(df, label_col_name):
    """
    Splits feature names like:
        sent_real_third Section 1 Sorry and Shame
        sent_shuf_quart Section 4 Despair
    """
    out = df.copy()

    extracted = out["Feature"].str.extract(
        r"^(?P<scheme>.+?)\s+(?P<section>Section\s+\d+)\s+(?P<label>.+)$"
    )

    out["scheme"] = extracted["scheme"]
    out["section"] = extracted["section"]
    out[label_col_name] = extracted["label"]

    return out


# ============================================================
# 12. PLOT HELPERS
# ============================================================

def plot_first_version_lineplot_by_age(
    combined_df,
    label_col,
    selected_labels,
    section_order,
    section_display,
    title,
    n_rows=2,
    n_cols=3,
    figsize=(15, 8)
):
    """
    First-version plot:
        x-axis = Age Group
        lines = Section
        panels = Theme/Sentiment
    """
    set_constant_plot_style()

    line_df = combined_df.copy()

    line_df = line_df[line_df[label_col].isin(selected_labels)].copy()
    line_df = line_df[line_df["section"].isin(section_order)].copy()

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharey=True
    )

    axes = axes.flatten()
    fig.subplots_adjust(right=0.82)

    for idx, label in enumerate(selected_labels):
        if idx >= len(axes):
            break

        ax = axes[idx]
        label_data = line_df[line_df[label_col] == label].copy()

        for section in section_order:
            line_data = (
                label_data[label_data["section"] == section]
                .sort_values("Age Group")
            )

            if line_data.empty:
                continue

            ax.plot(
                line_data["Age Group"],
                line_data["log_odds_ratio"],
                marker="o",
                label=section_display.get(section, section),
                linewidth=2
            )

            for _, row_ in line_data.iterrows():
                x = row_["Age Group"]
                y = row_["log_odds_ratio"]
                star = get_sig_star(row_["p-value"])

                if star and pd.notna(y):
                    ax.text(
                        x,
                        y,
                        star,
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        weight="bold"
                    )

        ax.set_title(
            f"{label}",
            fontsize=13,
            fontproperties=font_prop
        )

        tick_positions = [1, 2, 3, 4, 5]
        tick_labels = ["≤18", "19–34", "35–49", "50–64", "65+"]

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            tick_labels,
            fontsize=10,
            fontproperties=font_prop
        )

        ax.axhline(0, linestyle="--", color="gray", linewidth=1)

        ax.set_xlabel(
            "Age Group",
            fontsize=11,
            fontproperties=font_prop
        )

        ax.tick_params(bottom=True, left=True)

        if idx % n_cols == 0:
            ax.set_ylabel(
                "Log-Odds",
                fontsize=11,
                fontproperties=font_prop
            )

        if font_prop is not None:
            for tick_label in ax.get_yticklabels():
                tick_label.set_fontproperties(font_prop)
                tick_label.set_fontsize(12)

            for tick_label in ax.get_xticklabels():
                tick_label.set_fontproperties(font_prop)
                tick_label.set_fontsize(12)

    for idx in range(len(selected_labels), len(axes)):
        fig.delaxes(axes[idx])

    handles, labels = axes[0].get_legend_handles_labels()

    if handles:
        fig.legend(
            handles,
            labels,
            title="Section",
            loc="center left",
            bbox_to_anchor=(0.87, 0.5),
            frameon=False,
            fontsize=11,
            title_fontsize=12
        )

    plt.suptitle(
        title,
        fontsize=14,
        weight="bold",
        y=0.95,
        fontproperties=font_prop
    )

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    plt.show()

    plt.rcdefaults()


def plot_combined_demographic_heatmap(
    age_df,
    gender_df,
    feature_order,
    row_label_kind,
    title,
    figsize=(12, 10),
    section_display_map=None
):
    set_constant_plot_style()

    if section_display_map is None:
        section_display_map = {
            "Section 1": "Opening",
            "Section 2": "Middle",
            "Section 3": "Ending",
            "Section 4": "Q4",
        }

    common_cols = [
        "Feature",
        "Coefficient",
        "Odds Ratio",
        "p-value",
        "CI Lower",
        "CI Upper",
        "Demographic",
        "log_odds_ratio",
        "sig_star",
        "annot",
    ]

    age_tbl = age_df.copy()
    gender_tbl = gender_df.copy()

    if "Demographic" not in age_tbl.columns:
        age_tbl["Demographic"] = age_tbl["Age Group"].apply(lambda x: f"Age{x}")

    if "Demographic" not in gender_tbl.columns:
        gender_tbl["Demographic"] = "Gender (Male)"

    age_tbl = age_tbl[[c for c in common_cols if c in age_tbl.columns]].copy()
    gender_tbl = gender_tbl[[c for c in common_cols if c in gender_tbl.columns]].copy()

    merged_df = pd.concat([age_tbl, gender_tbl], ignore_index=True)

    demographic_order = [f"Age{i}" for i in AGE_ORDER] + ["Gender (Male)"]
    xtick_labels = [AGE_LABELS[i] for i in AGE_ORDER] + ["Gender (Male)"]

    heatmap_data = pd.pivot_table(
        merged_df,
        index="Feature",
        columns="Demographic",
        values="log_odds_ratio",
        aggfunc="first"
    ).reindex(index=feature_order, columns=demographic_order)

    eps = np.finfo(float).tiny

    merged_df["CI Lower"] = pd.to_numeric(
        merged_df["CI Lower"],
        errors="coerce"
    ).clip(lower=eps)

    merged_df["CI Upper"] = pd.to_numeric(
        merged_df["CI Upper"],
        errors="coerce"
    ).clip(lower=eps)

    merged_df["ci_lo_log"] = np.log(merged_df["CI Lower"])
    merged_df["ci_hi_log"] = np.log(merged_df["CI Upper"])

    merged_df["annot_main"] = (
        merged_df["log_odds_ratio"].round(2).astype(str)
        + merged_df["sig_star"].astype(str)
    )

    merged_df["annot_ci_only"] = (
        "["
        + merged_df["ci_lo_log"].round(2).astype(str)
        + ", "
        + merged_df["ci_hi_log"].round(2).astype(str)
        + "]"
    )

    annot_main_tbl = pd.pivot_table(
        merged_df,
        index="Feature",
        columns="Demographic",
        values="annot_main",
        aggfunc="first"
    ).reindex(index=feature_order, columns=demographic_order)

    annot_ci_tbl = pd.pivot_table(
        merged_df,
        index="Feature",
        columns="Demographic",
        values="annot_ci_only",
        aggfunc="first"
    ).reindex(index=feature_order, columns=demographic_order)

    plt.figure(figsize=figsize)

    ax = sns.heatmap(
        heatmap_data,
        annot=False,
        fmt="",
        cmap=FIG_STYLE["cmap"],
        center=FIG_STYLE["center"],
        linewidths=FIG_STYLE["heatmap_linewidths"],
        linecolor=FIG_STYLE["heatmap_linecolor"],
        cbar_kws={"label": "Log-Odds"}
    )

    mesh = ax.collections[0]

    def pick_text_color_from_value(val, mappable, thresh=0.53):
        rgba = mappable.cmap(mappable.norm(val if pd.notna(val) else 0.0))
        r, g, b, _ = rgba
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "black" if lum > thresh else "white"

    for i, row_key in enumerate(heatmap_data.index):
        for j, col_key in enumerate(heatmap_data.columns):
            val = heatmap_data.loc[row_key, col_key]
            main_txt = annot_main_tbl.loc[row_key, col_key]
            ci_txt = annot_ci_tbl.loc[row_key, col_key]

            if pd.isna(main_txt) and pd.isna(ci_txt):
                continue

            color = pick_text_color_from_value(val, mesh)

            if pd.notna(main_txt):
                ax.text(
                    j + 0.5,
                    i + 0.42,
                    str(main_txt),
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=color,
                    fontproperties=font_prop
                )

            if pd.notna(ci_txt):
                ax.text(
                    j + 0.5,
                    i + 0.68,
                    str(ci_txt),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                    fontproperties=font_prop
                )

    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel("Log-Odds", fontsize=12)

    if font_prop is not None:
        cbar.ax.yaxis.label.set_fontproperties(font_prop)

    ax.set_xticks(np.arange(len(demographic_order)) + 0.5)
    ax.set_xticklabels(
        xtick_labels,
        rotation=0,
        fontsize=11,
        fontproperties=font_prop
    )

    ytick_labels = [
        re.sub(r"^.+?\s+Section\s+\d+\s+", "", str(lbl))
        for lbl in heatmap_data.index
    ]

    ax.set_yticklabels(
        ytick_labels,
        rotation=0,
        fontsize=11,
        fontproperties=font_prop
    )

    ax.set_title(
        title,
        fontsize=14,
        fontproperties=font_prop
    )

    ax.set_xlabel(
        "Demographic",
        fontsize=12,
        fontproperties=font_prop
    )

    ax.set_ylabel(
        row_label_kind,
        fontsize=12,
        fontproperties=font_prop
    )

    ax.yaxis.set_label_coords(-0.35, 0.5)

    x_section_label = -1.4 if len(feature_order) > 20 else -1.2
    row_labels = list(heatmap_data.index)

    for section_prefix, display_name in section_display_map.items():
        rows_for_section = [
            i for i, lbl in enumerate(row_labels)
            if re.search(fr"\b{re.escape(section_prefix)}\b", str(lbl))
        ]

        if not rows_for_section:
            continue

        y_middle = (rows_for_section[0] + rows_for_section[-1]) / 2.0

        ax.text(
            x=x_section_label,
            y=y_middle,
            s=display_name,
            va="center",
            ha="center",
            rotation=90,
            fontsize=14,
            weight="bold",
            color="black",
            transform=ax.transData,
            clip_on=False,
            fontproperties=font_prop
        )

    ax.tick_params(bottom=True, left=True)

    plt.tight_layout()
    plt.show()


# ============================================================
# 13. RUN SENTENCE_DF THEME ANALYSIS
# ============================================================

sentence_theme_outputs = {}

for scheme_name, scheme_config in SENTENCE_SCHEMES.items():
    print("\n" + "=" * 80)
    print(f"SENTENCE_DF THEME ANALYSIS: {scheme_name}")
    print("=" * 80)

    try:
        scheme_df, feature_names = aggregate_sentence_flags_to_note_features(
            sentence_df_in=sentence_work,
            note_demo_df=note_demo,
            flag_cols=theme_flag_cols,
            label_names=THEMES,
            bin_col=scheme_config["bin_col"],
            n_bins=scheme_config["n_bins"],
            scheme_name=scheme_name,
            min_sentences=scheme_config["min_sentences"]
        )

        age_results = fit_age_ovr_logit(
            scheme_df,
            feature_names=feature_names,
            outcome_col="AGE2",
            sex_col="SEX"
        )

        age_results = split_sentence_scheme_section_and_label(
            age_results,
            label_col_name="theme"
        )

        gender_results, gender_model = fit_gender_logit_controlling_age(
            scheme_df,
            feature_names=feature_names,
            sex_col="SEX",
            age_col="AGE2"
        )

        gender_results = split_sentence_scheme_section_and_label(
            gender_results,
            label_col_name="theme"
        )

        sentence_theme_outputs[scheme_name] = {
            "df": scheme_df,
            "feature_names": feature_names,
            "age_results": age_results,
            "gender_results": gender_results,
            "gender_model": gender_model,
        }

        if SAVE_OUTPUTS:
            scheme_df.to_csv(
                os.path.join(OUTPUT_DIR, f"{scheme_name}_sentence_theme_features_df.csv"),
                index=False,
                encoding="utf-8-sig"
            )

            age_results.to_csv(
                os.path.join(OUTPUT_DIR, f"{scheme_name}_sentence_theme_age_results.csv"),
                index=False,
                encoding="utf-8-sig"
            )

            gender_results.to_csv(
                os.path.join(OUTPUT_DIR, f"{scheme_name}_sentence_theme_gender_results.csv"),
                index=False,
                encoding="utf-8-sig"
            )

    except Exception as e:
        print(f"[ERROR] Sentence theme analysis failed for {scheme_name}: {e}")


# ============================================================
# 14. RUN SENTENCE_DF SENTIMENT ANALYSIS
# ============================================================

sentence_sentiment_outputs = {}

for scheme_name, scheme_config in SENTENCE_SCHEMES.items():
    print("\n" + "=" * 80)
    print(f"SENTENCE_DF SENTIMENT ANALYSIS: {scheme_name}")
    print("=" * 80)

    try:
        scheme_df, feature_names = aggregate_sentence_flags_to_note_features(
            sentence_df_in=sentence_work,
            note_demo_df=note_demo,
            flag_cols=sentiment_flag_cols,
            label_names=SENTIMENTS,
            bin_col=scheme_config["bin_col"],
            n_bins=scheme_config["n_bins"],
            scheme_name=scheme_name,
            min_sentences=scheme_config["min_sentences"]
        )

        age_results = fit_age_ovr_logit(
            scheme_df,
            feature_names=feature_names,
            outcome_col="AGE2",
            sex_col="SEX"
        )

        age_results = split_sentence_scheme_section_and_label(
            age_results,
            label_col_name="sentiment"
        )

        gender_results, gender_model = fit_gender_logit_controlling_age(
            scheme_df,
            feature_names=feature_names,
            sex_col="SEX",
            age_col="AGE2"
        )

        gender_results = split_sentence_scheme_section_and_label(
            gender_results,
            label_col_name="sentiment"
        )

        sentence_sentiment_outputs[scheme_name] = {
            "df": scheme_df,
            "feature_names": feature_names,
            "age_results": age_results,
            "gender_results": gender_results,
            "gender_model": gender_model,
        }

        if SAVE_OUTPUTS:
            scheme_df.to_csv(
                os.path.join(OUTPUT_DIR, f"{scheme_name}_sentence_sentiment_features_df.csv"),
                index=False,
                encoding="utf-8-sig"
            )

            age_results.to_csv(
                os.path.join(OUTPUT_DIR, f"{scheme_name}_sentence_sentiment_age_results.csv"),
                index=False,
                encoding="utf-8-sig"
            )

            gender_results.to_csv(
                os.path.join(OUTPUT_DIR, f"{scheme_name}_sentence_sentiment_gender_results.csv"),
                index=False,
                encoding="utf-8-sig"
            )

    except Exception as e:
        print(f"[ERROR] Sentence sentiment analysis failed for {scheme_name}: {e}")


# ============================================================
# 15. PLOT SENTENCE_DF THEME RESULTS
# ============================================================

for scheme_name, output in sentence_theme_outputs.items():
    print("\n" + "=" * 80)
    print(f"PLOTTING SENTENCE_DF THEME RESULTS: {scheme_name}")
    print("=" * 80)

    scheme_config = SENTENCE_SCHEMES[scheme_name]
    section_order = scheme_config["section_labels"]
    section_display = scheme_config["section_display"]

    age_df = output["age_results"]
    gender_df = output["gender_results"]
    feature_names = output["feature_names"]

    plot_first_version_lineplot_by_age(
        combined_df=age_df,
        label_col="theme",
        selected_labels=THEMES,
        section_order=section_order,
        section_display=section_display,
        title=(
            f"{scheme_name}: Sentence-Level Theme-Section Effects on Age Group Membership\n"
            "(Controlling for Gender)"
        ),
        n_rows=2,
        n_cols=3,
        figsize=(15, 8)
    )

    plot_combined_demographic_heatmap(
        age_df=age_df,
        gender_df=gender_df,
        feature_order=feature_names,
        row_label_kind="Sentence-Level Thematic Feature",
        title=(
            f"{scheme_name}: Sentence-Level Theme Features across Demographics\n"
            "Top: log(OR) with stars, Bottom: 95% CI on log scale"
        ),
        figsize=(12, max(10, 0.45 * len(feature_names))),
        section_display_map=section_display
    )


# ============================================================
# 16. PLOT SENTENCE_DF SENTIMENT RESULTS
# ============================================================

for scheme_name, output in sentence_sentiment_outputs.items():
    print("\n" + "=" * 80)
    print(f"PLOTTING SENTENCE_DF SENTIMENT RESULTS: {scheme_name}")
    print("=" * 80)

    scheme_config = SENTENCE_SCHEMES[scheme_name]
    section_order = scheme_config["section_labels"]
    section_display = scheme_config["section_display"]

    age_df = output["age_results"]
    gender_df = output["gender_results"]
    feature_names = output["feature_names"]

    plot_first_version_lineplot_by_age(
        combined_df=age_df,
        label_col="sentiment",
        selected_labels=LINEPLOT_SENTIMENTS,
        section_order=section_order,
        section_display=section_display,
        title=(
            f"{scheme_name}: Sentence-Level Sentiment-Section Effects on Age Group Membership\n"
            "(Controlling for Gender)"
        ),
        n_rows=2,
        n_cols=3,
        figsize=(15, 8)
    )

    plot_combined_demographic_heatmap(
        age_df=age_df,
        gender_df=gender_df,
        feature_order=feature_names,
        row_label_kind="Sentence-Level Sentiment Feature",
        title=(
            f"{scheme_name}: Sentence-Level Sentiment Features across Demographics\n"
            "Top: log(OR) with stars, Bottom: 95% CI on log scale"
        ),
        figsize=(12, max(14, 0.38 * len(feature_names))),
        section_display_map=section_display
    )


# ============================================================
# 17. COMBINE AND SAVE ALL SENTENCE_DF RESULTS
# ============================================================

all_sentence_theme_age = []
all_sentence_theme_gender = []

for scheme_name, output in sentence_theme_outputs.items():
    a = output["age_results"].copy()
    g = output["gender_results"].copy()

    a["scheme"] = scheme_name
    g["scheme"] = scheme_name

    all_sentence_theme_age.append(a)
    all_sentence_theme_gender.append(g)

all_sentence_sentiment_age = []
all_sentence_sentiment_gender = []

for scheme_name, output in sentence_sentiment_outputs.items():
    a = output["age_results"].copy()
    g = output["gender_results"].copy()

    a["scheme"] = scheme_name
    g["scheme"] = scheme_name

    all_sentence_sentiment_age.append(a)
    all_sentence_sentiment_gender.append(g)


all_sentence_theme_age = (
    pd.concat(all_sentence_theme_age, ignore_index=True)
    if all_sentence_theme_age else pd.DataFrame()
)

all_sentence_theme_gender = (
    pd.concat(all_sentence_theme_gender, ignore_index=True)
    if all_sentence_theme_gender else pd.DataFrame()
)

all_sentence_sentiment_age = (
    pd.concat(all_sentence_sentiment_age, ignore_index=True)
    if all_sentence_sentiment_age else pd.DataFrame()
)

all_sentence_sentiment_gender = (
    pd.concat(all_sentence_sentiment_gender, ignore_index=True)
    if all_sentence_sentiment_gender else pd.DataFrame()
)


if SAVE_OUTPUTS:
    sentence_work.to_csv(
        os.path.join(OUTPUT_DIR, "sentence_df_with_theme_sentiment_flags_and_shuffled_bins.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    all_sentence_theme_age.to_csv(
        os.path.join(OUTPUT_DIR, "ALL_sentence_theme_age_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    all_sentence_theme_gender.to_csv(
        os.path.join(OUTPUT_DIR, "ALL_sentence_theme_gender_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    all_sentence_sentiment_age.to_csv(
        os.path.join(OUTPUT_DIR, "ALL_sentence_sentiment_age_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    all_sentence_sentiment_gender.to_csv(
        os.path.join(OUTPUT_DIR, "ALL_sentence_sentiment_gender_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )


print("\n" + "=" * 80)
print("DONE: SENTENCE_DF ANALYSIS")
print("=" * 80)

print("\nSentence-level theme schemes completed:")
print(list(sentence_theme_outputs.keys()))

print("\nSentence-level sentiment schemes completed:")
print(list(sentence_sentiment_outputs.keys()))

if SAVE_OUTPUTS:
    print("\nSaved outputs to:")
    print(OUTPUT_DIR)
    

#%%
# ============================================================
# SENTIMENT SETS FOR DIFFERENT FIGURES
# ============================================================
# Age-group sentiment figures:
# Keep these exactly as before.
# These are used for x-axis = age group plots.
# ============================================================

AGE_LINEPLOT_SENTIMENTS = [
    "Defeat",
    "Exhaustion",
    "Neutral",
    "Sadness",
    "Happiness",
    "Disappointment",
]

# Keep backward compatibility with your existing age plotting code.
# Your previous code already uses LINEPLOT_SENTIMENTS for age plots.
LINEPLOT_SENTIMENTS = AGE_LINEPLOT_SENTIMENTS


# ============================================================
# Gender sentiment figures:
# Use this different set only for x-axis = Female/Male plots.
# ============================================================

GENDER_LINEPLOT_SENTIMENTS = [
    "Defeat",
    "Hatred",
    "Neutral",
    "Happiness",
    "Anxious",
    "Disappointment",
]


# ============================================================
# 0. OUTPUT DIRECTORY
# ============================================================

GENDER_OR_FIG_DIR = os.path.join(
    OUTPUT_DIR,
    "gender_odds_ratio_plots"
)

if SAVE_OUTPUTS:
    os.makedirs(GENDER_OR_FIG_DIR, exist_ok=True)


# ============================================================
# 1. BASIC HELPER FUNCTIONS
# ============================================================

def clean_filename(text):
    """
    Make safe filenames for Windows/macOS/Linux.
    """
    text = str(text)
    text = re.sub(r"[^\w\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def save_gender_or_figure(
    fig,
    filename_base,
    outdir=GENDER_OR_FIG_DIR,
    dpi=600
):
    """
    Save both PDF and PNG versions.
    """
    if not SAVE_OUTPUTS:
        return

    filename_base = clean_filename(filename_base)

    pdf_path = os.path.join(outdir, f"{filename_base}.pdf")
    png_path = os.path.join(outdir, f"{filename_base}.png")

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def get_sig_star_safe(p):
    """
    Uses your existing get_sig_star if available, but keeps this block robust.
    """
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


# ============================================================
# 2. PREPARE GENDER ODDS-RATIO DATAFRAME
# ============================================================

def prepare_gender_or_plot_df(
    gender_df,
    label_col,
    selected_labels,
    section_order
):
    """
    Converts gender model output into a two-gender plotting dataframe.

    Input:
        One row per section-feature from the gender model.

    Output:
        Two rows per original estimate:
            Gender = Female
            Gender = Male

    Male:
        plot_or = fitted OR

    Female:
        plot_or = 1 / fitted OR

    Confidence intervals:
        Male CI:
            [CI Lower, CI Upper]

        Female reciprocal CI:
            [1 / CI Upper, 1 / CI Lower]

    This function explicitly creates Female first and Male second.
    """
    base = gender_df.copy()

    # Keep selected labels and valid sections only
    base = base[base[label_col].isin(selected_labels)].copy()
    base = base[base["section"].isin(section_order)].copy()

    # Ensure numeric values
    base["Odds Ratio"] = pd.to_numeric(
        base["Odds Ratio"],
        errors="coerce"
    )

    base["CI Lower"] = pd.to_numeric(
        base["CI Lower"],
        errors="coerce"
    )

    base["CI Upper"] = pd.to_numeric(
        base["CI Upper"],
        errors="coerce"
    )

    base["p-value"] = pd.to_numeric(
        base["p-value"],
        errors="coerce"
    )

    base = base.replace([np.inf, -np.inf], np.nan)

    base = base.dropna(
        subset=[
            "Odds Ratio",
            "CI Lower",
            "CI Upper",
            "p-value",
            label_col,
            "section"
        ]
    ).copy()

    eps = np.finfo(float).tiny

    base["Odds Ratio"] = base["Odds Ratio"].clip(lower=eps)
    base["CI Lower"] = base["CI Lower"].clip(lower=eps)
    base["CI Upper"] = base["CI Upper"].clip(lower=eps)

    # Significance stars from the original male-vs-female coefficient test
    base["sig_star"] = base["p-value"].apply(get_sig_star_safe)

    # --------------------------------------------------------
    # Female rows: reciprocal female-vs-male OR
    # --------------------------------------------------------
    female = base.copy()
    female["Gender"] = "Female"
    female["plot_or"] = 1.0 / female["Odds Ratio"]
    female["plot_ci_low"] = 1.0 / female["CI Upper"]
    female["plot_ci_high"] = 1.0 / female["CI Lower"]

    # --------------------------------------------------------
    # Male rows: fitted male-vs-female OR
    # --------------------------------------------------------
    male = base.copy()
    male["Gender"] = "Male"
    male["plot_or"] = male["Odds Ratio"]
    male["plot_ci_low"] = male["CI Lower"]
    male["plot_ci_high"] = male["CI Upper"]

    # Female first, Male second
    out = pd.concat([female, male], ignore_index=True)

    out[label_col] = pd.Categorical(
        out[label_col],
        categories=selected_labels,
        ordered=True
    )

    out["section"] = pd.Categorical(
        out["section"],
        categories=section_order,
        ordered=True
    )

    # This controls sorting and x-axis order
    out["Gender"] = pd.Categorical(
        out["Gender"],
        categories=["Female", "Male"],
        ordered=True
    )

    return out


# ============================================================
# 3. MAIN GENDER ODDS-RATIO LINE PLOT
# ============================================================

def plot_gender_or_lineplot(
    gender_df,
    label_col,
    selected_labels,
    section_order,
    section_display,
    title,
    filename_base=None,
    n_rows=2,
    n_cols=3,
    figsize=(15, 8),
    show_ci=True,
    use_log_scale=True
):
    """
    Plot gender-specific odds ratios.

    Figure structure:
        x-axis = Female, Male
        y-axis = Odds Ratio
        lines  = Section
        panels = Theme/Sentiment

    This version explicitly maps:
        Female -> x = 0
        Male   -> x = 1

    Therefore, the plotted values cannot be accidentally mislabeled.
    """
    set_constant_plot_style()

    plot_df = prepare_gender_or_plot_df(
        gender_df=gender_df,
        label_col=label_col,
        selected_labels=selected_labels,
        section_order=section_order
    )

    # Explicit x-axis mapping
    gender_order = ["Female", "Male"]

    gender_x_map = {
        "Female": 0,
        "Male": 1
    }

    plot_df["gender_x"] = plot_df["Gender"].map(gender_x_map).astype(float)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharey=True
    )

    axes = axes.flatten()

    for idx, label in enumerate(selected_labels):
        if idx >= len(axes):
            break

        ax = axes[idx]

        label_df = plot_df[plot_df[label_col] == label].copy()

        if label_df.empty:
            ax.set_visible(False)
            continue

        for section in section_order:
            sub = label_df[label_df["section"] == section].copy()

            if sub.empty:
                continue

            # Explicitly sort Female first, Male second
            sub = sub.sort_values("Gender").copy()

            x = sub["gender_x"].to_numpy(dtype=float)
            y = sub["plot_or"].to_numpy(dtype=float)

            if show_ci:
                ci_low = sub["plot_ci_low"].to_numpy(dtype=float)
                ci_high = sub["plot_ci_high"].to_numpy(dtype=float)

                yerr_lower = np.maximum(y - ci_low, 0)
                yerr_upper = np.maximum(ci_high - y, 0)

                line = ax.errorbar(
                    x,
                    y,
                    yerr=[yerr_lower, yerr_upper],
                    marker="o",
                    linewidth=2,
                    capsize=3,
                    label=section_display.get(section, section)
                )

                line_color = line.lines[0].get_color()

            else:
                line = ax.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=2,
                    label=section_display.get(section, section)
                )

                line_color = line[0].get_color()

            # Add significance stars
            for _, row_ in sub.iterrows():
                star = row_["sig_star"]
                x_val = row_["gender_x"]
                y_val = row_["plot_or"]

                if star and pd.notna(y_val):
                    ax.text(
                        x_val,
                        y_val,
                        star,
                        ha="center",
                        va="bottom",
                        fontsize=FIG_STYLE.get("star_fontsize", 11),
                        weight="bold",
                        color=line_color
                    )

        # OR = 1 means no gender contrast
        ax.axhline(
            1,
            linestyle=FIG_STYLE.get("zero_line_style", "--"),
            color=FIG_STYLE.get("zero_line_color", "gray"),
            linewidth=FIG_STYLE.get("zero_line_width", 1)
        )

        if use_log_scale:
            ax.set_yscale("log")

        ax.set_title(
            str(label),
            fontsize=13,
            fontproperties=font_prop
        )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            gender_order,
            rotation=0,
            fontsize=11,
            fontproperties=font_prop
        )

        ax.set_xlabel(
            "Gender",
            fontsize=11,
            fontproperties=font_prop
        )

        if idx % n_cols == 0:
            ax.set_ylabel(
                "Odds Ratio",
                fontsize=11,
                fontproperties=font_prop
            )

        ax.tick_params(bottom=True, left=True)

        if font_prop is not None:
            for tick_label in ax.get_yticklabels():
                tick_label.set_fontproperties(font_prop)
                tick_label.set_fontsize(11)

            for tick_label in ax.get_xticklabels():
                tick_label.set_fontproperties(font_prop)
                tick_label.set_fontsize(11)

    # Remove unused axes
    for idx in range(len(selected_labels), len(axes)):
        fig.delaxes(axes[idx])

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()

    if handles:
        fig.legend(
            handles,
            labels,
            title="Section",
            loc="center left",
            bbox_to_anchor=(0.88, 0.5),
            frameon=False,
            fontsize=11,
            title_fontsize=12
        )

    fig.suptitle(
        title,
        fontsize=14,
        weight="bold",
        y=0.96,
        fontproperties=font_prop
    )

    fig.text(
        0.5,
        0.01,
        "Female values show the reciprocal female-vs-male odds ratio, calculated as 1/OR. Male values show the fitted male-vs-female odds ratio. Models adjust for age group.",
        ha="center",
        va="bottom",
        fontsize=10,
        fontproperties=font_prop
    )

    plt.tight_layout(rect=[0, 0.04, 0.86, 0.94])

    if filename_base is not None:
        save_gender_or_figure(fig, filename_base)

    plt.show()
    plt.rcdefaults()


# ============================================================
# 4. OPTIONAL GENDER ODDS-RATIO HEATMAP
# ============================================================

def plot_gender_or_heatmap(
    gender_df,
    label_col,
    label_order,
    section_order,
    section_display,
    title,
    filename_base=None,
    figsize=(10, 6),
    use_log_color=True
):
    """
    Heatmap version.

    Rows:
        Theme/Sentiment × Gender

    Columns:
        Section

    Cell annotation:
        Odds ratio with significance stars

    Color:
        log(OR), centered at 0.
        This makes reciprocal male/female contrasts visually symmetric.
    """
    set_constant_plot_style()

    plot_df = prepare_gender_or_plot_df(
        gender_df=gender_df,
        label_col=label_col,
        selected_labels=label_order,
        section_order=section_order
    )

    plot_df["row_label"] = (
        plot_df[label_col].astype(str)
        + " | "
        + plot_df["Gender"].astype(str)
    )

    row_order = []

    for label in label_order:
        row_order.append(f"{label} | Female")
        row_order.append(f"{label} | Male")

    if use_log_color:
        plot_df["color_value"] = np.log(plot_df["plot_or"])
        cbar_label = "log(OR)"
    else:
        plot_df["color_value"] = plot_df["plot_or"]
        cbar_label = "Odds Ratio"

    heatmap_data = plot_df.pivot(
        index="row_label",
        columns="section",
        values="color_value"
    ).reindex(index=row_order, columns=section_order)

    plot_df["annot_main"] = (
        plot_df["plot_or"].round(2).astype(str)
        + plot_df["sig_star"].astype(str)
    )

    annot_data = plot_df.pivot(
        index="row_label",
        columns="section",
        values="annot_main"
    ).reindex(index=row_order, columns=section_order)

    vmax = np.nanmax(np.abs(heatmap_data.to_numpy(dtype=float)))

    if pd.isna(vmax) or vmax == 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        heatmap_data,
        ax=ax,
        annot=annot_data,
        fmt="",
        cmap=FIG_STYLE.get("cmap", "coolwarm"),
        center=0 if use_log_color else 1,
        vmin=-vmax if use_log_color else None,
        vmax=vmax if use_log_color else None,
        linewidths=FIG_STYLE.get("heatmap_linewidths", 0.5),
        linecolor=FIG_STYLE.get("heatmap_linecolor", "gray"),
        cbar_kws={"label": cbar_label}
    )

    ax.set_title(
        title,
        fontsize=14,
        weight="bold",
        fontproperties=font_prop
    )

    ax.set_xlabel(
        "Section",
        fontsize=12,
        fontproperties=font_prop
    )

    ax.set_ylabel(
        f"{label_col.capitalize()} and gender",
        fontsize=12,
        fontproperties=font_prop
    )

    ax.set_xticks(np.arange(len(section_order)) + 0.5)
    ax.set_xticklabels(
        [section_display.get(sec, sec) for sec in section_order],
        rotation=0,
        fontsize=11,
        fontproperties=font_prop
    )

    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=10,
        fontproperties=font_prop
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel(cbar_label, fontsize=12)

    if font_prop is not None:
        cbar.ax.yaxis.label.set_fontproperties(font_prop)

    fig.text(
        0.5,
        -0.02,
        "Annotations show odds ratios. Female = reciprocal female-vs-male OR; Male = fitted male-vs-female OR. Significance stars use the original coefficient test.",
        ha="center",
        va="top",
        fontsize=10,
        fontproperties=font_prop
    )

    plt.tight_layout()

    if filename_base is not None:
        save_gender_or_figure(fig, filename_base)

    plt.show()
    plt.rcdefaults()


# ============================================================
# 5. GENERIC RUNNER
# ============================================================

def plot_gender_or_figures_for_outputs(
    output_dict,
    scheme_configs,
    label_col,
    label_order,
    selected_line_labels,
    result_kind,
    output_prefix,
    make_heatmap=True
):
    """
    Runs gender OR line plots and heatmaps for every scheme in an
    output dictionary.

    output_dict examples:
        sentence_theme_outputs
        sentence_sentiment_outputs
        theme_scheme_outputs
        sentiment_scheme_outputs
    """
    if output_dict is None or len(output_dict) == 0:
        print(f"[SKIP] No output dictionary found for {output_prefix}.")
        return

    for scheme_name, output in output_dict.items():

        if scheme_name not in scheme_configs:
            print(f"[SKIP] Missing scheme config for {scheme_name}.")
            continue

        if "gender_results" not in output:
            print(f"[SKIP] No gender_results for {scheme_name}.")
            continue

        print("\n" + "=" * 80)
        print(f"PLOTTING GENDER ODDS-RATIO {result_kind.upper()} RESULTS: {scheme_name}")
        print("=" * 80)

        scheme_config = scheme_configs[scheme_name]
        section_order = scheme_config["section_labels"]
        section_display = scheme_config["section_display"]

        gender_df = output["gender_results"].copy()

        # ----------------------------------------------------
        # Main plot:
        # x = Female, Male
        # y = Odds Ratio
        # lines = Section
        # panels = Theme/Sentiment
        # ----------------------------------------------------
        plot_gender_or_lineplot(
            gender_df=gender_df,
            label_col=label_col,
            selected_labels=selected_line_labels,
            section_order=section_order,
            section_display=section_display,
            title=(
                f"{scheme_name}: Gender-Specific Odds Ratios in {result_kind}\n"
                "x = gender; y = odds ratio; lines = section"
            ),
            filename_base=(
                f"{output_prefix}_{scheme_name}_{label_col}_gender_or_lineplot"
            ),
            n_rows=2,
            n_cols=3,
            figsize=(15, 8),
            show_ci=True,
            use_log_scale=True
        )

        # ----------------------------------------------------
        # Optional heatmap:
        # useful as a supplementary table-style figure
        # ----------------------------------------------------
        if make_heatmap:
            plot_gender_or_heatmap(
                gender_df=gender_df,
                label_col=label_col,
                label_order=label_order,
                section_order=section_order,
                section_display=section_display,
                title=(
                    f"{scheme_name}: Gender-Specific Odds Ratios in {result_kind}\n"
                    "Female = reciprocal OR; Male = fitted OR"
                ),
                filename_base=(
                    f"{output_prefix}_{scheme_name}_{label_col}_gender_or_heatmap"
                ),
                figsize=(
                    max(10, len(section_order) * 2.4),
                    max(6, len(label_order) * 0.75)
                ),
                use_log_color=True
            )


# ============================================================
# 6. RUN FOR SENTENCE_DF THEME RESULTS
# ============================================================

if "sentence_theme_outputs" in globals() and "SENTENCE_SCHEMES" in globals():
    plot_gender_or_figures_for_outputs(
        output_dict=sentence_theme_outputs,
        scheme_configs=SENTENCE_SCHEMES,
        label_col="theme",
        label_order=THEMES,
        selected_line_labels=THEMES,
        result_kind="Sentence-Level Theme Features",
        output_prefix="sentence_level_theme",
        make_heatmap=True
    )

# ============================================================
# 6. RUN FOR SENTENCE_DF THEME RESULTS
# ============================================================

if "sentence_theme_outputs" in globals() and "SENTENCE_SCHEMES" in globals():
    plot_gender_or_figures_for_outputs(
        output_dict=sentence_theme_outputs,
        scheme_configs=SENTENCE_SCHEMES,
        label_col="theme",
        label_order=THEMES,
        selected_line_labels=THEMES,
        result_kind="Sentence-Level Theme Features",
        output_prefix="sentence_level_theme",
        make_heatmap=True
    )


# ============================================================
# 7. RUN FOR SENTENCE_DF SENTIMENT RESULTS
#     IMPORTANT:
#     Uses GENDER_LINEPLOT_SENTIMENTS, not LINEPLOT_SENTIMENTS.
# ============================================================

if "sentence_sentiment_outputs" in globals() and "SENTENCE_SCHEMES" in globals():
    plot_gender_or_figures_for_outputs(
        output_dict=sentence_sentiment_outputs,
        scheme_configs=SENTENCE_SCHEMES,
        label_col="sentiment",

        # Heatmap row order for gender sentiment figure
        label_order=GENDER_LINEPLOT_SENTIMENTS,

        # Lineplot panel order for gender sentiment figure
        selected_line_labels=GENDER_LINEPLOT_SENTIMENTS,

        result_kind="Sentence-Level Sentiment Features",
        output_prefix="sentence_level_sentiment",
        make_heatmap=True
    )


# ============================================================
# 8. OPTIONAL: ALSO RUN FOR ORIGINAL SECTION-LEVEL RESULTS
# ============================================================
# These only run if the objects exist from your previous section-level code.
# ============================================================

if "theme_scheme_outputs" in globals() and "SECTION_SCHEMES" in globals():
    plot_gender_or_figures_for_outputs(
        output_dict=theme_scheme_outputs,
        scheme_configs=SECTION_SCHEMES,
        label_col="theme",
        label_order=THEMES,
        selected_line_labels=THEMES,
        result_kind="Theme Features",
        output_prefix="section_level_theme",
        make_heatmap=True
    )


# ============================================================
# OPTIONAL ORIGINAL SECTION-LEVEL SENTIMENT GENDER FIGURES
#     Also uses GENDER_LINEPLOT_SENTIMENTS.
# ============================================================

if "sentiment_scheme_outputs" in globals() and "SECTION_SCHEMES" in globals():
    plot_gender_or_figures_for_outputs(
        output_dict=sentiment_scheme_outputs,
        scheme_configs=SECTION_SCHEMES,
        label_col="sentiment",

        # Heatmap row order for gender sentiment figure
        label_order=GENDER_LINEPLOT_SENTIMENTS,

        # Lineplot panel order for gender sentiment figure
        selected_line_labels=GENDER_LINEPLOT_SENTIMENTS,

        result_kind="Sentiment Features",
        output_prefix="section_level_sentiment",
        make_heatmap=True
    )


# ============================================================
# 9. QUICK NUMERIC CHECK
# ============================================================

def check_gender_or_values(
    output_dict,
    scheme_name,
    label_col,
    label_value,
    section_value,
    scheme_configs
):
    """
    Example:
        check_gender_or_values(
            output_dict=sentence_sentiment_outputs,
            scheme_name="sent_real_third",
            label_col="sentiment",
            label_value="Hatred",
            section_value="Section 2",
            scheme_configs=SENTENCE_SCHEMES
        )
    """
    gender_df = output_dict[scheme_name]["gender_results"].copy()
    section_order = scheme_configs[scheme_name]["section_labels"]

    test = prepare_gender_or_plot_df(
        gender_df=gender_df,
        label_col=label_col,
        selected_labels=[label_value],
        section_order=section_order
    )

    out = test[
        (test[label_col] == label_value)
        & (test["section"] == section_value)
    ][
        [
            "Gender",
            label_col,
            "section",
            "Odds Ratio",
            "CI Lower",
            "CI Upper",
            "plot_or",
            "plot_ci_low",
            "plot_ci_high",
            "p-value",
            "sig_star"
        ]
    ].copy()

    print("\nNumeric check:")
    print(out)

    return out


print("\n" + "=" * 80)
print("DONE: GENDER ODDS-RATIO FIGURES")
print("=" * 80)

if SAVE_OUTPUTS:
    print("\nSaved gender odds-ratio figures to:")
    print(GENDER_OR_FIG_DIR)