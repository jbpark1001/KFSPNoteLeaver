from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project import ROOT, input_path, output_path, load_frame
import numpy as np
import pandas as pd
import kss
from tqdm import tqdm
import random
from section_rules import primary_thirds



print("Loading data...")

kfsppororo=load_frame(ROOT / "derived/scored_notes.pkl")

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
from functools import lru_cache

# Reuse deterministic splits across thirds, quartiles, long-note and sentence datasets.
@lru_cache(maxsize=None)
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

# Primary analyses use the manuscript's remainder-to-ending allocation. The
# midpoint third/quartile/shuffle columns above remain distinct robustness bins.
primary = [primary_thirds(safe_split_sentences(t)) for t in kfsppororo['preprocessednotes']]
kfsppororo[['1st_section','2nd_section','3rd_section']] = pd.DataFrame(primary,index=kfsppororo.index)
kfsppororo['too_short'] = kfsppororo['third_total_n_sent'].lt(3)
kfsppororo['primary_eligible'] = kfsppororo['pred_label'].eq('raw') & ~kfsppororo['too_short']



kfsppororo.to_pickle(ROOT / "derived/sectioned_notes.pkl")
sentence_df.to_pickle(ROOT / "derived/sentences.pkl")
