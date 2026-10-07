from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project import ROOT, input_path, output_path, load_frame
import torch.nn as nn
from transformers import ElectraModel, AutoTokenizer
import torch
import pandas as pd

kfsppororo=load_frame(ROOT / "derived/keyword_notes.pkl")
sentence_df=load_frame(ROOT / "derived/keyword_sentences.pkl")

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

class KOTEtagger(nn.Module):
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
trained_model.load_state_dict(torch.load(input_path("kote_checkpoint"), map_location=device)) # <All keys matched successfully>라는 결과가 나오는지 확인!



trained_model.eval()

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





@torch.no_grad()
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

section_cols = ['1st_section','2nd_section','3rd_section'] + third_cols + quart_cols + third5_cols + shuf_cols

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

kfsppororo.to_pickle(ROOT / "derived/annotated_notes.pkl")
sentence_df.to_pickle(ROOT / "derived/annotated_sentences.pkl")
