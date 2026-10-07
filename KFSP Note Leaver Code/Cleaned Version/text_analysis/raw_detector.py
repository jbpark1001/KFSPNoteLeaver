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
kfsppororo= pd.read_excel(r'C:\Users\<user>\kfsp_sectioned.xlsx')
kfsp=pd.read_excel(r'G:\KFSP\Raw Data\KFSPdatacopy.xlsx')
allnotes = kfsp[~kfsp["NOTE_CONTENTDTL"].isna()].copy()

# Uncomment and edit as needed:
# kfsppororo = pd.read_excel(r'C:\Users\<user>\kfsppororosplit.xlsx')
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

    def fit(self, df: pd.DataFrame, validate_weak_holdout: bool = False):
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
                if validate_weak_holdout:
                    from sklearn.base import clone
                    # A separate pipeline learns vocabulary, features, calibration
                    # and coefficients only from Xtr. The final all-data model
                    # above is preserved for primary scoring.
                    validation_pipe = clone(self.pipe)
                    validation_pipe.fit(Xtr, ytr)
                    acc = float((validation_pipe.predict(Xte) == yte).mean())
                    self.weak_holdout_report_ = {'train_n':len(Xtr),'test_n':len(Xte),
                                                 'accuracy_vs_weak_labels':acc,
                                                 'source':'independent training split; not manual-label validation'}
                    print(f"[Info] Held-out accuracy vs. weak labels: {acc:.3f} (weak-label validation only)")
                else:
                    acc = (self.pipe.predict(Xte) == yte).mean()
                    print(f"[Info] In-sample weak-label agreement on a random subset: {acc:.3f}; not held-out validation")
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
