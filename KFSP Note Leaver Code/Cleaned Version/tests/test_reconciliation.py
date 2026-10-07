from pathlib import Path
import sys,importlib.util
import numpy as np
import pytest
import statsmodels.api as sm
import pandas as pd

P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P/'text_analysis'))
from section_rules import primary_thirds
from inference_checks import require_valid_logit

def test_primary_remainder_is_appended_to_ending():
    assert primary_thirds(['a','b','c','d','e']) == ('a','b','c d e')
    assert primary_thirds(['a','b','c','d','e','f','g','h']) == ('a b','c d','e f g h')

def test_sentiment_vector_values_have_correct_section_and_state_labels():
    spec=importlib.util.spec_from_file_location('centroid_labels',P/'figures/fig04_cluster_centroids.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    matrix=np.concatenate([m.section_sentiment_vector({'labels':[m.SENTIMENT_SOURCE_LABELS[s]]})
                           for s in ['Despair','Happiness','Neutral']])
    active=[m.SENTIMENT_PCA_COLUMNS[i] for i in np.flatnonzero(matrix)]
    assert active==['Section 1 Despair','Section 2 Happiness','Section 3 Neutral']

def test_unfinished_logistic_fit_cannot_supply_inference():
    rng=np.random.default_rng(3)
    x=sm.add_constant(rng.normal(size=(2000,2)))
    y=rng.binomial(1,1/(1+np.exp(-(x@np.array([.1,1.3,-.8])))))
    unfinished=sm.Logit(y,x).fit(disp=False,maxiter=1)
    with pytest.raises(ValueError,match='did not converge'):
        require_valid_logit(unfinished)
    complete=sm.Logit(y,x).fit(disp=False,maxiter=100)
    require_valid_logit(complete)

def test_optional_weak_holdout_fits_an_independent_pipeline():
    from raw_detector import SummarizationDetector
    notes=pd.DataFrame({'NOTE_CONTENTDTL':
        [f'엄마 아빠 고마워요 사랑해요 나는 미안해요 내 마지막 부탁 잘 살아 {i}' for i in range(40)] +
        [f'수사관이 유서를 발견했다고 함. 내용에는 다음과 같이 기재되어 있음. 가족에게 미안하다고 함 {i}' for i in range(40)]})
    detector=SummarizationDetector(text_col='NOTE_CONTENTDTL',min_rule_conf=.34,use_length_feature=False)
    detector.fit(notes,validate_weak_holdout=True)
    report=detector.weak_holdout_report_
    assert report['train_n']+report['test_n'] == len(detector._weak_label_dataframe(notes))
    assert report['test_n']>0 and report['train_n']<len(notes)
    assert 0<=report['accuracy_vs_weak_labels']<=1
