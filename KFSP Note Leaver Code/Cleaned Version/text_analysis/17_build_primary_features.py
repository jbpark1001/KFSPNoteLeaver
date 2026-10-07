"""Build the primary feature schema from newly annotated, eligible raw notes.

This is a rebuild branch, never a replacement for published saved cohorts.
"""
from pathlib import Path
import sys,ast,json
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project import ROOT

notes=pd.read_pickle(ROOT/'derived/annotated_notes.pkl')
required=['pred_label','primary_eligible','too_short','SEX','AGE2']
required += [f'tokenizedkluekeywordsentencetransformer_{c}' for c in ['1st_section','2nd_section','3rd_section']]
required += [f'{c}_sentiment' for c in ['1st_section','2nd_section','3rd_section']]
missing=set(required)-set(notes)
if missing:raise ValueError('Rebuild stages 11–13 with primary sections first: '+str(sorted(missing)))
data=notes.loc[notes.primary_eligible & notes.pred_label.eq('raw'),required].copy()
source=ROOT/'reference/originals/Splitting raw notes (1. Gender controlled).py'
definitions={}
for node in ast.parse(source.read_text(encoding='utf-8-sig')).body:
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
        name=node.targets[0].id
        if name in ['sorry','loveandgratitude','burdensome','despair','pmaffairs']:
            definitions[name]=ast.literal_eval(node.value)
theme_names={'Sorry and Shame':'sorrysentence_presence','Love and Gratitude':'loveandgratitudesentence_presence',
             'Burden':'burdensomesentence_presence','Despair':'despairsentence_presence','Post-mortem Affairs':'pmaffairssentence_presence'}
for prefix,outcol in zip(['sorry','loveandgratitude','burdensome','despair','pmaffairs'],theme_names.values()):
    targets=set(definitions[prefix]);cols=[f'tokenizedkluekeywordsentencetransformer_{c}' for c in ['1st_section','2nd_section','3rd_section']]
    data[outcol]=[[int(any(str(token) in targets for token,score in entry)) if isinstance(entry,list) else 0
                   for entry in values] for values in data[cols].itertuples(index=False,name=None)]
data['full_theme_vector']=[[value[i] for i in range(3) for value in values]
                           for values in data[list(theme_names.values())].itertuples(index=False,name=None)]
data.to_pickle(ROOT/'derived/primary_theme_features.pkl')
data.to_pickle(ROOT/'derived/primary_sentiment_features.pkl')
metadata={'branch':'rebuilt primary, not published snapshot','notes':len(data),'theme_order':list(theme_names),
          'primary_rule':'floor thirds with remainder appended to ending','raw_filter':'new classifier pred_label == raw',
          'theme_dictionary_source':str(source),'section_sentiment_threshold':'> 0.7 (stage 13)',
          'cluster_assignments':'not frozen; new fitting and interpretation required'}
(ROOT/'derived/primary_features_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(json.dumps(metadata,indent=2))
