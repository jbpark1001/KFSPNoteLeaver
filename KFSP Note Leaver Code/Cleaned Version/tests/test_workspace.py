"""Focused verification of organization and the intentional FDR change; no registry data."""
from pathlib import Path
import ast,hashlib,importlib.util,json,sys,unittest
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
class WorkspaceChecks(unittest.TestCase):
    def test_originals_unchanged(self):
        for entry in json.loads((ROOT/'docs/source_manifest.json').read_text(encoding='utf-8')):
            for p in [ROOT/entry['archive']]:
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),entry['sha256'])
    def test_working_syntax(self):
        for folder in ['scripts','figures','text_analysis']:
            for p in (ROOT/folder).glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
    def test_fdr_preserves_estimates_and_uses_four_families(self):
        current=load('current_heatmap',ROOT/'figures/fig05_06_association_heatmaps.py')
        original=load('original_heatmap',ROOT/'reference/originals/Figure Codes/figure5-6a_theme_and_sentiment_heatmaps.py')
        rng=np.random.default_rng(820)
        meta=pd.DataFrame({'SEX':rng.choice([1,2],1200),'AGE2':rng.choice([1,2,3,4,5],1200)})
        for nfeatures,expected in [(15,(75,15)),(45,(225,45))]:
            names=[f'Section {i//(nfeatures//3)+1} Feature{i}' for i in range(nfeatures)]
            x=pd.DataFrame(rng.binomial(1,.3,(1200,nfeatures)),columns=names,dtype=float)
            old=original.fit_demographic_models(meta,x,names)
            new=current.fit_demographic_models(meta,x,names)
            np.testing.assert_allclose(new['Coefficient'],old['Coefficient'])
            np.testing.assert_allclose(new['p-value raw'],old['p-value'])
            for sex,count in zip([False,True],expected):
                rows=new['Demographic'].eq('Gender (Male)').eq(sex)
                self.assertEqual(int(rows.sum()),count)
                np.testing.assert_allclose(new.loc[rows,'p-value FDR'],multipletests(old.loc[rows,'p-value'],method='fdr_bh')[1])
            np.testing.assert_allclose(new['p-value'],new['p-value FDR'])
    def test_relative_sectioning_keeps_every_sentence(self):
        source=(ROOT/'text_analysis/11_segment_notes.py').read_text(encoding='utf-8')
        function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='sectionize_note')
        env={'np':np,'pd':pd,'safe_split_sentences':lambda x:x.split('|')}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'sectioning','exec'),env)
        for n in range(3,15):
            result=env['sectionize_note']('|'.join(map(str,range(n))),min_sentences=3)
            self.assertEqual(sum(result[f'third_{i}_n_sent'] for i in [1,2,3]),n)
            self.assertTrue(all(result[f'third_{i}_n_sent']>0 for i in [1,2,3]))
if __name__=='__main__':unittest.main(verbosity=2)
