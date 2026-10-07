"""Run a named manuscript analysis, with explicit inputs and local logs."""
import argparse
import ast
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
from project import ROOT, input_path

STAGES = {
    '01-clean-demographics': ('scripts/01_reorganize_demographic_data.py', ['demographic_source']),
    '02-select-subsets': ('scripts/02_select_predictor_subsets.py', ['@derived/cleaned_demographic_data.xlsx']),
    '03-compare-models': ('scripts/03_compare_logistic_models.R', ['@derived/cuts/Cut_0_DF_32Columns.xlsx']),
    '04-final-model': ('scripts/04_final_logistic_regression.R', ['@derived/cuts/Cut_2_DF_50Columns.xlsx']),
    '05-demographic-plots': ('scripts/05_plot_demographic_results.py', ['@outputs/tables/or_table_Cut_2_DF_50Columns.xlsx','@outputs/tables/var_tests_Cut_2_DF_50Columns.xlsx']),
    '10-classify': ('text_analysis/10_classify_notes.py', ['eligible_notes']),
    '11-segment': ('text_analysis/11_segment_notes.py', ['@derived/scored_notes.pkl']),
    '12-keywords': ('text_analysis/12_extract_keywords.py', ['@derived/sectioned_notes.pkl','@derived/sentences.pkl']),
    '13-sentiments': ('text_analysis/13_extract_sentiments.py', ['@derived/keyword_notes.pkl','@derived/keyword_sentences.pkl','kote_checkpoint']),
    '14-section-robustness': ('text_analysis/14_section_robustness.py', ['section_annotations']),
    '15-sentence-robustness': ('text_analysis/15_sentence_robustness.py', ['section_annotations','sentence_annotations']),
    '16-transition-tables': ('text_analysis/16_transition_tables.py', ['raw_features','sentiment_features']),
    '17-build-primary-features': ('text_analysis/17_build_primary_features.py', ['@derived/annotated_notes.pkl']),
    'fig01a': ('figures/fig01a_geography.py', ['registry','population','shapefile']),
    'fig01b': ('figures/fig01b_monthly.py', ['registry']),
    'fig02-saved': ('figures/fig02_saved_model.py', ['existing_odds_ratios','existing_factor_tests']),
    'fig03': ('figures/fig03_lexical_profile.py', ['noun_counts','verb_counts']),
    'fig04': ('figures/fig04_cluster_centroids.py', ['theme_clusters','sentiment_clusters']),
    'fig05-06': ('figures/fig05_06_association_heatmaps.py', ['raw_features','sentiment_features']),
    'fig05-06-lines': ('figures/fig05_06_association_lines.py', ['@outputs/figures/fig05_06/figure6a_theme_heatmap_estimates.csv','@outputs/figures/fig05_06/figure6b_sentiment_heatmap_estimates.csv']),
    'fig07': ('figures/fig07_transitions.py', ['theme_clusters','sentiment_clusters']),
    'ed01': ('figures/ed01_monthly_counts.py', ['registry']),
    'ed02': ('figures/ed02_pca_projections.py', ['theme_clusters','sentiment_clusters']),
    'ed03-06': ('figures/ed03_06_robustness_panels.py', ['section_results','sentence_results']),
    'ed03e': ('figures/ed03e_real_vs_shuffled.py', ['section_results','sentence_results']),
    'ed07-10': ('figures/ed07_10_descriptive_proportions.py', ['proportion_summary']),
}

def needed(key):
    if key.startswith(('@outputs/figures/', '@outputs/tables/transitions/')) and os.environ.get('MANUSCRIPT_INPUT_PROFILE')=='rebuilt':
        return ROOT/'outputs/rebuilt'/key[len('@outputs/'):]
    return ROOT / key[1:] if key.startswith('@') else input_path(key)

def preflight():
    results=[]
    for name,(script,keys) in STAGES.items():
        missing=[str(needed(k)) for k in keys if not needed(k).exists()]
        results.append({'stage':name,'script':script,'inputs_present':not missing,'missing':missing})
        print(f"{name:26} {'INPUTS PRESENT' if not missing else 'MISSING INPUTS'}")
        for path in missing: print('  '+path)
    for directory in ('scripts','figures','text_analysis'):
        for path in (ROOT/directory).glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
    (ROOT/'docs/preflight.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('Python syntax checks passed. Input presence does not establish statistical reproducibility.')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['list','check',*STAGES])
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--inputs', choices=['saved','rebuilt'], default='saved',
                        help='Published saved snapshots or explicitly rebuilt features/robustness outputs')
    args,extra=parser.parse_known_args()
    os.environ['MANUSCRIPT_INPUT_PROFILE']=args.inputs
    if args.stage=='list':
        for name,(script,_) in STAGES.items(): print(f'{name:26} {script}')
        return
    for folder in ('docs','derived','outputs/logs','outputs/tables','outputs/figures'):
        (ROOT/folder).mkdir(parents=True,exist_ok=True)
    if args.stage=='check':
        preflight(); return
    script,keys=STAGES[args.stage]
    runtime=json.loads((ROOT/'config/runtime.json').read_text(encoding='utf-8'))
    exe=runtime['rscript' if script.endswith('.R') else 'python']
    command=[exe,str(ROOT/script),*extra]
    print(subprocess.list2cmdline(command),flush=True)
    if args.dry_run:return
    missing=[str(needed(k)) for k in keys if not needed(k).exists()]
    if missing: raise SystemExit('Missing inputs:\n'+'\n'.join(missing)+'\nSee config/paths.json and docs/REPRODUCIBILITY.md.')
    if not Path(exe).is_file():raise SystemExit('Runtime not found; update config/runtime.json.')
    env=os.environ.copy();env.update(MPLBACKEND='Agg',PYTHONUTF8='1',PYTHONIOENCODING='utf-8',MANUSCRIPT_PYTHON=runtime['python'],MANUSCRIPT_PROJECT_ROOT=str(ROOT))
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    # R reads expressions incrementally. Freeze its source before execution so
    # edits in the working file cannot corrupt a long-running profile-CI job.
    if script.endswith('.R'):
        snapshot=ROOT/'outputs/logs/source_snapshots'/f'{stamp}_{Path(script).name}'
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/script,snapshot)
        command[1]=str(snapshot)
    log=ROOT/'outputs/logs'/f'{stamp}_{args.stage}.log'
    with log.open('w',encoding='utf-8') as handle:
        handle.write('Input profile: '+args.inputs+'\n');handle.flush()
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
    print('Log:',log)
    if result.returncode:raise SystemExit(f'Stage failed ({result.returncode}). Read the log above.')
    print('Completed',args.stage)

if __name__=='__main__':main()
