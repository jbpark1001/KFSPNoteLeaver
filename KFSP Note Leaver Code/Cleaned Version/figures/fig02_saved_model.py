"""Render Fig. 2 from saved model tables without implying a newly fitted model."""
from pathlib import Path
import importlib.util
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project import ROOT,input_path,output_path
or_path=input_path('existing_odds_ratios')
var_path=input_path('existing_factor_tests')
if or_path.parent != var_path.parent:
    raise ValueError('The saved tables must be in the same directory for the original plotting interface.')
if or_path.name != 'or_table_Cut_2_DF_50Columns.xlsx' or var_path.name != 'var_tests_Cut_2_DF_50Columns.xlsx':
    raise ValueError('Saved-table names must match the Cut_2 interface; use the refitted-model route otherwise.')
spec=importlib.util.spec_from_file_location('demographic_plot',ROOT/'scripts/05_plot_demographic_results.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.TABLE_DIR=or_path.parent
module.FIGURE_DIR=output_path('figures/fig02_saved/.keep').parent
module.main()
