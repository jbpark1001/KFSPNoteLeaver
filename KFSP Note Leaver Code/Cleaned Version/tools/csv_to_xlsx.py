"""Packaging bridge for R exports; does not fit or change statistical models."""
import sys
import pandas as pd

if __name__ == '__main__':
    pd.read_csv(sys.argv[1], keep_default_na=False).to_excel(sys.argv[2], index=False)
