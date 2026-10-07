"""Plot the same FDR-adjusted estimates used for the main heatmaps; do not refit."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from project import ROOT,output_path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

for domain,stem in [('theme','figure6a_theme'),('sentiment','figure6b_sentiment')]:
    table=pd.read_csv(output_path(f'figures/fig05_06/{stem}_heatmap_estimates.csv'))
    for comparison in ('age','sex'):
        labels=list(table['Feature label'].drop_duplicates())
        ncols=3;nrows=int(np.ceil(len(labels)/ncols))
        fig,axes=plt.subplots(nrows,ncols,figsize=(15,3.3*nrows),squeeze=False)
        for ax,label in zip(axes.flat,labels):
            for offset,section in enumerate(['Section 1','Section 2','Section 3']):
                rows=table[(table['Feature label']==label)&(table['Section']==section)].set_index('Demographic')
                order=[f'Age{i}' for i in range(1,6)] if comparison=='age' else ['Gender (Male)']
                rows=rows.loc[order];x=np.arange(len(order))+(offset-1)*.09
                y=rows['Coefficient'].to_numpy()
                ci=np.stack([y-rows['CI Lower'].to_numpy(),rows['CI Upper'].to_numpy()-y])
                ax.errorbar(x,y,yerr=ci,marker='o',capsize=2,label=['Opening','Middle','Ending'][offset])
                for xx,yy,p in zip(x,y,rows['p-value FDR']):
                    star='***' if p<.001 else '**' if p<.01 else '*' if p<.05 else ''
                    ax.annotate(star,(xx,yy),xytext=(0,6),textcoords='offset points',ha='center',fontsize=8)
            ax.axhline(0,color='0.6',lw=.8);ax.set_title(label)
            ax.set_xticks(np.arange(len(order)),['≤18','19–34','35–49','50–64','≥65'] if comparison=='age' else ['Male vs female'])
            ax.set_ylabel('Log odds ratio (95% CI)')
        for ax in axes.flat[len(labels):]:ax.set_visible(False)
        axes.flat[0].legend(frameon=False,fontsize=8)
        fig.suptitle(f'{domain.title()} associations by {comparison}; BH-FDR significance',fontsize=14)
        fig.tight_layout()
        for ext in ['pdf','png']:
            fig.savefig(output_path(f'figures/fig05_06_lines/{domain}_{comparison}.{ext}'),dpi=300,bbox_inches='tight')
        plt.close(fig)
