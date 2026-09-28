from pathlib import Path
import csv,hashlib,argparse
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from orca_plot import Theme,export
p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args();root=Path(__file__).resolve().parents[1];out=a.output or root/'figures';source=root/'data/ablation_metrics.csv';data=list(csv.DictReader(source.open()));names=['Position_only_Ridge_A','OneHot_Ridge_A','ESM2_score_zero_shot','ESM2_score_Ridge_A','Hsu_ESM2_OneHot_Ridge_A'];labels=['Position-only\n+ Ridge','Amino-acid one-hot\n+ Ridge','ESM-2 zero-shot','ESM-2 score\n+ Ridge','ESM-2 + one-hot\n+ Ridge'];colors=['#6B6880','#27877F','#E89B3C','#9D9CF4','#2E267C'];d=[next(r for r in data if r['model']==n and r['subset']=='all') for n in names];theme=Theme(root/'orca-figure-style.json',defaults=Path(__file__).with_name('orca-style.json'))
for name,c in zip(names,colors):theme.config['group_styles'][name]={'color':c,'marker':'o'}
theme.save();prov={'source':'data/ablation_metrics.csv','sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'n':66,'evaluation':'grouped OOF; same inner A rule for supervised models','script':'scripts/plot_ablations.py','uncertainty':'not estimated','reporting':'Top30 chosen retrospectively'}
with theme.context():
 for slug,key,xlabel,lim in [('ablation_spearman','Spearman','Spearman correlation',( -.38,.54)),('ablation_top30','Top30_overlap','Overlap with measured Top30 (count)',(0,30))]:
  fig,ax=theme.figure(width=11,height=7);values=[float(r[key]) for r in d];y=np.arange(5);ax.barh(y,values,color=colors,height=.62);ax.set(yticks=y,yticklabels=labels,xlabel=xlabel,xlim=lim);ax.invert_yaxis();ax.tick_params(axis='y',length=0,labelsize=18);ax.spines['left'].set_visible(False)
  for i,v in enumerate(values):ax.text(v+(.015 if key=='Spearman' else .35) if v>=0 else v-.015,i,f'{v:.3f}' if key=='Spearman' else f'{int(v)}/30',ha='left' if v>=0 else 'right',va='center',fontsize=18)
  if key=='Spearman':
   ax.axvline(0,color='#6B6880',linewidth=1);caption='Matched feature ablation on 66 complete constructs. Spearman correlations: position-only Ridge 0.154702; amino-acid one-hot Ridge 0.291848; ESM-2 zero-shot −0.265421; score-only Ridge 0.097756; combined 0.404947. Identical grouped CV and A selection for supervised methods. No uncertainty intervals estimated.';alt='Horizontal bars show combined model highest correlation, one-hot second, position-only third, score-only Ridge fourth, and negative zero-shot correlation.'
  else:
   ax.axvline(900/66,color='#171536',linestyle='--',linewidth=1.7,label='Random expectation: 13.64');ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),borderaxespad=0);caption='Top30 overlaps for the same 66 constructs: position-only Ridge 14, amino-acid one-hot Ridge 16, ESM-2 zero-shot 9, score-only Ridge 16, and combined 20. The dashed line marks the uniform random expectation 30²/66 = 13.64. Top30 was chosen retrospectively for reporting; no model selection uses outer labels.';alt='Five bars report Top30 hits 14, 16, 9, 16 and 20. Random expectation is 13.64.'
  export(fig,out/slug,theme,caption=caption,alt_text=alt,provenance=prov);plt.close(fig)
print('Exported two ablation figures')
