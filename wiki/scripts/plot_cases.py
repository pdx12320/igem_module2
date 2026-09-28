from pathlib import Path
import csv,hashlib
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from orca_plot import Theme,export
r=Path(__file__).resolve().parents[1];p=r/'data/selected_rank_cases.csv';d=list(csv.DictReader(p.open()));theme=Theme(r/'orca-figure-style.json',defaults=Path(__file__).with_name('orca-style.json'))
with theme.context():
 fig,ax=theme.figure(width=11,height=10);y=np.arange(10);pr=[int(x['predicted_rank']) for x in d];tr=[int(x['measured_rank']) for x in d]
 ax.axvspan(.5,30.5,color='#E9E8FF',alpha=.55);ax.axvspan(36.5,66.5,color='#E4F1EF',alpha=.5);ax.axhline(4.5,color='#6B6880',linestyle=':',linewidth=1.2)
 for i,(a,b) in enumerate(zip(pr,tr)):ax.plot([a,b],[i,i],color='#6B6880',linewidth=2)
 ax.scatter(pr,y,s=90,color='#2E267C',label='Predicted rank',zorder=3);ax.scatter(tr,y,s=70,color='#27877F',marker='D',label='Measured rank',zorder=4)
 for i,(a,b) in enumerate(zip(pr,tr)):
  ax.annotate(str(a),(a,i),xytext=(0,9),textcoords='offset points',ha='center',fontsize=16,color='#2E267C');ax.annotate(str(b),(b,i),xytext=(0,-21),textcoords='offset points',ha='center',fontsize=16,color='#27877F')
 ax.set(yticks=y,yticklabels=[x['target'].replace(' + ',' +\n') for x in d],xlabel='Rank among 66 constructs (lower is better)',xlim=(.5,66.5),ylim=(9.65,-.7),xticks=[1,15,30,45,66]);ax.tick_params(axis='y',length=0,labelsize=17);ax.spines['left'].set_visible(False);ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=2,borderaxespad=0)
 export(fig,r/'figures/selected_cases',theme,caption='Five higher-ranked examples (top block) and five lower-ranked examples (bottom block), excluding the original five targets. Higher cases are in both Top30 sets; lower cases are in both bottom-30 sets. Cases are selected post hoc by absolute rank error, then absolute S error and identifier. These selected agreements are illustrative, not an unbiased validation set. Predictions remain grouped OOF.',alt_text='Ten selected constructs with paired predicted and measured ranks. Top group rank errors are 0–3; bottom group errors are 0–4. All original five targets are excluded.',provenance={'source':'data/selected_rank_cases.csv','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'selection':'data/case_selection_protocol.json','n_cohort':66,'n_examples':10,'post_hoc':True,'script':'scripts/plot_cases.py'})
