from pathlib import Path
import csv,json,hashlib,argparse,shutil
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from orca_plot import Theme,export
p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args()
root=Path(__file__).resolve().parents[1];dest=a.output or root/'figures'
style=Theme(root/'orca-figure-style.json',defaults=Path(__file__).with_name('orca-style.json'))
style.config['group_styles'].update({'Hsu A':{'color':'#2E267C','marker':'o'},'Random expectation':{'color':'#E89B3C','marker':'s'},'Measured rank':{'color':'#27877F','marker':'D'}});style.save()
def read(n):return list(csv.DictReader((root/'data'/n).open(encoding='utf-8-sig')))
def save(fig,name,caption,alt,files):
 export(fig,dest/name,style,caption=caption,alt_text=alt,provenance={'sources':{f:hashlib.sha256((root/'data'/f).read_bytes()).hexdigest() for f in files},'script':'scripts/plot_results.py','cohort':'66 complete constructs; development-stage grouped OOF','missing':'No imputation','excluded_outliers':0});plt.close(fig)
with style.context():
 d=read('topk_random_comparison.csv');fig,ax=style.figure();x=np.arange(len(d));w=.32
 b=ax.bar(x-w/2,[int(r['observed_overlap']) for r in d],w,color='#2E267C',label='Hsu-inspired model A')
 c=ax.bar(x+w/2,[float(r['random_expected']) for r in d],w,color='#E89B3C',edgecolor='#171536',linewidth=.7,hatch='///',label='Random expectation')
 ax.bar_label(b,labels=[str(int(r['observed_overlap'])) for r in d],padding=8,fontsize=20);ax.bar_label(c,labels=[f"{float(r['random_expected']):.2f}" for r in d],padding=8,fontsize=18)
 ax.set(xticks=x,xticklabels=[f"Top{r['K']}" for r in d],ylabel='Overlap with measured Top-K (count)',ylim=(0,30));ax.set_yticks(range(0,31,5));ax.legend(loc='upper left')
 save(fig,'topk_random','Figure 2. Top-K recovery in development-stage cross-validation for K = 5, 15 and 30. A hit appears in both the predicted and measured Top-K sets among the same 66 constructs. Model labels show integer hit counts: 1, 6 and 20. Uniform random selection gives expected overlap K²/66: 0.38, 3.41 and 13.64. Fractional baseline values are expectations, not observed counts.','Grouped bars for Top5, Top15 and Top30. Hsu-inspired model A has integer hit counts 1, 6 and 20; random expectations are 0.38, 3.41 and 13.64.',['topk_random_comparison.csv'])
 d=read('hsu_A_oof_rankings.csv');fig,ax=style.figure();xx=np.array([float(r['S']) for r in d]);yy=np.array([float(r['S_hat']) for r in d]);mask=np.array([int(r['predicted_rank66'])<=30 for r in d]);ax.scatter(xx[~mask],yy[~mask],s=45,c='#C9C7D0',alpha=.8,label='Remaining 36');ax.scatter(xx[mask],yy[mask],s=65,c='#2E267C',alpha=.85,edgecolor='white',linewidth=.7,label='Predicted Top30');lo=min(0,xx.min(),yy.min());hi=max(xx.max(),yy.max())+.03;ax.plot([lo,hi],[lo,hi],color='#6B6880',linestyle='--',linewidth=1.5,label='Identity');ax.set(xlabel='Measured specificity, S',ylabel='Out-of-fold predicted specificity, S',xlim=(lo,hi),ylim=(lo,hi));ax.legend(loc='upper left')
 save(fig,'oof_scatter','Measured versus grouped out-of-fold predicted S for 66 complete constructs. Spearman 0.404947 and Kendall 0.264336. All observations are shown, with the predicted Top30 highlighted; the dashed line indicates equality, not a fitted trend.','Scatter plot of all 66 measured and out-of-fold predicted specificity scores, with an identity line and predicted Top30 highlighted.',['hsu_A_oof_rankings.csv'])
 d=read('five_target_summary.csv');fig,ax=style.figure();y=np.arange(5);pred=[int(r['predicted_rank']) for r in d];true=[int(r['measured_rank']) for r in d]
 ax.axvspan(.5,30.5,color='#E9E8FF',zorder=0);ax.axvline(30.5,color='#6B6880',linestyle=':',linewidth=1.2)
 for i,(pr,tr) in enumerate(zip(pred,true)):ax.plot([tr,pr],[i,i],color='#D9D7E5',linewidth=2,zorder=1)
 ax.scatter(pred,y,s=100,color='#2E267C',marker='o',label='Predicted rank',zorder=3);ax.scatter(true,y,s=80,color='#27877F',marker='D',label='Measured rank',zorder=3)
 for i,(pr,tr) in enumerate(zip(pred,true)):
  ax.annotate(str(pr),(pr,i),xytext=(0,9),textcoords='offset points',ha='center',color='#2E267C',fontsize=17)
  ax.annotate(str(tr),(tr,i),xytext=(0,-22),textcoords='offset points',ha='center',color='#27877F',fontsize=17)
 ax.set(yticks=y,yticklabels=[r['target'] for r in d],xlabel='Rank among 66 constructs (lower is better)',xlim=(.5,66.5),ylim=(4.65,-.8),xticks=[1,15,30,45,66]);ax.legend(loc='upper right');ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
 save(fig,'five_targets','Ranks of all five highlighted constructs within the 66 complete constructs. Shading marks Top30. All five constructs enter predicted Top30. P9-GNS and P9-NPS rank 4 and 15 by prediction; the other three rank 19–24. Each OOF fold excludes its own test sequence group, not all five jointly. Measurements predate this retrospective analysis.','Five paired rank markers. P8-GVE: predicted 22, measured 2; P9-NTQ: 19, 6; P9-NPS: 15, 8; P9-NSG: 24, 11; P9-GNS: 4, 13.',['five_target_summary.csv'])
print(dest)
