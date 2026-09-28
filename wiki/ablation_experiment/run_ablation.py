from pathlib import Path
import argparse,json,time,math,hashlib,platform
import numpy as np,pandas as pd,scipy,sklearn
from scipy.stats import spearmanr,kendalltau
from scipy.sparse import csr_matrix
from model import encode,fit,predict
p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=66);a=p.parse_args();root=Path(__file__).resolve().parent;inp=root/'inputs';out=root/'results';out.mkdir(exist_ok=True)
cfg=json.loads((root/'protocol.json').read_text());mf=pd.read_csv(inp/'construct_manifest.csv');ta=pd.read_csv(inp/'construct_targets.csv');assert mf.construct_id.tolist()==ta.construct_id.tolist();Y=ta[['C295','C388','C871']].to_numpy();ref=json.loads((inp/'reference.json').read_text())['reference'];logp=np.load(inp/'reference_masked_logp.npy');onehot,score=encode(mf.puf_sequence.tolist(),ref,logp);pos=csr_matrix(np.array([[a!=b for a,b in zip(s,ref)] for s in mf.puf_sequence],float));groups=mf.leakage_group.to_numpy();folds=json.loads((inp/'splits.json').read_text());assert len(mf)==81 and np.isfinite(Y).all(1).sum()==66 and len(ref)==493
assert (pos.sum(1).A.ravel()==onehot.getnnz(1)/2).all()
def S(y):return y[:,1]-(y[:,0]+y[:,2])/2
def obj(y,q,ids):
 t=S(y);e=S(q);rho=float(spearmanr(t,e).statistic);rho=rho if np.isfinite(rho) else -1.;k=math.ceil(15/66*len(t));sel=np.lexsort((ids,-e))[:k];return [rho,float(t[sel].mean()),-float(np.mean((y-q)**2))]
start=time.time();rows=[];mnames=['Position_only_Ridge','OneHot_Ridge','ESM2_score_Ridge','Hsu_ESM2_OneHot_Ridge']
for k,f in enumerate(folds[:a.limit]):
 tr=np.array(f['train']);te=np.array(f['test']);assert not set(groups[tr])&set(groups[te]);assert set(tr)==set(np.flatnonzero(~np.isin(groups,groups[te])))
 path=out/f'fold_{k+1:03d}.json'
 if path.exists():rr=json.loads(path.read_text())
 else:
  rr={}
  for name in mnames:
   X=pos if name=='Position_only_Ridge' else onehot;key='OneHot_Ridge' if name=='Position_only_Ridge' else name;candidates=[]
   for alpha in cfg['alphas']:
    for weight in (cfg['score_weights'] if name=='Hsu_ESM2_OneHot_Ridge' else [1]):
     spec=dict(alpha=alpha,score_weight=weight);vals=[]
     for sp in f['inner']:
      ti=np.array(sp['train81']);vi=np.array(sp['validation66']);assert not set(groups[ti])&set(groups[vi]);assert set(ti)<=set(tr) and set(vi)<=set(tr)
      q=predict(fit(X,score,Y,ti,key,spec),X,score,vi);vals.append(obj(Y[vi],q,vi))
     candidates.append(dict(spec=spec,metrics=np.mean(vals,axis=0).tolist()))
   best=max(candidates,key=lambda x:tuple(x['metrics']));q=predict(fit(X,score,Y,tr,key,best['spec']),X,score,te);assert np.isfinite(q).all();rr[name]=dict(selected=best,candidates=candidates,predictions=q.tolist())
  path.write_text(json.dumps(rr,indent=2))
 for name in mnames:
  for i,q in zip(te,rr[name]['predictions']):
   r=dict(construct_id=mf.iloc[i].construct_id,batch=mf.iloc[i].batch,double_repeat=len(json.loads(mf.iloc[i].repeat_position_set))>1,model=name+'_A',fold=k+1,S=float(S(Y[i:i+1])[0]),S_hat=float(S(np.array([q]))[0]))
   for j,c in enumerate(['C295','C388','C871']):r[c]=Y[i,j];r[c+'_hat']=q[j]
   rows.append(r)
 print('fold',k+1,'seconds',round(time.time()-start,2),flush=True)
if a.limit<66:print('PILOT PASS');raise SystemExit
for i in np.flatnonzero(np.isfinite(Y).all(1)):rows.append(dict(construct_id=mf.iloc[i].construct_id,batch=mf.iloc[i].batch,double_repeat=len(json.loads(mf.iloc[i].repeat_position_set))>1,model='ESM2_score_zero_shot',fold='',S=float(S(Y[i:i+1])[0]),S_hat=score[i]))
df=pd.DataFrame(rows);metrics=[];ranked=[]
for name,d in df.groupby('model'):
 assert len(d)==66;d=d.sort_values(['S_hat','construct_id'],ascending=[False,True]);d['predicted_rank66']=range(1,67);d['measured_rank66']=d.S.rank(ascending=False,method='min').astype(int);ranked.append(d)
 for subset in ['all','batch1','batch2','double_repeat']:
  e=d if subset=='all' else d[d.double_repeat] if subset=='double_repeat' else d[d.batch.eq(subset)];n=len(e);m=dict(model=name,subset=subset,n=n,Spearman=spearmanr(e.S,e.S_hat).statistic,Kendall=kendalltau(e.S,e.S_hat).statistic)
  if n>=30:
   pred=e.head(30);truth=e.sort_values(['S','construct_id'],ascending=[False,True]).head(30);h=len(set(pred.construct_id)&set(truth.construct_id));m.update(Top30_overlap=h,Top30_hit_rate=h/30,Top30_random_expected=900/n,Top30_enrichment=h/(900/n),Top30_mean_S=pred.S.mean())
  metrics.append(m)
ranked=pd.concat(ranked);ranked.to_csv(out/'oof_ablation.csv',index=False);pd.DataFrame(metrics).to_csv(out/'ablation_metrics.csv',index=False)
old=pd.read_csv(inp/'oof_all_models.csv');diff={}
for name in ['OneHot_Ridge_A','ESM2_score_Ridge_A','Hsu_ESM2_OneHot_Ridge_A','ESM2_score_zero_shot']:
 d=ranked[ranked.model.eq(name)].merge(old[old.model.eq(name)],on='construct_id',suffixes=('_new','_old'));delta=float(abs(d.S_hat_new-d.S_hat_old).max());assert delta<1e-7,(name,delta);diff[name]=delta
(out/'validation.json').write_text(json.dumps(dict(seconds=time.time()-start,folds=66,OOF_rows=len(ranked),group_leakage_checks=True,old_predictions_max_difference=diff,software=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__)),indent=2));print(pd.DataFrame(metrics).query("subset=='all'").to_string(index=False))
