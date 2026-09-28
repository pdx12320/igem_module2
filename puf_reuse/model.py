import numpy as np
from scipy.sparse import csr_matrix,hstack
from sklearn.linear_model import Ridge

AA='ACDEFGHIKLMNPQRSTVWY'
def encode(sequences,ref,logp):
 rows=[];cols=[];vals=[];scores=[]
 for k,s in enumerate(sequences):
  if len(s)!=493 or set(s)-set(AA):raise ValueError('Expected 493 canonical amino acids')
  value=0.
  for i,(x,y) in enumerate(zip(ref,s)):
   if x!=y:
    rows.extend([k,k]);cols.extend([i*20+AA.index(x),i*20+AA.index(y)]);vals.extend([-1.,1.]);value+=float(logp[i,AA.index(y)]-logp[i,AA.index(x)])
  scores.append(value)
 return csr_matrix((vals,(rows,cols)),shape=(len(sequences),493*20)),np.array(scores)

def design(onehot,score,indices,name,weight,mu,sd):
 if name=='OneHot_Ridge':return onehot[indices]
 q=csr_matrix(((score[indices]-mu)/sd*weight)[:,None])
 return q if name=='ESM2_score_Ridge' else hstack([onehot[indices],q],format='csr')

def fit(onehot,score,Y,tr,name,spec):
 fits=[]
 for j in range(3):
  ix=np.array(tr)[np.isfinite(Y[np.array(tr),j])];mu=float(score[ix].mean());sd=max(float(score[ix].std()),1e-8);weight=spec.get('score_weight',1)
  x=design(onehot,score,ix,name,weight,mu,sd);reg=Ridge(alpha=spec['alpha'],solver='lsqr',tol=1e-6);reg.fit(x,Y[ix,j]);fits.append(dict(model=reg,mean=mu,std=sd,name=name,weight=weight))
 return fits

def predict(fits,onehot,score,indices):
 return np.column_stack([f['model'].predict(design(onehot,score,indices,f['name'],f['weight'],f['mean'],f['std'])) for f in fits])
