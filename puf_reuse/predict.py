import argparse,joblib
import numpy as np
import pandas as pd
from pathlib import Path
from model import encode,predict
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--fasta',required=True);p.add_argument('--output',required=True);a=p.parse_args();assert not Path(a.output).exists();records=[];name=None;seq=[]
for line in Path(a.fasta).read_text().splitlines():
 line=line.strip()
 if not line:continue
 if line.startswith('>'):
  if name is not None:records.append((name,''.join(seq)))
  name=line[1:];seq=[]
 else:
  if name is None:raise ValueError('FASTA header required')
  seq.append(line)
if name is not None:records.append((name,''.join(seq)))
assert records and all(r[0] for r in records) and len({r[0] for r in records})==len(records)
b=joblib.load(a.model);X,score=encode([r[1] for r in records],b['reference'],b['score_logp']);v=predict(b['fits'],X,score,np.arange(len(records)));d=pd.DataFrame(dict(candidate_id=[r[0] for r in records],C295_hat=v[:,0],C388_hat=v[:,1],C871_hat=v[:,2],S_hat=v[:,1]-(v[:,0]+v[:,2])/2,ESM2_score=score,classifier_status='untrained: no negative class'));d=d.sort_values(['S_hat','candidate_id'],ascending=[False,True]);d.insert(0,'predicted_rank',range(1,len(d)+1));d.to_csv(a.output,index=False)
