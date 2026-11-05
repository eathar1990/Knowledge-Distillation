"""Dataset diagnostics and train/validation/test split audits."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_experiments import load_dataset,split_data

def audit(dataset,output=None):
    X,y,batches=load_dataset(dataset)
    tr,va,te,method=split_data(X,y,batches)
    sets=[set(tr),set(va),set(te)]
    assert all(not sets[i].intersection(sets[j]) for i in range(3) for j in range(i+1,3))
    assert len(set.union(*sets))==len(y)
    info={'dataset':dataset,'rows':len(y),'features':X.shape[1],
          'split':method,'missing_values':int(X.isna().sum().sum()),
          'duplicate_feature_rows':int(X.duplicated().sum()),
          'class_counts':y.value_counts().to_dict(),
          'split_class_counts':{name:y.iloc[ix].value_counts().to_dict()
              for name,ix in [('train',tr),('validation',va),('test',te)]}}
    if batches is not None:
        info['batch_class_counts']={str(b):y.iloc[np.flatnonzero(batches==b)].value_counts().to_dict()
            for b in sorted(set(batches))}
    if output:
        Path(output).parent.mkdir(parents=True,exist_ok=True)
        Path(output).write_text(json.dumps(info,indent=2,default=str))
    return info

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['phishing','har','all'],default='all')
    a=ap.parse_args()
    for d in (['phishing','har'] if a.dataset=='all' else [a.dataset]):
        print(json.dumps(audit(d,f'results/{d}_audit.json'),indent=2,default=str))
