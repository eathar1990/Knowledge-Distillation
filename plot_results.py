"""Generate figures and CSV tables strictly from measured JSON results."""
import argparse,json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate(directory='results'):
    root=Path(directory); root.mkdir(exist_ok=True)
    rows=[]
    for d in ('phishing','har'):
        source=next((p for p in (root/f'{d}_tuned.json',root/f'{d}_full.json') if p.exists()),None)
        if not source:continue
        data=json.loads(source.read_text())
        for model,metrics in data.get('metrics',{}).items():
            rows.append({'dataset':d,'model':model,**{k:v for k,v in metrics.items() if isinstance(v,(int,float,str))}})
    if not rows:raise FileNotFoundError('No measured experiment JSON results; run training first')
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with (root/'metrics_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    for metric in ('accuracy','macro_f1','agreement','kl_teacher_student'):
        ds=['phishing','har'];models=['teacher','pce_hdmr','ebm']
        fig,ax=plt.subplots(figsize=(8,4.5));x=np.arange(len(ds));width=.25
        for j,model in enumerate(models):
            vals=[next((r.get(metric,np.nan) for r in rows if r['dataset']==d and r['model']==model),np.nan) for d in ds]
            ax.bar(x+(j-1)*width,vals,width,label=model)
        ax.set_xticks(x,ds);ax.set_ylabel(metric);ax.legend();fig.tight_layout()
        fig.savefig(root/f'{metric}.png',dpi=200);fig.savefig(root/f'{metric}.pdf');plt.close(fig)
    return len(rows)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--results',default='results');args=ap.parse_args()
    print('Exported rows:',generate(args.results))
