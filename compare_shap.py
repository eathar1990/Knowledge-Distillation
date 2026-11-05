"""Actual, like-for-like permutation SHAP on teacher and two distilled students.
Run AFTER tune_hyperparameters.py; reads selected configs, refits on train only.
All models explained on the SAME full-feature input, background, samples and
scalar output: log P(class=k | x). Computationally expensive; use --samples 12 first.
"""
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import log_softmax
from scipy.stats import spearmanr
from sklearn.metrics import log_loss
from xgboost import XGBClassifier
from run_experiments import (load_dataset,split_data,SEED,OUT,
                             PCEStudent,EBMStudent,select_features)
from tune_hyperparameters import fit_ebm

def build(dataset):
    config_path=OUT/f'{dataset}_tuned.json'
    if not config_path.exists():
        raise FileNotFoundError(f'{config_path} missing. Run tune_hyperparameters.py --dataset {dataset} first.')
    saved=json.loads(config_path.read_text(encoding='utf-8'))
    X,y,batches=load_dataset(dataset)
    X=X.replace([np.inf,-np.inf],np.nan)
    tr,va,te,split=split_data(X,y,batches)
    med=X.iloc[tr].median().fillna(0)
    X=X.fillna(med).fillna(0)
    classes=np.sort(y.unique()); enc=np.searchsorted(classes,y.to_numpy()); n=len(classes)
    tc=saved['best_hyperparameters']['teacher']
    teacher=XGBClassifier(**tc,subsample=.9,colsample_bytree=.85,
        objective='multi:softprob' if n>2 else 'binary:logistic',
        eval_metric='mlogloss' if n>2 else 'logloss',random_state=SEED,
        n_jobs=4,tree_method='hist')
    teacher.fit(X.iloc[tr],enc[tr],verbose=False)
    trainp=teacher.predict_proba(X.iloc[tr])
    configs=saved['best_hyperparameters']
    student_models={}; chosen={}
    for key in ('pce_hdmr','ebm'):
        cfg=configs[key]
        idx=select_features(X.iloc[tr],teacher,int(cfg['n_features']))
        # Verify exact selected feature names saved in tuned results.
        recorded=saved.get('metrics',{}).get(key,{}).get('selected_features')
        names=list(X.columns[idx])
        if recorded is not None and names!=recorded:
            raise ValueError(f'{key}: selected features changed from tuned run. Check versions/data.')
        A=X.iloc[tr,idx].to_numpy(dtype=float)
        if key=='pce_hdmr':
            model=PCEStudent(alpha=cfg['alpha']).fit(A,trainp,names)
        else:
            params={k:cfg[k] for k in ('learning_rate','max_rounds','interactions')}
            model=fit_ebm(A,trainp,names,params)
        student_models[key]=model;chosen[key]=idx
    return X,y,classes,tr,te,teacher,student_models,chosen,saved

def run(dataset,samples,background,seed,class_index):
    import shap
    X,y,classes,tr,te,teacher,students,selected,saved=build(dataset)
    rng=np.random.default_rng(seed)
    eval_ids=np.sort(rng.choice(te,size=min(samples,len(te)),replace=False))
    bg_ids=np.sort(rng.choice(tr,size=min(background,len(tr)),replace=False))
    # Use the same background and rows for all models; full original feature space.
    B=X.iloc[bg_ids].to_numpy(dtype=float)
    E=X.iloc[eval_ids].to_numpy(dtype=float)
    d=X.shape[1]
    k=int(class_index) if class_index is not None else (1 if len(classes)==2 else 0)
    if not 0<=k<len(classes):raise ValueError('class-index out of range')
    names=list(X.columns)
    all_models={'teacher':teacher,**students}
    outputs={}; values={}; residuals={}
    for name,model in all_models.items():
        idx=selected.get(name)
        def scalar_logprob(arr,model=model,idx=idx):
            arr=np.asarray(arr,dtype=float)
            if arr.ndim==1:arr=arr.reshape(1,-1)
            features=arr if idx is None else arr[:,idx]
            prob=np.asarray(model.predict_proba(features),dtype=float)
            return np.log(np.clip(prob[:,k],1e-9,1))
        # Actual SHAP permutation attributions; background-based interventional
        # masking. Explains log class probability, NOT teacher raw tree margins.
        masker=shap.maskers.Independent(B,max_samples=len(B))
        explainer=shap.Explainer(scalar_logprob,masker,algorithm='permutation',
                                feature_names=names,seed=seed)
        explanation=explainer(E,max_evals=2*d+1,batch_size=128,silent=True)
        v=np.asarray(explanation.values,dtype=float)
        base=np.asarray(explanation.base_values,dtype=float).reshape(-1)
        predicted=scalar_logprob(E)
        err=float(np.max(np.abs(base+v.sum(axis=1)-predicted)))
        values[name]=v
        residuals[name]=err
        outputs[name]={'mean_abs_shap':np.mean(np.abs(v),axis=0).tolist(),
                       'mean_signed_shap':np.mean(v,axis=0).tolist(),
                       'local_additivity_max_abs_error':err}
        print(f'{dataset}: {name} SHAP complete; additivity error {err:.3g}',flush=True)
    def comparison(a,b):
        ia=np.mean(np.abs(values[a]),axis=0);ib=np.mean(np.abs(values[b]),axis=0)
        rho=spearmanr(ia,ib).statistic
        topa=set(np.argsort(ia)[-10:]);topb=set(np.argsort(ib)[-10:])
        return {'global_spearman':None if not np.isfinite(rho) else float(rho),
                'top10_overlap':len(topa&topb)/min(10,d),
                'local_mean_abs_difference':float(np.mean(np.abs(values[a]-values[b])))}
    result={'dataset':dataset,'method':'SHAP permutation, independent background masker',
        'target':f'log P(y={classes[k]} | x)','class_index':k,'class_label':str(classes[k]),
        'n_original_features':d,'feature_names':names,
        'evaluation_n':len(eval_ids),'background_n':len(bg_ids),
        'evaluation_indices':eval_ids.tolist(),'background_indices':bg_ids.tolist(),
        'random_seed':seed,'student_selected_features':{z:list(X.columns[i]) for z,i in selected.items()},
        'global_explanations':outputs,
        'agreement':{'pce_vs_teacher':comparison('pce_hdmr','teacher'),
                     'ebm_vs_teacher':comparison('ebm','teacher'),
                     'pce_vs_ebm':comparison('pce_hdmr','ebm')},
        'limitations':['Small evaluation/background samples are approximate.',
          'Independent masking breaks correlations; SHAP is associational, not causal.',
          'SHAP comparisons use log class probability, not raw class probability.',
          'The same held-out test partition has previously been inspected; exploratory only.',
          'Only one class is explained per run; repeat class-index for other classes.']}
    suffix=f'{dataset}_shap_class{k}'
    path=OUT/f'{suffix}.json'
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    np.savez_compressed(OUT/f'{suffix}_local.npz',eval_indices=eval_ids,
                        teacher=values['teacher'],pce_hdmr=values['pce_hdmr'],ebm=values['ebm'])
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,6))
        order=np.argsort(np.mean(np.abs(values['teacher']),axis=0))[-12:]
        yy=np.arange(len(order));w=.26
        for j,name in enumerate(('teacher','pce_hdmr','ebm')):
            ax.barh(yy+(j-1)*w,np.mean(np.abs(values[name]),axis=0)[order],height=w,label=name)
        ax.set_yticks(yy,labels=[names[i] for i in order])
        ax.set_xlabel('Mean absolute permutation SHAP (log probability units)')
        ax.set_title(f'{dataset}: class {classes[k]} — shared feature space')
        ax.legend();fig.tight_layout()
        fig.savefig(OUT/f'{suffix}_comparison.png',dpi=180)
        plt.close(fig)
    except Exception as e:
        warnings.warn(f'Plot generation failed: {e}')
    print(json.dumps({'output':str(path),'agreement':result['agreement']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['phishing','har','all'],default='all')
    p.add_argument('--samples',type=int,default=12)
    p.add_argument('--background',type=int,default=10)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--class-index',type=int,default=None,
                   help='Zero-based class index; default 1 for binary, 0 for multiclass')
    a=p.parse_args()
    for dataset in (['phishing','har'] if a.dataset=='all' else [a.dataset]):
        run(dataset,a.samples,a.background,a.seed,a.class_index)
