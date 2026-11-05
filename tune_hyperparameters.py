"""Validation-only model selection; held-out test is evaluated once after selection.
Run: python tune_hyperparameters.py --dataset all
"""
import argparse, json, platform, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import log_loss
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingRegressor
from run_experiments import (load_dataset,split_data,SEED,OUT,
    PCEStudent,EBMStudent,select_features,prob_metrics,fidelity,rank_agreement)

TEACHER_GRID=[
 dict(n_estimators=200,max_depth=3,learning_rate=.05),
 dict(n_estimators=350,max_depth=4,learning_rate=.06),
 dict(n_estimators=500,max_depth=3,learning_rate=.04),
 dict(n_estimators=300,max_depth=5,learning_rate=.04),
]
PCE_ALPHAS=[.0003,.001,.003,.01]
FEATURE_COUNTS=[8,12,16]
EBM_GRID=[
 dict(learning_rate=.025,max_rounds=250,interactions=0),
 dict(learning_rate=.025,max_rounds=400,interactions=3),
 dict(learning_rate=.05,max_rounds=400,interactions=3),
 dict(learning_rate=.025,max_rounds=400,interactions=6),
]

def kl(p,q):
    p=np.clip(p,1e-9,1);q=np.clip(q,1e-9,1)
    return float(np.mean(np.sum(p*np.log(p/q),axis=1)))

def fit_ebm(X,p,names,params):
    student=EBMStudent(p.shape[1]);student.feature_names=list(names)
    student.models=[]
    for k in range(p.shape[1]):
        model=ExplainableBoostingRegressor(
            **params,outer_bags=1,early_stopping_rounds=30,max_bins=128,
            n_jobs=1,random_state=SEED)
        model.fit(X,np.log(np.clip(p[:,k],1e-6,1)))
        student.models.append(model)
    return student

def run(dataset):
    X,y,batches=load_dataset(dataset)
    X=X.replace([np.inf,-np.inf],np.nan)
    tr,va,te,split=split_data(X,y,batches)
    median=X.iloc[tr].median().fillna(0)
    X=X.fillna(median).fillna(0)
    classes=np.sort(y.unique())
    encoded=np.searchsorted(classes,y.to_numpy())
    nclass=len(classes)
    trainX=X.iloc[tr]; valX=X.iloc[va]; testX=X.iloc[te]
    best_teacher=None;teacher_trials=[]
    for cfg in TEACHER_GRID:
        model=XGBClassifier(**cfg,subsample=.9,colsample_bytree=.85,
            objective='multi:softprob' if nclass>2 else 'binary:logistic',
            eval_metric='mlogloss' if nclass>2 else 'logloss',
            random_state=SEED,n_jobs=4,tree_method='hist')
        model.fit(trainX,encoded[tr],verbose=False)
        valp=model.predict_proba(valX)
        score=float(log_loss(encoded[va],valp,labels=np.arange(nclass)))
        teacher_trials.append(dict(params=cfg,validation_log_loss=score))
        print(dataset,'teacher',cfg,'validation logloss',round(score,5),flush=True)
        if best_teacher is None or score<best_teacher[0]:
            best_teacher=(score,cfg,model)
    _,teacher_cfg,teacher=best_teacher
    trainp=teacher.predict_proba(trainX)
    valp=teacher.predict_proba(valX)
    teacher_testp=teacher.predict_proba(testX) # Predictions evaluated only after tuning
    selected={n:select_features(trainX,teacher,n) for n in FEATURE_COUNTS}
    pce_trials=[];ebm_trials=[]
    best_pce=None;best_ebm=None
    for n,idx in selected.items():
        names=list(X.columns[idx]); A=X.iloc[:,idx].to_numpy(dtype=float)
        for alpha in PCE_ALPHAS:
            try:
                student=PCEStudent(alpha=alpha).fit(A[tr],trainp,names)
                score=kl(valp,student.predict_proba(A[va]))
                cfg=dict(n_features=n,alpha=alpha)
                pce_trials.append(dict(params=cfg,validation_kl=score,
                                       complexity_terms=student.complexity()))
                print(dataset,'PCE',cfg,'validation KL',round(score,5),flush=True)
                if best_pce is None or score<best_pce[0]:best_pce=(score,cfg,student,idx)
            except Exception as e:
                pce_trials.append(dict(params=dict(n_features=n,alpha=alpha),error=str(e)))
        for ebm_cfg in EBM_GRID:
            try:
                student=fit_ebm(A[tr],trainp,names,ebm_cfg)
                score=kl(valp,student.predict_proba(A[va]))
                cfg=dict(n_features=n,**ebm_cfg)
                ebm_trials.append(dict(params=cfg,validation_kl=score,
                                       complexity_terms=student.complexity()))
                print(dataset,'EBM',cfg,'validation KL',round(score,5),flush=True)
                if best_ebm is None or score<best_ebm[0]:best_ebm=(score,cfg,student,idx)
            except Exception as e:
                ebm_trials.append(dict(params=dict(n_features=n,**ebm_cfg),error=str(e)))
    if best_pce is None or best_ebm is None:raise RuntimeError('All student fits failed; inspect trials')
    output=dict(dataset=dataset,split=split,split_sizes=dict(train=len(tr),validation=len(va),test=len(te)),
        selection_metric=dict(teacher='validation log loss vs true labels',
                              students='validation KL(teacher || student)'),
        best_hyperparameters=dict(teacher=teacher_cfg,pce_hdmr=best_pce[1],ebm=best_ebm[1]),
        validation_scores=dict(teacher_log_loss=best_teacher[0],pce_kl=best_pce[0],ebm_kl=best_ebm[0]),
        trials=dict(teacher=teacher_trials,pce_hdmr=pce_trials,ebm=ebm_trials),
        metrics={},python_version=platform.python_version(),random_seed=SEED)
    output['metrics']['teacher']=prob_metrics(y.iloc[te],teacher_testp,classes)
    for name,best in [('pce_hdmr',best_pce),('ebm',best_ebm)]:
        _,cfg,model,idx=best
        probs=model.predict_proba(X.iloc[te,idx].to_numpy(dtype=float))
        metrics=prob_metrics(y.iloc[te],probs,classes)
        metrics.update(fidelity(teacher_testp,probs))
        metrics['complexity_terms']=model.complexity()
        metrics['selected_features']=list(X.columns[idx])
        output['metrics'][name]=metrics
    path=OUT/f'{dataset}_tuned.json'
    path.write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nFINAL HELD-OUT TEST RESULTS:',dataset,json.dumps(output['metrics'],indent=2),flush=True)
    print('Saved',path,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['phishing','har','all'],default='all')
    args=parser.parse_args()
    for dataset in (['phishing','har'] if args.dataset=='all' else [args.dataset]):run(dataset)
