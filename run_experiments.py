"""Reproducible research prototype. Run: python run_experiments.py --dataset all
Requires internet on first run for official UCI downloads.
"""
import argparse, json, zipfile, io, warnings
from pathlib import Path
from urllib.request import urlopen
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import QuantileTransformer, PolynomialFeatures, StandardScaler
from sklearn.linear_model import MultiTaskLasso
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import spearmanr
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingRegressor
from ucimlrepo import fetch_ucirepo

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results"; OUT.mkdir(exist_ok=True)
SEED=42
def load_phishing():
    ds=fetch_ucirepo(id=327)
    X=ds.data.features.copy().apply(pd.to_numeric,errors="coerce")
    y=ds.data.targets.iloc[:,0].astype(str)
    assert X.shape[1]==30 and len(X)>=2000
    return X,y,None

def load_har():
    """Official UCI HAR release, preserving the original subject-disjoint test split.

    Returns (X, y, group_metadata). Group metadata is a dictionary so the
    original train/test designation is retained explicitly.
    """
    url = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
    archive = zipfile.ZipFile(io.BytesIO(urlopen(url, timeout=120).read()))
    members = archive.namelist()
    # Some UCI releases wrap the dataset directory in an inner ZIP.
    if not any(name.endswith("train/X_train.txt") for name in members):
        nested = [name for name in members if name.lower().endswith(".zip")]
        if len(nested) != 1:
            raise RuntimeError("HAR archive missing expected train files or single nested ZIP")
        archive = zipfile.ZipFile(io.BytesIO(archive.read(nested[0])))
        members = archive.namelist()
    def read_member(suffix):
        matches = [name for name in members if name.endswith(suffix) and not name.startswith("__MACOSX/")]
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one {suffix} in official UCI HAR archive; got {matches}")
        return archive.read(matches[0]).decode("utf-8-sig")
    features = [line.strip() for line in read_member("features.txt").splitlines() if line.strip()]
    # UCI HAR features.txt entries are '<index> <name>'. Duplicate names exist.
    if len(features) != 561:
        raise RuntimeError(f"Expected 561 HAR features; got {len(features)}")
    names = [f"har_feature_{i:03d}" for i in range(1, 562)]
    frames=[]; labels=[]; subjects=[]; partitions=[]
    for part in ("train", "test"):
        x = pd.read_csv(io.StringIO(read_member(f"{part}/X_{part}.txt")), sep=r"\s+", header=None)
        y = pd.read_csv(io.StringIO(read_member(f"{part}/y_{part}.txt")), header=None).iloc[:,0]
        group = pd.read_csv(io.StringIO(read_member(f"{part}/subject_{part}.txt")), header=None).iloc[:,0]
        if not (len(x)==len(y)==len(group)) or x.shape[1]!=561:
            raise ValueError("HAR features, targets, and subjects do not align")
        x.columns=names
        frames.append(x);labels.append(y);subjects.append(group);partitions.extend([part]*len(x))
    X=pd.concat(frames,ignore_index=True)
    y=pd.concat(labels,ignore_index=True).astype(str)
    groups=np.concatenate([v.to_numpy(dtype=int) for v in subjects])
    partition=np.asarray(partitions)
    if set(groups[partition=="train"]) & set(groups[partition=="test"]):
        raise ValueError("Official HAR train/test participants overlap")
    return X,y,{"kind":"har_subjects", "subjects":groups,"partition":partition}

def load_dataset(dataset):
    if dataset=="phishing": return load_phishing()
    if dataset=="har": return load_har()
    raise ValueError(f"Unknown dataset: {dataset}")

def split_data(X,y,batches):
    if isinstance(batches,dict) and batches.get("kind")=="har_subjects":
        subjects=np.asarray(batches["subjects"])
        partition=np.asarray(batches["partition"])
        original_train=np.flatnonzero(partition=="train")
        test=np.flatnonzero(partition=="test")
        splitter=GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=SEED)
        ti,vi=next(splitter.split(X.iloc[original_train],y.iloc[original_train],
                                  groups=subjects[original_train]))
        train=original_train[ti];validation=original_train[vi]
        assert not (set(subjects[train]) & set(subjects[validation]))
        assert not (set(subjects[train]) & set(subjects[test]))
        assert not (set(subjects[validation]) & set(subjects[test]))
        if len(set(y.iloc[train]))!=len(set(y)):
            raise ValueError("Training partition missing a class")
        return train,validation,test,"official_subject_disjoint_test_group_validation"
    if batches is None:
        a,b=train_test_split(np.arange(len(y)),test_size=.2,random_state=SEED,stratify=y)
        tr,va=train_test_split(a,test_size=.25,random_state=SEED,stratify=y.iloc[a])
        return tr,va,b, "stratified_random"
    raise ValueError("Unsupported split metadata; expected HAR subjects or Phishing without groups")

def prob_metrics(y, p, classes):
    pred=np.asarray(classes)[p.argmax(axis=1)]
    return dict(accuracy=float(accuracy_score(y,pred)),
                macro_f1=float(f1_score(y,pred,average="macro",zero_division=0)),
                log_loss=float(log_loss(y,p,labels=classes)))

def fidelity(p_teacher,p_student):
    eps=1e-9
    pt=np.clip(p_teacher,eps,1); ps=np.clip(p_student,eps,1)
    return dict(agreement=float(np.mean(pt.argmax(1)==ps.argmax(1))),
                probability_mae=float(np.mean(np.abs(pt-ps))),
                kl_teacher_student=float(np.mean(np.sum(pt*np.log(pt/ps),axis=1))))

def select_features(Xtr,teacher,limit):
    # Feature selection is based ONLY on the training-fitted teacher.
    order=np.argsort(teacher.feature_importances_)[::-1]
    return np.sort(order[:min(limit,Xtr.shape[1])])

class PCEStudent:
    """Sparse degree-2 Hermite PCE on selected marginal-Gaussian features.
    Analytic variance indices refer to an INDEPENDENT standard-normal reference
    distribution, NOT necessarily the empirical correlated data distribution.
    """
    def __init__(self,alpha=.003):
        self.alpha=alpha
    def fit(self,X,p,feature_names):
        self.feature_names=list(feature_names)
        self.qt=QuantileTransformer(n_quantiles=min(500,len(X)),
                                    output_distribution="normal",random_state=SEED)
        Z=np.clip(self.qt.fit_transform(X),-4,4)
        self.poly=PolynomialFeatures(degree=2,include_bias=False)
        B=self.poly.fit_transform(Z)
        # Hermite basis: x_i^2 -> (x_i^2-1)/sqrt(2), others already orthonormal
        powers=self.poly.powers_
        for j,powr in enumerate(powers):
            if 2 in powr: B[:,j]=(B[:,j]-1)/np.sqrt(2)
        self.model=MultiTaskLasso(alpha=self.alpha,max_iter=7000,random_state=SEED)
        self.model.fit(B,np.log(np.clip(p,1e-6,1)))
        self.powers=powers
        return self
    def _basis(self,X):
        Z=np.clip(self.qt.transform(X),-4,4)
        B=self.poly.transform(Z)
        for j,powr in enumerate(self.powers):
            if 2 in powr:B[:,j]=(B[:,j]-1)/np.sqrt(2)
        return B
    def predict_proba(self,X):
        return softmax(self.model.predict(self._basis(X)),axis=1)
    def importance(self):
        # coefficient energy, summed over class logits
        weights=np.sum(self.model.coef_**2,axis=0)
        total=np.zeros(len(self.feature_names))
        first=np.zeros_like(total)
        for j,powr in enumerate(self.powers):
            active=np.flatnonzero(powr)
            for i in active: total[i]+=weights[j]
            if len(active)==1:first[active[0]]+=weights[j]
        denom=weights.sum()+1e-12
        return first/denom,total/denom
    def complexity(self):
        return int(np.sum(np.any(np.abs(self.model.coef_)>1e-10,axis=0)))

class EBMStudent:
    def __init__(self,nclasses):
        self.nclasses=nclasses
    def fit(self,X,p,feature_names):
        self.feature_names=list(feature_names)
        self.models=[]
        for k in range(self.nclasses):
            m=ExplainableBoostingRegressor(interactions=3,outer_bags=1,
                max_rounds=300,early_stopping_rounds=30,max_bins=128,
                n_jobs=1,random_state=SEED)
            m.fit(X,np.log(np.clip(p[:,k],1e-6,1)))
            self.models.append(m)
        return self
    def predict_proba(self,X):
        return softmax(np.column_stack([m.predict(X) for m in self.models]),axis=1)
    def importance(self):
        # InterpretML term importance; interaction importance distributed equally.
        imp=np.zeros(len(self.feature_names))
        for m in self.models:
            for term,score in zip(m.term_features_,m.term_importances()):
                for j in term:imp[j]+=float(score)/len(term)
        return imp/max(self.nclasses,1)
    def complexity(self):
        return int(sum(len(m.term_features_) for m in self.models))

def rank_agreement(a,b):
    v=spearmanr(a,b).statistic
    return None if not np.isfinite(v) else float(v)

def run(dataset,fast=False):
    X,y,batches=load_dataset(dataset)
    X=X.replace([np.inf,-np.inf],np.nan)
    tr,va,te,split=split_data(X,y,batches)
    # Impute from training ONLY; no target leakage.
    med=X.iloc[tr].median().fillna(0)
    X=X.fillna(med).fillna(0)
    classes=np.sort(y.unique())
    encoded=pd.Categorical(y,categories=classes).codes
    teacher=XGBClassifier(n_estimators=120 if fast else 350,max_depth=4,
        learning_rate=.06,subsample=.9,colsample_bytree=.85,
        objective="multi:softprob" if len(classes)>2 else "binary:logistic",
        eval_metric="mlogloss" if len(classes)>2 else "logloss",
        random_state=SEED,n_jobs=4,tree_method="hist")
    teacher.fit(X.iloc[tr],encoded[tr],verbose=False)
    pt_train=teacher.predict_proba(X.iloc[tr])
    pt_test=teacher.predict_proba(X.iloc[te])
    chosen=select_features(X.iloc[tr],teacher,8 if fast else 12)
    names=list(X.columns[chosen])
    # Students fit ONLY on teacher predictions for training rows.
    # No test data or test labels used for selection/tuning.
    qtX=X.iloc[:,chosen].to_numpy(dtype=float)
    pce=PCEStudent(alpha=.006 if fast else .003).fit(qtX[tr],pt_train,names)
    ebm=EBMStudent(len(classes)).fit(qtX[tr],pt_train,names)
    output={"dataset":dataset,"n_samples":len(X),"n_features":X.shape[1],
        "classes":list(map(str,classes)),"split":split,
        "split_sizes":{"train":len(tr),"validation":len(va),"test":len(te)},
        "selected_features":names,"mode":"pilot" if fast else "full",
        "metrics":{}}
    for name,model in [("teacher",teacher),("pce_hdmr",pce),("ebm",ebm)]:
        pp=pt_test if name=="teacher" else model.predict_proba(qtX[te])
        item=prob_metrics(y.iloc[te],pp,classes)
        if name!="teacher":
            item.update(fidelity(pt_test,pp))
            item["complexity_terms"]=model.complexity()
        output["metrics"][name]=item
    first,total=pce.importance()
    eimp=ebm.importance()
    # Compare teacher's built-in importance as an exploratory proxy.
    # SHAP is computed separately in optional run_shap.py.
    timp=teacher.feature_importances_[chosen]
    output["importance"]={"features":names,"pce_first_order":first.tolist(),
        "pce_total_reference":total.tolist(),"ebm":eimp.tolist(),
        "teacher_gain_proxy":timp.tolist(),
        "spearman_pce_ebm":rank_agreement(total,eimp),
        "spearman_pce_teacher_proxy":rank_agreement(total,timp)}
    path=OUT/f"{dataset}_{'pilot' if fast else 'full'}.json"
    path.write_text(json.dumps(output,indent=2,ensure_ascii=False))
    print(json.dumps({"dataset":dataset,"results_file":str(path),
                      "metrics":output["metrics"]},indent=2))
    return output

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",choices=["phishing","har","all"],default="all")
    ap.add_argument("--pilot",action="store_true",help="Faster feasibility run")
    args=ap.parse_args()
    for ds in (["phishing","har"] if args.dataset=="all" else [args.dataset]):
        run(ds,args.pilot)
