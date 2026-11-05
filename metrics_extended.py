"""Supplemental validated classification and distillation metrics."""
import numpy as np
from scipy.special import rel_entr
from sklearn.metrics import (accuracy_score,balanced_accuracy_score, f1_score,
                             precision_score,recall_score,confusion_matrix,log_loss)

def validate_probabilities(p):
    p=np.asarray(p,dtype=float)
    if p.ndim!=2 or p.shape[1]<2 or not np.all(np.isfinite(p)) or np.any(p<0) or not np.allclose(p.sum(axis=1),1,atol=1e-6):
        raise ValueError('Probabilities must be finite, nonnegative, normalized N x K array')
    return p

def kl_divergence(p,q):
    p,q=validate_probabilities(p),validate_probabilities(q)
    if p.shape!=q.shape:raise ValueError('Shape mismatch')
    return float(np.mean(np.sum(rel_entr(p,np.clip(q,1e-12,1)),axis=1)))

def js_divergence(p,q):
    p,q=validate_probabilities(p),validate_probabilities(q)
    m=(p+q)/2
    return (kl_divergence(p,m)+kl_divergence(q,m))/2

def classification_metrics(y,p,classes):
    p=validate_probabilities(p)
    if p.shape[1]!=len(classes):raise ValueError('Class count mismatch')
    pred=np.asarray(classes)[p.argmax(axis=1)]
    return dict(accuracy=float(accuracy_score(y,pred)),
        balanced_accuracy=float(balanced_accuracy_score(y,pred)),
        macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),
        weighted_f1=float(f1_score(y,pred,average='weighted',zero_division=0)),
        macro_precision=float(precision_score(y,pred,average='macro',zero_division=0)),
        macro_recall=float(recall_score(y,pred,average='macro',zero_division=0)),
        log_loss=float(log_loss(y,p,labels=classes)),
        confusion_matrix=confusion_matrix(y,pred,labels=classes).tolist())

def distillation_metrics(t,s):
    t,s=validate_probabilities(t),validate_probabilities(s)
    if t.shape!=s.shape:raise ValueError('Shape mismatch')
    return dict(hard_agreement=float(np.mean(t.argmax(axis=1)==s.argmax(axis=1))),
        probability_mae=float(np.mean(np.abs(t-s))),
        kl_teacher_student=kl_divergence(t,s),
        js_divergence=js_divergence(t,s))
