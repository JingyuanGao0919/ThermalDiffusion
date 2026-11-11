                                                                                  
from __future__ import annotations
import math
import numpy as np


def paired_ids(predictions,targets):
    if set(predictions)!=set(targets):
        raise ValueError(f"Evaluation ID mismatch: missing={sorted(set(targets)-set(predictions))[:4]}, "
                         f"unexpected={sorted(set(predictions)-set(targets))[:4]}")
    if not targets: raise ValueError("Empty evaluation")
    return sorted(targets)


def psnr(pred,gt,mask=None):
    pred,gt=np.asarray(pred,dtype=np.float64),np.asarray(gt,dtype=np.float64)
    if pred.shape!=gt.shape or not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError("Invalid evaluation images")
    if mask is None: mask=np.ones(pred.shape,dtype=bool)
    mask=np.asarray(mask,dtype=bool)
    if mask.shape!=pred.shape: raise ValueError("Mask/image shape mismatch")
    if not mask.any(): return None                                               
    mse=np.mean(np.square(pred[mask]-gt[mask]))
    return float("inf") if mse==0 else -10*math.log10(mse)


def region_psnr(pred,gt,proxy_confidence,threshold=.5):
    C=np.asarray(proxy_confidence)
    if not np.isfinite(C).all() or ((C<0)|(C>1)).any(): raise ValueError("Invalid fixed proxy mask")
    return {"observable":psnr(pred,gt,C>threshold),"hole":psnr(pred,gt,C<=threshold)}


def macro_average(groups):
    if set(groups)!={"Indoor","Outdoor","UAV"}: raise ValueError("Missing scenario group")
    metrics={"PSNR","SSIM","LPIPS"}
    if any(set(v)!=metrics for v in groups.values()): raise ValueError("Metric group mismatch")
    return {metric:sum(groups[g][metric] for g in groups)/3 for metric in sorted(metrics)}
