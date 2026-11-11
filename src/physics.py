                                                                                
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch


def finite(name, value):
    if not torch.isfinite(value).all():
        raise ValueError(f"Nonfinite {name}")


def confidence_from_moments(opacity, depth_numerator, depth2_numerator, *,
                            delta_depth=1e-7, delta_variance=1e-7,
                            midpoint=0.5, variance_scale=0.1, slope=10.0):
    if not opacity.shape == depth_numerator.shape == depth2_numerator.shape:
        raise ValueError("Moment map shapes differ")
    for name, x in (("opacity",opacity),("depth numerator",depth_numerator),
                    ("depth second moment",depth2_numerator)):
        finite(name,x)
    if (opacity < -1e-5).any() or (opacity > 1+1e-5).any():
        raise ValueError("Opacity must be composited coverage in [0,1]")
    if min(delta_depth,delta_variance,variance_scale,slope) <= 0:
        raise ValueError("Nonpositive confidence constant")
    O = opacity.clamp(0,1)
    D = depth_numerator/(O+delta_depth)
                                                                                
    V = ((depth2_numerator-2*D*depth_numerator+D.square()*O)/
         (O+delta_variance)).clamp_min(0)
    C = torch.sigmoid(slope*(O-midpoint))*torch.exp(-V/variance_scale)
    return D,V,C


def laplacian(field, depth, beta=0.1):
    if field.shape != depth.shape or field.ndim < 2 or beta <= 0:
        raise ValueError("Laplacian requires matching spatial maps and positive beta")
    finite("depth",depth); finite("field",field)
    z = depth.detach()
    horizontal = torch.exp(-(z[..., :, 1:]-z[..., :, :-1]).abs()/beta)
    vertical = torch.exp(-(z[..., 1:, :]-z[..., :-1, :]).abs()/beta)
                                                                               
    result = torch.zeros_like(field)
    dh = horizontal*(field[..., :, :-1]-field[..., :, 1:])
    dv = vertical*(field[..., :-1, :]-field[..., 1:, :])
    result[..., :, :-1] += dh
    result[..., :, 1:] -= dh
    result[..., :-1, :] += dv
    result[..., 1:, :] -= dv
    return result


def sparse_laplacian(depth, beta=0.1):
    depth = np.asarray(depth,dtype=np.float64)
    if depth.ndim != 2 or not np.isfinite(depth).all() or beta <= 0:
        raise ValueError("Invalid source depth")
    h,w = depth.shape
    ids = np.arange(h*w).reshape(h,w)
    left = np.concatenate([ids[:,:-1].ravel(),ids[:-1,:].ravel()])
    right = np.concatenate([ids[:,1:].ravel(),ids[1:,:].ravel()])
    weights = np.exp(-np.abs(depth.ravel()[left]-depth.ravel()[right])/beta)
    i = np.concatenate([left,right,left,right])
    j = np.concatenate([left,right,right,left])
    v = np.concatenate([weights,weights,-weights,-weights])
    return sp.coo_matrix((v,(i,j)),shape=(h*w,h*w)).tocsr()


@torch.no_grad()
def solve_source(S,D,C,*,lambda_q=0.1,beta=0.1,jitter=1e-8):
                                                                                    

                                                                               
                                                                                   
       
    if S.shape != D.shape or S.shape != C.shape or lambda_q < 0 or jitter <= 0:
        raise ValueError("Invalid source-field inputs")
    for name,x in (("S",S),("D",D),("C",C)): finite(name,x)
    if ((C<0)|(C>1)).any(): raise ValueError("Confidence outside [0,1]")
    h,w = S.shape[-2:]
    fields = []
    for s,d,c in zip(S.reshape(-1,h,w),D.reshape(-1,h,w),C.reshape(-1,h,w)):
        L = sparse_laplacian(d.cpu().numpy(),beta)
        s0 = s.cpu().numpy().astype(np.float64).ravel()
        c0 = c.cpu().numpy().astype(np.float64).ravel()
        q0 = L@s0
        A = sp.diags(c0+jitter)+2*lambda_q*L
        rhs = c0*q0
        q = spla.spsolve(A.tocsc(),rhs)
        if not np.isfinite(q).all() or np.linalg.norm(A@q-rhs) > 1e-6*(1+np.linalg.norm(rhs)):
            raise RuntimeError("Source-field solve failed; refusing an unguided fallback")
        fields.append(torch.as_tensor(q.reshape(h,w),device=S.device,dtype=S.dtype))
    return torch.stack(fields).reshape(S.shape)


@torch.no_grad()
def align_radiometry(X,S,C,*,lambda_chi=0.01,eps=1e-8):
                                                                                  
    if X.shape != S.shape or X.shape != C.shape or lambda_chi <= 0:
        raise ValueError("REA requires matching maps and positive identity regularization")
    for name,x in (("X",X),("S",S),("C",C)): finite(name,x)
    if ((C<0)|(C>1)).any(): raise ValueError("Confidence outside [0,1]")
                                                                           
    x,s,c = X.double(),S.double(),C.double()
    axes = (-2,-1)
    total = c.sum(axes,keepdim=True)
    a11 = (c*s*s).sum(axes,keepdim=True)+lambda_chi
    a12 = (c*s).sum(axes,keepdim=True)
    b1 = (c*s*x).sum(axes,keepdim=True)+lambda_chi
    b2 = (c*x).sum(axes,keepdim=True)
    det = a11*total-a12.square()
    valid = (total>eps)&(det>eps*eps)
    denominator = torch.where(valid,det,torch.ones_like(det))
    chi = torch.where(valid,(b1*total-a12*b2)/denominator,torch.ones_like(det))
    psi = torch.where(valid,(a11*b2-a12*b1)/denominator,torch.zeros_like(det))
    return chi.to(X.dtype),psi.to(X.dtype)


def guidance_energy(X,S,D,C,q,*,lambda_rea=1.,lambda_dspg=1.,lambda_s=.01,
                    lambda_chi=.01,xi=1e-3,beta=.1):
    if not X.shape == S.shape == D.shape == C.shape == q.shape:
        raise ValueError("Proxy and decoded predictions must share frame/spatial alignment")
    chi,psi = align_radiometry(X,S.detach(),C.detach(),lambda_chi=lambda_chi)
    rea = (C.detach()*torch.sqrt((X-chi*S.detach()-psi).square()+xi*xi)).sum()
    residual = laplacian(X,D,beta)-lambda_s*X-q.detach()
    dspg = ((1-C.detach())*residual.square()).sum()
    return lambda_rea*rea+lambda_dspg*dspg
