                                                                                    
from __future__ import annotations
import contextlib
import torch
from .physics import finite


def trailing_timesteps(total=1000,steps=20):
    if not 1 <= steps <= total:
        raise ValueError("DDIM steps must be in [1,num_train_timesteps]")
                                                                          
    return [total-1-(i*total//steps) for i in range(steps)]


def v_predictions(x,v,alpha):
                                                                       
    if not 0 <= float(alpha) <= 1:
        raise ValueError("Alpha outside [0,1]")
    a = torch.as_tensor(alpha,device=x.device,dtype=x.dtype)
    return a.sqrt()*x-(1-a).sqrt()*v, a.sqrt()*v+(1-a).sqrt()*x


def transition(x0,eps,alpha,previous_alpha,*,eta=0.,noise=None):
    a,prev = float(alpha),float(previous_alpha)
    if not 0 <= a <= prev <= 1 or prev == 0 or not 0 <= eta <= 1:
        raise ValueError("Invalid reverse DDIM transition")
    variance = 0. if a == 1 else (1-prev)/(1-a)*(1-a/prev)
    sigma = eta*max(0.,variance)**.5
    direction = max(0.,1-prev-sigma*sigma)**.5
    result = prev**.5*x0+direction*eps
    if sigma:
        if noise is None: raise ValueError("Stochastic DDIM needs explicit sampled noise")
        result = result+sigma*noise
    return result


def guided_step(x,*,predict_v,alpha,previous_alpha,decode=None,energy=None,
                eta=0.,noise=None,gamma=.2,clip_norm=1.,x0_scale=1.,decode_x0_scale=1.,
                diagnostics=None):
    if (decode is None) != (energy is None):
        raise ValueError("Guidance needs both a differentiable decoder and objective")
    if gamma < 0 or clip_norm <= 0 or x0_scale <= 0 or decode_x0_scale <= 0:
        raise ValueError("Invalid guidance controls")
    guided = energy is not None
    with torch.enable_grad() if guided else torch.no_grad():
        current = x.detach().requires_grad_(guided)
        v = predict_v(current)
        if v.shape != current.shape: raise ValueError("Denoiser shape mismatch")
        finite("v prediction",v)
        raw_x0,eps = v_predictions(current,v,alpha)
        x0 = raw_x0*x0_scale                                                      
        previous = transition(x0,eps,alpha,previous_alpha,eta=eta,noise=noise)
        if guided:
                                                                              
                                                                     
            loss = energy(decode(raw_x0*decode_x0_scale))
            if loss.numel()!=1 or not loss.requires_grad:
                raise ValueError("Guidance disconnected from current latent")
            finite("guidance objective",loss)
            grad, = torch.autograd.grad(loss,current)
            finite("guidance gradient",grad)
            norm = grad.to(torch.float64 if grad.dtype==torch.float64 else torch.float32).norm()
            scale = gamma/torch.clamp(norm/clip_norm,min=1.)
            correction = scale*grad
            if diagnostics is not None:
                proposal_norm = previous.detach().float().norm()
                correction_norm = correction.detach().float().norm()
                diagnostics(dict(energy=float(loss.detach()),gradient_norm=float(norm.detach()),
                    correction_norm=float(correction_norm),proposal_norm=float(proposal_norm),
                    relative_correction=float(correction_norm/proposal_norm.clamp_min(1e-12)),
                    gradient_clipped=bool(norm.detach()>clip_norm),gamma=gamma,clip_norm=clip_norm))
            previous = previous-correction
        finite("DDIM sample",previous)
    return previous.detach(),x0.detach()


def sample(*,model_v,initial_noise,alphas_cumprod,steps=20,decode=None,energy=None,
           eta=0.,gamma=.2,clip_norm=1.,generator=None,dynamic_scales=None,callback=None,
           diagnostics_callback=None):
    times = trailing_timesteps(len(alphas_cumprod),steps)
    x = initial_noise
    for k,t in enumerate(times):
        previous = times[k+1] if k+1<len(times) else -1
        a = float(alphas_cumprod[t])
        pa = float(alphas_cumprod[previous]) if previous>=0 else 1.
        if dynamic_scales is None:
            x0_scale,decode_x0_scale = 1.,1.
        else:
            current_scale = float(dynamic_scales[t])
            previous_scale = float(dynamic_scales[previous]) if previous>=0 else float(dynamic_scales[0])
            x0_scale = previous_scale/current_scale
            decode_x0_scale = 1./current_scale
        noise = (torch.randn(x.shape,device=x.device,dtype=x.dtype,generator=generator)
                 if eta else None)
        x,_ = guided_step(x,predict_v=lambda z:model_v(z,t),alpha=a,previous_alpha=pa,
            decode=decode,energy=energy,eta=eta,noise=noise,gamma=gamma,
            clip_norm=clip_norm,x0_scale=x0_scale,decode_x0_scale=decode_x0_scale,
            diagnostics=(lambda row:diagnostics_callback(dict(step=k+1,timestep=t,**row)))
                if diagnostics_callback is not None else None)
        if callback is not None:callback(k,t)
    return x
