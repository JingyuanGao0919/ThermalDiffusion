                                                                                      
from __future__ import annotations
import torch
from .physics import finite


def distillation_loss(observations,generated,render,*,lambda_gen=.3):
                                                                              

                                                                         
                                                                              
                                                                      
                                                                                   
       
    if not observations or lambda_gen < 0:
        raise ValueError("Missing sparse observations or invalid pseudo weight")
    if set(observations)&set(generated):
        raise ValueError("A generated target cannot overwrite an observed frame")
    obs_loss = None
    gen_loss = None
    for kind,items in (("observed",observations),("generated",generated)):
        for fid,row in items.items():
            if kind=="generated" and row.get("source")!="diffusion":
                raise ValueError("Generated supervision must come from frozen diffusion")
            result = render(fid,row["camera_id"])
            if result["frame_id"]!=fid or result["camera_id"]!=row["camera_id"]:
                raise ValueError("Render/target camera mismatch")
            image,target = result["image"],row["image"].detach()
            if image.shape != target.shape:
                raise ValueError("Supervision shape mismatch; resolve calibration upstream")
            finite("render",image); finite("supervision",target)
            error = torch.abs(image-target)
            if kind=="generated":
                C = row["confidence"].detach()
                if C.shape!=error.shape or ((C<0)|(C>1)).any():
                    raise ValueError("Bad confidence alignment or range")
                finite("confidence",C)
                term = (C*error).sum()
                gen_loss = term if gen_loss is None else gen_loss+term
            else:
                term = error.sum()
                obs_loss = term if obs_loss is None else obs_loss+term
    if gen_loss is None: gen_loss=obs_loss*0.
                                                                            
    pixels = next(iter(observations.values()))["image"].numel()
    return (obs_loss+lambda_gen*gen_loss)/pixels
