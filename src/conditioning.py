                                                                           

                                                                            
                                                                            
   
from __future__ import annotations
import torch

CONVENTION = "dynamicrafter_observed_first_frame_repeat_v1"


@torch.no_grad()
def build_conditioning(model, observed_image, *, frames=16, drop_image=None):
                                                                             

                                                                          
                                                                               
                                                                   
       
    if observed_image.ndim != 4 or observed_image.shape[1] != 3:
        raise ValueError("Expected Bx3xHxW observed image")
    if observed_image.shape[0] < 1 or frames != 16:
        raise ValueError("This convention requires a nonempty batch and 16 frames")
    if not torch.is_floating_point(observed_image) or not torch.isfinite(observed_image).all():
        raise ValueError("Observed pixels must be finite floating point values")
    if observed_image.min() < -1 or observed_image.max() > 1:
        raise ValueError("Expected normalized observed pixels in [-1,1]")
    batch = observed_image.shape[0]
    if drop_image is None:
        drop_image = torch.zeros(batch, device=observed_image.device, dtype=torch.bool)
    if drop_image.shape != (batch,) or drop_image.dtype != torch.bool or drop_image.device != observed_image.device:
        raise ValueError("Image dropout must be a B boolean mask on the image device")
    latent = model.encode_first_stage(observed_image.unsqueeze(2))
    if latent.ndim != 5 or latent.shape[:3] != (batch, 4, 1):
        raise ValueError("Unexpected observed-image latent shape")
    concat = latent.expand(-1, -1, frames, -1, -1).contiguous().detach()
    text = model.get_learned_conditioning([""] * batch).detach()
    masked = observed_image * (~drop_image).to(observed_image.dtype).reshape(batch, 1, 1, 1)
    image = model.image_proj_model(model.embedder(masked))
    null = model.image_proj_model(model.embedder(torch.zeros_like(observed_image)))
    cond = {"c_concat": [concat], "c_crossattn": [torch.cat([text, image], dim=1).detach()]}
    uncond = {"c_concat": [concat], "c_crossattn": [torch.cat([text, null], dim=1).detach()]}
    if not all(torch.isfinite(v[0]).all() for c in (cond, uncond) for v in c.values()):
        raise RuntimeError("Nonfinite conditioning")
    return cond, uncond


def sample_image_dropout(batch, *, probability=.1, device="cpu", generator=None):
                                                                        
    if batch < 1 or not 0 <= probability <= 1:
        raise ValueError("Invalid dropout settings")
    return torch.rand(batch, device=device, generator=generator) < probability
