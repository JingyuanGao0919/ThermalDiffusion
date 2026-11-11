                                                                                  

                                                                          
                                                                             
                                                                        
   
from __future__ import annotations
from dataclasses import dataclass
import torch
from .physics import confidence_from_moments, finite


@dataclass
class Camera:
    frame_id: str
    camera_id: str
    image_width: int
    image_height: int
    fx: float
    fy: float
    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor
    camera_center: torch.Tensor


def make_camera(frame_id,calibration,*,height=320,width=512,device="cuda"):
    c = calibration
    sx,sy = width/c["width"],height/c["height"]
    fx,fy = c["fx"]*sx,c["fy"]*sy
    cx,cy = (c["cx"]+.5)*sx-.5,(c["cy"]+.5)*sy-.5
    view = torch.eye(4,dtype=torch.float32,device=device)
    view[:3,:3] = torch.tensor(c["R"],device=device)
    view[:3,3] = torch.tensor(c["T"],device=device)
    projection = torch.zeros(4,4,device=device)
    near,far = .01,100.
    projection[0,0],projection[1,1] = 2*fx/width,2*fy/height
                                                         
    projection[0,2],projection[1,2] = (2*cx+1-width)/width,(2*cy+1-height)/height
    projection[2,2],projection[2,3] = far/(far-near),-far*near/(far-near)
    projection[3,2] = 1.
    view = view.T.contiguous()
    return Camera(frame_id,frame_id,width,height,fx,fy,view,
                  view@projection.T,torch.linalg.inv(view)[3,:3])


def _raster(camera,gaussians,*,colors=None,rasterizer_factory=None,settings_factory=None):
    if rasterizer_factory is None:
        from diff_gaussian_rasterization import GaussianRasterizer,GaussianRasterizationSettings
        rasterizer_factory,settings_factory = GaussianRasterizer,GaussianRasterizationSettings
    xyz = gaussians.get_xyz
    settings = settings_factory(image_height=camera.image_height,image_width=camera.image_width,
        tanfovx=camera.image_width/(2*camera.fx),tanfovy=camera.image_height/(2*camera.fy),
        bg=torch.zeros(3,device=xyz.device,dtype=xyz.dtype),scale_modifier=1.,
        viewmatrix=camera.world_view_transform,projmatrix=camera.full_proj_transform,
        sh_degree=gaussians.active_sh_degree,campos=camera.camera_center,
        prefiltered=False,debug=False,confidence=torch.ones_like(gaussians.get_opacity))
    screen = torch.zeros_like(xyz,requires_grad=True)
    if screen.requires_grad: screen.retain_grad()
    output = rasterizer_factory(raster_settings=settings)(means3D=xyz,means2D=screen,
        shs=gaussians.get_features if colors is None else None,colors_precomp=colors,
        opacities=gaussians.get_opacity,scales=gaussians.get_scaling,
        rotations=gaussians.get_rotation,cov3D_precomp=None)
    if len(output)!=4:
        raise RuntimeError("Installed rasterizer must return (color,radii,depth,alpha)")
    return dict(image=output[0],radii=output[1],alpha=output[3],
                viewspace_points=screen,visibility_filter=output[1]>0,
                frame_id=camera.frame_id,camera_id=camera.camera_id)


def render_thermal(camera,gaussians,**kwargs):
    out = _raster(camera,gaussians,**kwargs)
    out["image"] = out["image"].mean(dim=0,keepdim=True)
    return out


@torch.no_grad()
def render_controls(camera,gaussians,**kwargs):
    intensity = render_thermal(camera,gaussians,**kwargs)
    xyz = gaussians.get_xyz
    homogeneous = torch.cat([xyz,torch.ones_like(xyz[:,:1])],dim=1)
    z = (homogeneous@camera.world_view_transform)[:,2]
    attributes = torch.stack([torch.ones_like(z),z,z.square()],dim=1)
    feature_pass = _raster(camera,gaussians,colors=attributes,**kwargs)
    O,m1,m2 = feature_pass["image"].split(1,dim=0)
                                                                              
    if not torch.allclose(O,feature_pass["alpha"].reshape_as(O),atol=2e-4,rtol=2e-4):
        raise RuntimeError("Rasterizer feature coverage differs from alpha; moment pass invalid")
    D,V,C = confidence_from_moments(O,m1,m2)
    finite("thermal control",intensity["image"])
    return dict(frame_id=camera.frame_id,camera_id=camera.camera_id,S=intensity["image"],
                O=O,D=D,V=V,C=C,depth_numerator=m1,depth2_numerator=m2)
