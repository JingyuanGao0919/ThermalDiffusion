                                                                                   
from __future__ import annotations
import os
import argparse
import fcntl
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from .io import checked,load_json,write_json,sha256_file
from .initialization import read_bundle
from .protocol import validate_saved
from .raster import make_camera,render_thermal,render_controls

ROOT=Path(os.environ["THERMAL_ROOT"])
OPT=dict(position_lr_init=.00016,position_lr_final=.0000016,position_lr_delay_mult=.01,
    position_lr_max_steps=3000,feature_lr=.0025,opacity_lr=.05,scaling_lr=.005,
    rotation_lr=.001,percent_dense=.01)
CONVENTIONS=dict(sh_degree=3,seed=42,lambda_dssim=.2,densify_from=500,densify_until=3000,
    densification_interval=100,densify_grad_threshold=.0005,prune_opacity=.005,
    opacity_reset="none at final iteration",training_size_hw=[320,512],optimizer=OPT,
    status="explicit implementation settings; paper specifies 3000 steps but not all optimizer/densification details")


def verify_sparse_contract(bundle,clip):
    if bundle.get("stage")!="sparse_proxy" or bundle.get("clip_id")!=clip["clip_id"]:
        raise ValueError("Not the requested inference bundle")
    if len(bundle["observations"])!=3 or len(bundle["target_cameras"])!=13:
        raise ValueError("Inference proxy requires exactly 3 observations and 13 target cameras")
    if [r["frame_id"] for r in bundle["observations"]]!=clip["input_frame_ids"]:
        raise ValueError("Input-view identity mismatch")
    if [r["frame_id"] for r in bundle["target_cameras"]]!=clip["target_frame_ids"]:
        raise ValueError("Target camera identity mismatch")
    if any(set(r)!={"frame_id","camera"} for r in bundle["target_cameras"]):
        raise ValueError("Target truth pixel access is forbidden")


def fit(bundle_path,initialization,output):
    output=checked(output,must_exist=False)
    if output.exists():raise FileExistsError(output)
    bundle=read_bundle(bundle_path);valid=validate_saved(ROOT/"protocol_v1")
    setting=bundle["clip_id"].split("_",1)[0]
    clip=next(c for c in load_json(ROOT/"protocol_v1"/f"test_{setting}.json")["clips"] if c["clip_id"]==bundle["clip_id"])
    verify_sparse_contract(bundle,clip)
    if bundle["protocol_sha256"]!=valid["protocol_sha256"]:raise ValueError("Protocol mismatch")
    init=load_json(Path(initialization)/"receipt.json")
    if init["source_bundle_sha256"]!=bundle["bundle_sha256"]:raise ValueError("Initialization source mismatch")
    if set(init["seed_frame_ids"])!=set(clip["input_frame_ids"]):raise ValueError("Sparse initialization used other views")
    if sha256_file(Path(initialization)/"points.npz")!=init["npz_sha256"]:raise ValueError("Changed initialization array")
    with (ROOT/"runs/pseudo_adaptation_v1/gpu_inference.lock").open("a") as slot:
        fcntl.flock(slot,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not torch.cuda.is_available():raise RuntimeError("GPU_REQUIRED")
        torch.cuda.set_per_process_memory_fraction(.25);torch.cuda.reset_peak_memory_stats()
        torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
        sys.path.insert(0,str(checked(os.environ["GUIDEDVD_ROOT"])))
        from scene.gaussian_model import GaussianModel
        from utils.graphics_utils import BasicPointCloud
        from utils.loss_utils import ssim
        data=np.load(checked(Path(initialization)/"points.npz"))
        points=data["points"].astype(np.float32);colors=data["colors"].mean(1,keepdims=True).repeat(3,axis=1)
        centers=np.stack([-np.array(r["camera"]["R"]).T@r["camera"]["T"] for r in bundle["observations"]])
        extent=max(float(np.linalg.norm(centers-centers.mean(0),axis=1).max())*1.1,.01)
        model=GaussianModel(SimpleNamespace(sh_degree=3,use_color=True,train_bg=False,prune_from_iter=500,dist_thres=10.))
        model.create_from_pcd(BasicPointCloud(points=points,colors=colors,normals=np.zeros_like(points)),extent)
        model.training_setup(SimpleNamespace(**OPT))
        observations=[]
        for row in bundle["observations"]:
            with Image.open(checked(row["image_path"])) as image:
                pixels=np.asarray(image.convert("L").resize((512,320),Image.Resampling.BILINEAR),dtype=np.float32).copy()/255
            observations.append((make_camera(row["frame_id"],row["camera"]),torch.tensor(pixels,device="cuda").unsqueeze(0)))
        output.mkdir(parents=True)
        state=dict(stage="training",iteration=0,total_iterations=3000,training_frame_ids=clip["input_frame_ids"],
            target_truth_used=False,failed=None,source_bundle_sha256=bundle["bundle_sha256"])
        def status(stage):
            state.update(stage=stage,updated_unix=time.time(),gaussians=len(model.get_xyz))
            write_json(output/"status.json",state,overwrite=True)
        status("training");order=[];rng=random.Random(42);started=time.monotonic()
        try:
            with (output/"steps.jsonl").open("x") as log:
                for iteration in range(1,3001):
                    if not order:order=list(range(3));rng.shuffle(order)
                    camera,truth=observations[order.pop()]
                    model.update_learning_rate(iteration)
                    if iteration%1000==0:model.oneupSHdegree()
                    rendered=render_thermal(camera,model)
                    loss=.8*(rendered["image"]-truth).abs().mean()+.2*(1-ssim(rendered["image"].unsqueeze(0),truth.unsqueeze(0)))
                    if not torch.isfinite(loss):raise RuntimeError("Nonfinite sparse-proxy loss")
                    loss.backward()
                    with torch.no_grad():
                        if iteration<3000:
                            visible=rendered["visibility_filter"]
                            model.max_radii2D[visible]=torch.maximum(model.max_radii2D[visible],rendered["radii"][visible])
                            model.add_densification_stats(rendered["viewspace_points"],visible)
                            if iteration>500 and iteration%100==0:model.densify_and_prune(.0005,.005,extent,None,iteration)
                        model.optimizer.step();model.optimizer.zero_grad(set_to_none=True)
                    if iteration%100==0:
                        state.update(iteration=iteration,last_training_loss=float(loss));status("training")
                        log.write(json.dumps(dict(iteration=iteration,loss=float(loss),gaussians=len(model.get_xyz)))+"\n");log.flush()
                        print(iteration,"/3000",float(loss),"gaussians",len(model.get_xyz),flush=True)
            for key in ("_xyz","_features_dc","_features_rest","_scaling","_rotation","_opacity"):
                if not torch.isfinite(getattr(model,key)).all():raise RuntimeError("Nonfinite proxy parameters")
            status("saving_and_rendering_controls")
            ply=output/"point_cloud.ply";model.save_ply(str(ply))
            lookup={row["frame_id"]:row["camera"] for row in bundle["observations"]+bundle["target_cameras"]}
            fields={key:[] for key in ("S","O","D","V","C")}
            (output/"controls_preview").mkdir()
            for index,frame_id in enumerate(clip["frame_ids"]):
                controls=render_controls(make_camera(frame_id,lookup[frame_id]),model)
                for key in fields:
                    array=controls[key].detach().cpu().numpy()
                    if not np.isfinite(array).all():raise RuntimeError("Nonfinite rendered control")
                    fields[key].append(array)
                for key in ("S","C"):
                    pixels=np.rint(np.clip(fields[key][-1][0],0,1)*255).astype(np.uint8)
                    Image.fromarray(pixels).save(output/"controls_preview"/f"{index:02d}_{key}.png")
            fields={key:np.stack(value) for key,value in fields.items()}
            np.savez_compressed(output/"controls.npz",**fields)
            result=dict(protocol_sha256=valid["protocol_sha256"],clip_id=clip["clip_id"],iterations=3000,
                purpose="single_clip_inference_stage_integration",training_frame_ids=clip["input_frame_ids"],
                rendered_frame_ids=clip["frame_ids"],target_truth_used=False,source_bundle_sha256=bundle["bundle_sha256"],
                initialization_receipt=str(checked(Path(initialization)/"receipt.json")),conventions=CONVENTIONS,
                point_cloud_sha256=sha256_file(ply),controls_sha256=sha256_file(output/"controls.npz"),
                elapsed_seconds=time.monotonic()-started,gaussian_count=len(model.get_xyz),
                peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                coverage_mean=float(fields["O"].mean()),confidence_mean=float(fields["C"].mean()),
                confident_pixel_fraction=float((fields["C"]>.5).mean()))
            write_json(output/"receipt.json",result);status("coarse_proxy_and_controls_complete")
            return result
        except Exception as error:state["failed"]=str(error);status("failed");raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--bundle",required=True);p.add_argument("--initialization",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();print(json.dumps(fit(a.bundle,a.initialization,a.output),indent=2))
