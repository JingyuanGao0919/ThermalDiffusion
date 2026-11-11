                                                                                     
from __future__ import annotations
import os
import argparse
import json
import struct
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from .io import checked,load_json,write_json,canonical_hash,sha256_file


def read_bundle(path):
    data=load_json(path)
    digest=data.pop("bundle_sha256")
    if canonical_hash(data)!=digest: raise ValueError("Observation bundle integrity failure")
    data["bundle_sha256"]=digest
    if len(data["observations"])<3: raise ValueError("This initialization requires at least 3 observations")
    for row in data["observations"]:
        if sha256_file(row["image_path"])!=row["source_sha256"]:
            raise ValueError("Observation content changed")
    return data


def camera_to_world(camera):
    matrix=np.eye(4,dtype=np.float32)
    matrix[:3,:3]=np.asarray(camera["R"],dtype=np.float32)
    matrix[:3,3]=np.asarray(camera["T"],dtype=np.float32)
    return np.linalg.inv(matrix)


def write_ply(path,points,colors):
    dtype=[(name,"<f4") for name in ("x","y","z","nx","ny","nz")]
    dtype += [(name,"u1") for name in ("red","green","blue")]
    array=np.zeros(len(points),dtype=dtype)
    for i,name in enumerate(("x","y","z")): array[name]=points[:,i]
    for i,name in enumerate(("red","green","blue")):
        array[name]=np.clip(np.rint(colors[:,i]*255),0,255).astype(np.uint8)
    header=("ply\nformat binary_little_endian 1.0\nelement vertex "+str(len(points))+"\n"+
        "".join("property float "+x+"\n" for x in ("x","y","z","nx","ny","nz"))+
        "".join("property uchar "+x+"\n" for x in ("red","green","blue"))+"end_header\n")
    with Path(path).open("xb") as stream:
        stream.write(header.encode()); stream.write(array.tobytes())


def dust3r_initialize(bundle_path,output,*,iterations=300,max_points=200000,seed_indices=None,confidence_policy="strict",
    dust_root=os.environ["DUST3R_ROOT"],
    checkpoint=os.environ["DUST3R_CHECKPOINT"]):
    if not torch.cuda.is_available(): raise RuntimeError("GPU_REQUIRED")
    output=checked(output,must_exist=False)
    if output.exists(): raise FileExistsError(output)
    data=read_bundle(bundle_path)
    observations=data["observations"]
    indices=np.linspace(0,len(observations)-1,3,dtype=int).tolist() if seed_indices is None else list(seed_indices)
    if len(indices)<3 or len(set(indices))!=len(indices) or any(i<0 or i>=len(observations) for i in indices):
        raise ValueError("Invalid DUSt3R seed indices")
    rows=[observations[i] for i in indices]
    sys.path.insert(0,str(checked(dust_root)))
    sys.path.insert(0,str(checked(Path(dust_root)/"croco")))
    from dust3r.inference import inference,load_model
    from dust3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images
    from dust3r.cloud_opt import global_aligner,GlobalAlignerMode

    torch.set_num_threads(8); torch.manual_seed(42); np.random.seed(42)
    images=load_images([str(checked(row["image_path"])) for row in rows],size=512)
    model=load_model(str(checked(checkpoint)),device="cuda").eval()
    pairs=make_pairs(images,scene_graph="complete",prefilter=None,symmetrize=True)
    predictions=inference(pairs,model,"cuda",batch_size=1)
    del model; torch.cuda.empty_cache()
    scene=global_aligner(predictions,device="cuda",mode=GlobalAlignerMode.PointCloudOptimizer,
                         optimize_pp=True)
    poses=np.stack([camera_to_world(row["camera"]) for row in rows])
    intrinsics=[]
    for row,image in zip(rows,images):
        c=row["camera"]; w,h=c["width"],c["height"]
                                                                                       
        scale=512/max(w,h); rw,rh=round(w*scale),round(h*scale)
        th,tw=map(int,image["true_shape"][0])
        left,top=rw//2-tw//2,rh//2-th//2
        intrinsics.append(dict(fx=c["fx"]*rw/w,fy=c["fy"]*rh/h,
            cx=(c["cx"]+.5)*rw/w-.5-left,cy=(c["cy"]+.5)*rh/h-.5-top))
    scene.preset_pose(torch.tensor(poses,device="cuda"))
    scene.preset_focal([(k["fx"]+k["fy"])/2 for k in intrinsics])
    scene.preset_principal_point([np.array([k["cx"],k["cy"]]) for k in intrinsics])
    scene.compute_global_alignment(init="known_poses",niter=iterations,schedule="cosine",lr=.01)
    fitted=scene.get_im_poses().detach().cpu().numpy()
    if not np.allclose(fitted,poses,atol=2e-4): raise RuntimeError("Known DUSt3R poses moved")
    if confidence_policy not in ("strict","reference"):raise ValueError("Unknown confidence policy")
    if confidence_policy=="reference":
        scene.min_conf_thr=float(scene.conf_trf(torch.tensor(3.)))
    depths=[x.detach().cpu().numpy() for x in scene.get_depthmaps()]
    masks=[x.detach().cpu().numpy() for x in scene.get_masks()]
    print(json.dumps(dict(seed_indices=indices,reliable_pixels=[int(x.sum()) for x in masks])),flush=True)
    points=[]; colors=[]
    for i,(depth,mask,k,image) in enumerate(zip(depths,masks,intrinsics,images)):
        h,w=depth.shape; yy,xx=np.mgrid[:h,:w]
        camera_points=np.stack([(xx-k["cx"])*depth/k["fx"],
                                (yy-k["cy"])*depth/k["fy"],depth],axis=-1)
        world=camera_points@poses[i,:3,:3].T+poses[i,:3,3]
        rgb=(image["img"][0].permute(1,2,0).numpy()+1)/2
        valid=mask&np.isfinite(world).all(-1)&(depth>0)
        points.append(world[valid]); colors.append(rgb[valid])
    points=np.concatenate(points).astype(np.float32)
    colors=np.concatenate(colors).astype(np.float32)
    if len(points)<100: raise RuntimeError("Too few reliable DUSt3R points")
    original_count=len(points)
    if len(points)>max_points:
        keep=np.random.default_rng(42).choice(len(points),max_points,replace=False)
        points,colors=points[keep],colors[keep]
    output.mkdir(parents=True)
    write_ply(output/"points3D.ply",points,colors)
    np.savez_compressed(output/"points.npz",points=points,colors=colors)
    receipt=dict(status="dust3r_known_pose_initialization_complete",
        protocol_sha256=data["protocol_sha256"],source_bundle_sha256=data["bundle_sha256"],
        bundle_path=str(checked(bundle_path)),source_frame_ids=[r["frame_id"] for r in observations],
        seed_frame_ids=[r["frame_id"] for r in rows],seed_indices=indices,global_alignment_iterations=iterations,
        initialization_points=len(points),original_reliable_points=original_count,max_points=max_points,
        known_poses_fixed=True,original_sfm_points_used=False,
        confidence_policy=confidence_policy,point_confidence_threshold=float(scene.min_conf_thr),
        focal_optimization_convention="fixed mean fx/fy; exact anisotropic K for depth backprojection",
        ply_sha256=sha256_file(output/"points3D.ply"),npz_sha256=sha256_file(output/"points.npz"))
    write_json(output/"receipt.json",receipt)
    print(json.dumps(receipt,indent=2),flush=True)
    return receipt


def export_colmap_scene(bundle_path,initialization,output):
    bundle=read_bundle(bundle_path); receipt=load_json(Path(initialization)/"receipt.json")
    if receipt["source_bundle_sha256"]!=bundle["bundle_sha256"]:
        raise ValueError("Initialization belongs to another observation bundle")
    if sha256_file(Path(initialization)/"points3D.ply")!=receipt["ply_sha256"]:
        raise ValueError("Initialization point cloud changed before export")
    output=checked(output,must_exist=False)
    if output.exists(): raise FileExistsError(output)
    (output/"images").mkdir(parents=True)
    sparse=output/"sparse"/"0"; sparse.mkdir(parents=True)
    rows=bundle["observations"]
    with (sparse/"cameras.bin").open("xb") as cams,(sparse/"images.bin").open("xb") as imgs:
        cams.write(struct.pack("<Q",len(rows))); imgs.write(struct.pack("<Q",len(rows)))
        for i,row in enumerate(rows,1):
            c=row["camera"]; name=Path(row["image_path"]).name
            (output/"images"/name).symlink_to(checked(row["image_path"]))
            cams.write(struct.pack("<iiQQ4d",i,1,c["width"],c["height"],c["fx"],c["fy"],c["cx"],c["cy"]))
            q=Rotation.from_matrix(c["R"]).as_quat(); qwxyz=[q[3],q[0],q[1],q[2]]
            imgs.write(struct.pack("<i4d3di",i,*qwxyz,*c["T"],i))
            imgs.write(name.encode()+b"\0"+struct.pack("<Q",0))
    import shutil
    shutil.copyfile(checked(Path(initialization)/"points3D.ply"),sparse/"points3D.ply")
    write_json(output/"provenance.json",dict(protocol_sha256=bundle["protocol_sha256"],
        source_bundle_sha256=bundle["bundle_sha256"],source_frame_ids=[r["frame_id"] for r in rows],
        initialization=receipt,original_sparse_points_used=False,target_truth_exported=False))
    return str(output)


if __name__=="__main__":
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="stage",required=True)
    d=sub.add_parser("dust3r"); d.add_argument("--bundle",required=True); d.add_argument("--output",required=True)
    d.add_argument("--iterations",type=int,default=300)
    d.add_argument("--seed-indices",type=int,nargs="+")
    d.add_argument("--confidence-policy",choices=["strict","reference"],default="strict")
    d=sub.add_parser("export"); d.add_argument("--bundle",required=True); d.add_argument("--initialization",required=True); d.add_argument("--output",required=True)
    a=p.parse_args()
    if a.stage=="dust3r":
        from .initialization_cache import initialize_once
        initialize_once(a.bundle,a.output,iterations=a.iterations,seed_indices=a.seed_indices,confidence_policy=a.confidence_policy)
    else: print(export_colmap_scene(a.bundle,a.initialization,a.output))
