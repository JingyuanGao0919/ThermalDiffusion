                                                                                  
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from .io import checked,load_json,write_json,sha256_file
from .initialization import read_bundle
from .lora import inject,load
from .ddim import sample
from .physics import guidance_energy
from .conditioning import build_conditioning

ROOT=Path(os.environ["THERMAL_ROOT"])


def run(sparse_dir,adapter,output,*,setting="easy",formal=False):
    sparse_dir=checked(sparse_dir);adapter=checked(adapter);output=checked(output,must_exist=False)
    if output.exists():raise FileExistsError(output)
    if setting not in ("easy", "hard"):
        raise ValueError("setting must be easy or hard")
    proxy=load_json(sparse_dir/"proxy/receipt.json");source=load_json(sparse_dir/"source_fields/receipt.json")
    if sha256_file(sparse_dir/"proxy/controls.npz")!=proxy["controls_sha256"]:raise ValueError("Controls changed")
    if sha256_file(sparse_dir/"source_fields/q.npz")!=source["q_sha256"]:raise ValueError("Source fields changed")
    if source["frame_ids"]!=proxy["rendered_frame_ids"] or source["source_controls_sha256"]!=proxy["controls_sha256"]:
        raise ValueError("Source/trajectory alignment mismatch")
    test_doc=load_json(ROOT/"protocol_v1"/("test_"+setting+".json"))
    clip=next((row for row in test_doc["clips"] if row["clip_id"]==proxy["clip_id"]),None)
    if clip is None:raise ValueError("Proxy clip is outside the selected benchmark setting")
    if proxy["rendered_frame_ids"]!=clip["frame_ids"] or proxy["training_frame_ids"]!=clip["input_frame_ids"]:
        raise ValueError("Proxy frame/input mapping differs from the frozen test protocol")
    bundle=read_bundle(ROOT/"inputs_v1/test"/setting/proxy["clip_id"]/"bundle.json")
    if bundle["bundle_sha256"]!=proxy["source_bundle_sha256"]:raise ValueError("Input bundle changed")
    adapter_header=load_json(adapter.with_suffix(adapter.suffix+".json"))
    adapter_meta=adapter_header.get("metadata",{})
    if formal and (adapter_meta.get("formal_training") is not True or adapter_meta.get("phase")!="refit" or
                   adapter_meta.get("protocol_sha256")!=proxy["protocol_sha256"]):
        raise ValueError("Formal inference requires a completed refit adapter from this protocol")
    if not formal and adapter_meta.get("formal_training") is True:
                                                                                      
                                                                                       
        pass
    with (ROOT/"runs/pseudo_adaptation_v1/gpu_inference.lock").open("a") as slot:
        fcntl.flock(slot,fcntl.LOCK_EX|fcntl.LOCK_NB)
        free=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
        if free<48*1024:raise RuntimeError("Need 48 GiB free for guided inference integration")
        torch.cuda.set_per_process_memory_fraction(.5);torch.cuda.reset_peak_memory_stats()
        torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        output.mkdir(parents=True)
        state=dict(pid=os.getpid(),stage="loading_backbone",
            purpose="formal_sparse_tir_inference" if formal else "pilot_lora_guided_inference_integration",
            formal_adapter=bool(formal),formal_paper_evaluation=False,target_truth_used=False,completed_steps=0,total_steps=20,failed=None)
        def status(stage):
            state.update(stage=stage,updated_unix=time.time(),peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
            write_json(output/"status.json",state,overwrite=True)
        try:
            status("loading_backbone")
            os.environ["HF_HUB_OFFLINE"]="1";os.environ["TRANSFORMERS_OFFLINE"]="1"
            sys.path.insert(0,str(checked(os.environ["VIEWCRAFTER_ROOT"])))
            from omegaconf import OmegaConf
            from utils.diffusion_utils import instantiate_from_config,load_model_checkpoint
                                                                                 
                                                                                   
                                             
            config=ROOT/"runs/backbone_conditioning_audit_v1/official_raw_snapshot_v1/configs/inference_512_v1.0.yaml"
            base=checked(ROOT/"runs/backbone_conditioning_audit_v1/dynamicrafter_512/model.ckpt")
            base_hash=sha256_file(base);config_hash=sha256_file(config)
            cfg=OmegaConf.load(config).model;cfg.params.unet_config.params.use_checkpoint=True
            cfg.params.cond_stage_config.params.version=None;cfg.params.img_cond_stage_config.params.version=None
            model=load_model_checkpoint(instantiate_from_config(cfg),str(base))
            model.eval().requires_grad_(False).cuda();model.cond_stage_model.device="cuda";model.perframe_ae=True
            layers=inject(model.model.diffusion_model,rank=16,alpha=16.)
            expected=dict(protocol_sha256=proxy["protocol_sha256"],
                base_checkpoint_sha256=base_hash,base_config_sha256=config_hash)
            if formal:
                expected.update(formal_training=True,phase="refit",manifest_sha256=adapter_meta.get("manifest_sha256"))
            else:
                expected.update(purpose=adapter_meta.get("purpose"),formal_training=adapter_meta.get("formal_training"))
            metadata=load(adapter,layers,expected)
            model.requires_grad_(False)
            data=np.load(sparse_dir/"proxy/controls.npz");q=np.load(sparse_dir/"source_fields/q.npz")["q"]
            fields={key:torch.tensor(data[key],device="cuda").permute(1,0,2,3).unsqueeze(0) for key in ("S","D","C")}
            source_q=torch.tensor(q,device="cuda").permute(1,0,2,3).unsqueeze(0)
                                                                                
                                                                                  
                                                                        
            first_observation=bundle["observations"][0]
            if first_observation["frame_id"]!=clip["input_frame_ids"][0]:
                raise ValueError("Condition image is not the first protocol observation")
            with Image.open(checked(first_observation["image_path"])) as image_pil:
                pixels=np.asarray(image_pil.convert("L").resize((512,320),Image.Resampling.BILINEAR),dtype=np.float32).copy()/127.5-1
            image=torch.tensor(pixels,device="cuda").unsqueeze(0).unsqueeze(0).expand(-1,3,-1,-1).contiguous()
            with torch.cuda.amp.autocast():
                cond,uncond=build_conditioning(model,image,frames=len(clip["frame_ids"]))
                concat=cond["c_concat"][0]
            del image
            generator=torch.Generator(device="cuda").manual_seed(42)
            initial=torch.randn(concat.shape,device="cuda",generator=generator,dtype=torch.float32)
            fs=torch.tensor([10],device="cuda",dtype=torch.long);current={};started=time.monotonic()
            def predict_v(z,t):
                current["timestep"]=t
                times=torch.tensor([t],device="cuda",dtype=torch.long)
                with torch.cuda.amp.autocast():
                    v_cond=model.apply_model(z,times,cond,fs=fs)
                    v_null=model.apply_model(z,times,uncond,fs=fs)
                return (v_null+7.5*(v_cond-v_null)).float()
            def decode_frame(z):
                with torch.cuda.amp.autocast(enabled=False):
                    rgb=model.differentiable_decode_first_stage(z.float())
                    return (rgb.mean(1,keepdim=True)+1)*.5
            def decode(z):
                                                                                   
                                                                             
                return torch.cat([torch_checkpoint(decode_frame,z[:,:,i:i+1],use_reentrant=False)
                                  for i in range(z.shape[2])],dim=2)
            with (output/"steps.jsonl").open("x") as log:
                def energy(pixels):
                    value=guidance_energy(pixels,fields["S"],fields["D"],fields["C"],source_q)
                    current["energy"]=float(value.detach())
                    return value
                def completed(index,timestep):
                    state["completed_steps"]=index+1
                    row=dict(step=index+1,timestep=timestep,energy=current["energy"],elapsed_seconds=time.monotonic()-started)
                    log.write(json.dumps(row)+"\n");log.flush();print(json.dumps(row),flush=True)
                    status("guided_sampling")
                latent=sample(model_v=predict_v,initial_noise=initial,alphas_cumprod=model.alphas_cumprod,
                    steps=20,decode=decode,energy=energy,eta=0.,gamma=.2,clip_norm=1.,
                    dynamic_scales=model.scale_arr,callback=completed)
            status("decoding_and_saving")
            with torch.no_grad():pixels=decode(latent).clamp(0,1)[0,0].cpu().numpy()
            if not np.isfinite(pixels).all():raise RuntimeError("Nonfinite generated frames")
            (output/"frames").mkdir();frames=[]
            for index,(frame_id,array) in enumerate(zip(proxy["rendered_frame_ids"],pixels)):
                path=output/"frames"/f"{index:02d}.png"
                Image.fromarray(np.rint(array*255).astype(np.uint8)).save(path)
                frames.append(dict(frame_id=frame_id,path=str(path),sha256=sha256_file(path),source="diffusion"))
            if any(p.grad is not None for p in model.parameters()):raise RuntimeError("Frozen inference prior received gradients")
            result=dict(purpose=state["purpose"],formal_adapter=bool(formal),formal_paper_evaluation=False,protocol_sha256=proxy["protocol_sha256"],
                clip_id=proxy["clip_id"],setting=setting,frames=frames,training_input_frame_ids=proxy["training_frame_ids"],
                source_controls_sha256=proxy["controls_sha256"],source_q_sha256=source["q_sha256"],
                base_checkpoint_sha256=base_hash,adapter_sha256=sha256_file(adapter),adapter_metadata=metadata,
                conditioning="first observed frame repeated in c_concat; proxy controls used only for DSPG/REA",
                ddim_steps=20,cfg=7.5,eta=0.,gamma=.2,gradient_clip_norm=1.,
                dynamic_scales="actual model buffer; transition and decoder use separate units",
                vae_guidance_precision="float32 with per-frame activation checkpointing",target_truth_used=False,
                elapsed_seconds=time.monotonic()-started,peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
            write_json(output/"receipt.json",result);status("guided_inference_integration_complete")
            return result
        except Exception as error:state["failed"]=str(error);status("failed");raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--sparse-dir",required=True);p.add_argument("--adapter",required=True);p.add_argument("--output",required=True)
    p.add_argument("--setting",required=True,choices=["easy","hard"]);p.add_argument("--formal",action="store_true")
    a=p.parse_args();print(json.dumps(run(a.sparse_dir,a.adapter,a.output,setting=a.setting,formal=a.formal),indent=2))
