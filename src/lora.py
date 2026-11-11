                                                                                    
from __future__ import annotations
import os
from pathlib import Path
import torch
from torch import nn
from .io import checked,load_json,write_json,sha256_file

TARGETS = tuple(f"{attn}.{proj}" for attn in ("attn1","attn2")
                for proj in ("to_q","to_k","to_v","to_out"))


class LoRALinear(nn.Module):
    def __init__(self,base,rank=16,alpha=16.):
        super().__init__()
        if rank<=0 or alpha<=0: raise ValueError("Invalid LoRA rank/alpha")
        self.base,self.rank,self.alpha = base,rank,alpha
        self.base.requires_grad_(False)
        self.A=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device))
        self.B=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device))
        nn.init.kaiming_uniform_(self.A,a=5**.5)

    def forward(self,x):
        base = self.base(x)
        adapted = (x.to(self.A.dtype)@self.A.T)@self.B.T
        return base+(self.alpha/self.rank)*adapted.to(base.dtype)


def inject(model,*,rank=16,alpha=16.,targets=TARGETS):
    model.requires_grad_(False)
    replacements = [(name,m) for name,m in model.named_modules()
                    if isinstance(m,nn.Linear) and any(x in name for x in targets)]
    if not replacements: raise ValueError("No matching attention projection layers")
    layers={}
    for name,base in replacements:
        parent,_,attribute=name.rpartition(".")
        module=model.get_submodule(parent) if parent else model
        layer=LoRALinear(base,rank,alpha)
        setattr(module,attribute,layer)
        layers[name]=layer
    return layers


def save(path,layers,metadata,*,overwrite=False):
    required={"protocol_sha256","manifest_sha256","base_checkpoint_sha256","base_config_sha256",
              "selection_metric","epoch_zero_based","train_loss","seed"}
    if not required<=set(metadata): raise ValueError(f"Missing LoRA provenance: {required-set(metadata)}")
    path=checked(path,must_exist=False)
    path.parent.mkdir(parents=True,exist_ok=True)
    sidecar=path.with_suffix(path.suffix+".json")
    if (path.exists() or sidecar.exists()) and not overwrite: raise FileExistsError(path)
                                                                              
                                                                                  
    payload={name:{"A":layer.A.detach().cpu(),"B":layer.B.detach().cpu()}
             for name,layer in layers.items()}
    tmp=path.with_name(path.name+f".tmp-{os.getpid()}")
    with tmp.open("xb") as stream: torch.save(payload,stream,pickle_protocol=2)
    os.replace(tmp,path)
    write_json(sidecar,dict(schema_version=1,metadata=metadata,
        tensor_archive_sha256=sha256_file(path),
        layer_specs={name:{"rank":layer.rank,"alpha":layer.alpha} for name,layer in layers.items()}),
        overwrite=overwrite)


def load(path,layers,expected):
    path=checked(path)
    header=load_json(path.with_suffix(path.suffix+".json"))
    if header.get("schema_version")!=1: raise ValueError("Unsupported/unprovenanced LoRA checkpoint")
    if header["tensor_archive_sha256"]!=sha256_file(path): raise ValueError("LoRA archive integrity failure")
    for key,value in expected.items():
        if header["metadata"].get(key)!=value: raise ValueError(f"LoRA provenance mismatch: {key}")
    payload=torch.load(path,map_location="cpu",weights_only=True)
    if set(payload)!=set(layers): raise ValueError("LoRA module mapping mismatch")
    for name,layer in layers.items():
        spec=header["layer_specs"][name]
        if spec!={"rank":layer.rank,"alpha":layer.alpha}: raise ValueError("LoRA scaling mismatch")
        for key,param in (("A",layer.A),("B",layer.B)):
            if payload[name][key].shape!=param.shape: raise ValueError("LoRA tensor shape mismatch")
    with torch.no_grad():
        for name,layer in layers.items():
            for key,param in (("A",layer.A),("B",layer.B)):
                param.copy_(payload[name][key])
    return header["metadata"]
