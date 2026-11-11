                                                                             
import fcntl
from pathlib import Path
from .io import checked,load_json,sha256_file


def initialize_once(bundle_path,output,**kwargs):
    from .initialization import read_bundle,dust3r_initialize
    output=checked(output,must_exist=False);output.parent.mkdir(parents=True,exist_ok=True)
    with (output.parent/(output.name+".lock")).open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (output/"receipt.json").exists():
            bundle=read_bundle(bundle_path);receipt=load_json(output/"receipt.json")
            if receipt["source_bundle_sha256"]!=bundle["bundle_sha256"]:
                raise ValueError("Cached initialization source mismatch")
            if sha256_file(output/"points3D.ply")!=receipt["ply_sha256"]:
                raise ValueError("Cached initialization point cloud changed")
            if "npz_sha256" in receipt and sha256_file(output/"points.npz")!=receipt["npz_sha256"]:
                raise ValueError("Cached initialization array changed")
            print("Reusing hash-verified initialization",str(output),flush=True)
            return receipt
        if output.exists():raise RuntimeError("Partial initialization output requires inspection")
        import torch
        if torch.cuda.is_available():torch.cuda.set_per_process_memory_fraction(.15)
                                                                                
                                                                             
        bundle=read_bundle(bundle_path)
        n=len(bundle["observations"])
        initial=kwargs.get("seed_indices")
        candidates=[initial]
        if bundle.get("stage")=="pseudo_proxy_adaptation_only":
            stride=min(8,(n-1)//2)
            for fraction in (.5,.25,.75):
                center=max(stride,min(n-1-stride,int((n-1)*fraction)))
                candidates.append([center-stride,center,center+stride])
        attempted=set()
        for indices in candidates:
            key=tuple(indices) if indices is not None else None
            if key in attempted:continue
            attempted.add(key)
            options={**kwargs,"seed_indices":indices}
            try:
                return dust3r_initialize(bundle_path,output,**options)
            except RuntimeError as error:
                if str(error)!="Too few reliable DUSt3R points" or output.exists():raise
                print("Reliable-point guard failed for seeds",indices,flush=True)
                                                                            
                                                                    
            import gc
            gc.collect();torch.cuda.empty_cache()
        raise RuntimeError("Too few reliable DUSt3R points")
