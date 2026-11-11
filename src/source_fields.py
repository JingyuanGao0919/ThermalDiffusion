                                                                                
import argparse
from pathlib import Path
import numpy as np
import torch
from .io import checked,load_json,write_json,sha256_file
from .physics import solve_source


def run(proxy,output):
    proxy=checked(proxy);output=checked(output,must_exist=False)
    if output.exists():raise FileExistsError(output)
    receipt=load_json(proxy/"receipt.json");path=proxy/"controls.npz"
    if sha256_file(path)!=receipt["controls_sha256"]:raise ValueError("Controls changed")
    fields=np.load(path);torch.set_num_threads(4)
    output.mkdir(parents=True);sources=[]
    for index,(s,d,c) in enumerate(zip(fields["S"],fields["D"],fields["C"])):
        q=solve_source(torch.from_numpy(s),torch.from_numpy(d),torch.from_numpy(c))
        sources.append(q.numpy());print("source",index+1,"/",len(fields["S"]),flush=True)
    np.savez_compressed(output/"q.npz",q=np.stack(sources))
    result=dict(clip_id=receipt["clip_id"],protocol_sha256=receipt["protocol_sha256"],
        frame_ids=receipt["rendered_frame_ids"],source_controls_sha256=receipt["controls_sha256"],
        q_sha256=sha256_file(output/"q.npz"),shape=list(np.stack(sources).shape),
        resolution="full 320x512",solver="CPU sparse direct solve with checked residual",target_truth_used=False)
    write_json(output/"receipt.json",result)
    return result


if __name__=="__main__":
    import json
    p=argparse.ArgumentParser();p.add_argument("--proxy",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();print(json.dumps(run(a.proxy,a.output),indent=2))
