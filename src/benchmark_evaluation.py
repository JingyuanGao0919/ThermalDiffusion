                                                                        
from __future__ import annotations
import os
import argparse, json, math, platform, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from .evaluation import paired_ids
from .io import checked, load_json, sha256_file, write_json
from .protocol import validate_saved

ROOT = Path(os.environ["THERMAL_ROOT"])
HEIGHT, WIDTH = 320, 512


def _load_gray(path, *, root, expected_sha256=None):
    path = checked(path, root=root)
    digest = sha256_file(path, root=root)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f'SHA256 mismatch: {path}')
    with Image.open(path) as image:
        image.load()
        array = np.asarray(image.convert('L').resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR), dtype=np.float32).copy() / 255.0
    if array.shape != (HEIGHT, WIDTH) or not np.isfinite(array).all():
        raise ValueError(f'Invalid grayscale frame: {path}')
    return array, digest


def _psnr(pred, target, mask=None):
    pred, target = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    select = np.ones(pred.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if pred.shape != target.shape or select.shape != pred.shape or not select.any():
        return None
    mse = float(np.mean(np.square(pred[select] - target[select])))
    return float('inf') if mse == 0 else -10.0 * math.log10(mse)


def _score(pred, target, *, lpips_model, device):
    import torch
    from pytorch_msssim import ssim
    p = torch.from_numpy(np.asarray(pred, dtype=np.float32)).to(device)[None, None]
    g = torch.from_numpy(np.asarray(target, dtype=np.float32)).to(device)[None, None]
    ssim_value = float(ssim(p, g, data_range=1.0, size_average=True).item())
                                                                                   
    p3 = p.expand(-1, 3, -1, -1) * 2.0 - 1.0
    g3 = g.expand(-1, 3, -1, -1) * 2.0 - 1.0
    with torch.no_grad():
        lpips_value = float(lpips_model(p3, g3).mean().item())
    return {'PSNR': _psnr(pred, target), 'SSIM': ssim_value, 'LPIPS': lpips_value}


def _mean(values):
    values = [float(x) for x in values if x is not None]
    return float(np.mean(values)) if values else None


def _serial_metric(value):
    if value is None: return None
    if math.isinf(value): return 'Infinity' if value > 0 else '-Infinity'
    if not math.isfinite(value): raise ValueError('Non-finite metric')
    return float(value)


def evaluate(*, protocol_root, dataset_root, receipts_root, output, setting,
             clip_id=None, device='cuda', confidence_threshold=0.5):
    import torch, lpips
    protocol_root, dataset_root, receipts_root = map(checked, (protocol_root, dataset_root, receipts_root))
    output = checked(output, must_exist=False)
    if output.exists(): raise FileExistsError(output)
    if setting not in ('easy', 'hard'): raise ValueError('setting must be easy or hard')
    if not 0.0 <= confidence_threshold <= 1.0: raise ValueError('confidence threshold outside [0,1]')
    protocol = validate_saved(protocol_root)
    inventory = load_json(protocol_root/'inventory.json')['frames']
    rows = load_json(protocol_root/f'test_{setting}.json')['clips']
    if clip_id is not None:
        rows = [row for row in rows if row['clip_id'] == clip_id]
        if not rows: raise ValueError(f'Unknown {setting} clip: {clip_id}')
    expected_ids = [row['clip_id'] for row in rows]
    available = {p.name for p in receipts_root.iterdir() if p.is_dir()}
    missing, unexpected = set(expected_ids)-available, available-set(expected_ids)
    if missing or (clip_id is None and unexpected):
        raise ValueError(f'Receipt directory mismatch: missing={sorted(missing)[:5]}, unexpected={sorted(unexpected)[:5]}')
    if device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable for LPIPS')
    torch_device = torch.device(device)
    lpips_model = lpips.LPIPS(net='vgg', verbose=False).eval().to(torch_device)
    clip_results, receipt_summaries = [], []
    start = time.time()
    for row in rows:
        cid = row['clip_id']
        if len(row['frame_ids']) != 16 or len(row['target_frame_ids']) != 13: raise ValueError(f'Malformed clip: {cid}')
        if row['input_frame_ids'] != [row['frame_ids'][i] for i in (0,7,15)]: raise ValueError(f'Sparse input mismatch: {cid}')
        if set(row['input_frame_ids']) & set(row['target_frame_ids']): raise ValueError(f'Input/target overlap: {cid}')
        if set(row['input_frame_ids']) | set(row['target_frame_ids']) != set(row['frame_ids']): raise ValueError(f'Input/target coverage mismatch: {cid}')
        clip_dir = checked(receipts_root/cid, root=receipts_root)
        receipt_path = checked(clip_dir/'receipt.json', root=clip_dir)
        receipt = load_json(receipt_path)
        if receipt.get('clip_id') != cid or receipt.get('protocol_sha256') != protocol['protocol_sha256']:
            raise ValueError(f'Receipt identity/protocol mismatch: {cid}')
        if receipt.get('target_truth_used') is not False: raise ValueError(f'Target-truth boundary absent: {cid}')
        predictions = {}
        for frame in receipt.get('frames', []):
            fid = frame.get('frame_id')
            if not fid or fid in predictions: raise ValueError(f'Missing/duplicate frame ID: {cid}')
            if frame.get('source') != 'diffusion': raise ValueError(f'Non-diffusion prediction: {cid}/{fid}')
            predictions[fid] = _load_gray(frame['path'], root=clip_dir, expected_sha256=frame.get('sha256'))
        paired_ids(predictions, {fid: None for fid in row['frame_ids']})
        if list(predictions) != row['frame_ids']: raise ValueError(f'Prediction order mismatch: {cid}')
        proxy_dir = checked(clip_dir/'proxy', root=clip_dir)
        proxy_receipt_path = checked(proxy_dir/'receipt.json', root=proxy_dir)
        proxy_receipt = load_json(proxy_receipt_path)
        if proxy_receipt.get('clip_id') != cid or proxy_receipt.get('protocol_sha256') != protocol['protocol_sha256']:
            raise ValueError(f'Proxy identity/protocol mismatch: {cid}')
        if proxy_receipt.get('target_truth_used') is not False:
            raise ValueError(f'Proxy receipt does not certify target-truth exclusion: {cid}')
        controls_path = checked(proxy_dir/'controls.npz', root=proxy_dir)
        if sha256_file(controls_path, root=proxy_dir) != proxy_receipt.get('controls_sha256'):
            raise ValueError(f'Proxy controls checksum mismatch: {cid}')
        if proxy_receipt.get('rendered_frame_ids') != row['frame_ids']:
            raise ValueError(f'Proxy rendered frame order mismatch: {cid}')
        with np.load(controls_path, allow_pickle=False) as controls:
            if 'C' not in controls: raise ValueError(f'Missing confidence C: {cid}')
            confidence = np.asarray(controls['C'], dtype=np.float32)
        if confidence.shape == (16,1,HEIGHT,WIDTH): confidence = confidence[:,0]
        if confidence.shape != (16,HEIGHT,WIDTH) or not np.isfinite(confidence).all(): raise ValueError(f'Invalid confidence maps: {cid}')
        if (confidence < 0).any() or (confidence > 1).any(): raise ValueError(f'Confidence outside [0,1]: {cid}')
        if receipt.get('proxy_rendered_frame_ids', row['frame_ids']) != row['frame_ids']:
            raise ValueError(f'Confidence/frame alignment mismatch: {cid}')
        frame_rows = []
        for fid in row['target_frame_ids']:
            truth_info = inventory.get(fid)
            if truth_info is None or truth_info.get('split') != 'heldout': raise ValueError(f'Non-heldout target: {cid}/{fid}')
            truth, truth_digest = _load_gray(truth_info['path'], root=dataset_root, expected_sha256=truth_info['sha256'])
            index = row['frame_ids'].index(fid)
            pred, pred_digest = predictions[fid]
            full = _score(pred, truth, lpips_model=lpips_model, device=torch_device)
            obs_mask = confidence[index] > confidence_threshold
            hole_mask = ~obs_mask
            frame_rows.append(dict(frame_id=fid, prediction_sha256=pred_digest, target_sha256=truth_digest,
                full={k:_serial_metric(v) for k,v in full.items()},
                region_psnr={'observable':_serial_metric(_psnr(pred,truth,obs_mask)), 'hole':_serial_metric(_psnr(pred,truth,hole_mask))},
                region_pixels={'observable':int(obs_mask.sum()),'hole':int(hole_mask.sum())}))
        clip_metrics = {metric:_mean([x['full'][metric] for x in frame_rows]) for metric in ('PSNR','SSIM','LPIPS')}
        for region in ('observable','hole'):
            clip_metrics[region+'_PSNR'] = _mean([x['region_psnr'][region] for x in frame_rows])
        clip_results.append(dict(clip_id=cid,scene_name=row['scene_name'],group=row['group'],num_target_frames=len(frame_rows),
                                 metrics={k:_serial_metric(v) for k,v in clip_metrics.items()},frames=frame_rows))
        receipt_summaries.append(dict(clip_id=cid,receipt_sha256=sha256_file(receipt_path,root=clip_dir),
            receipt_purpose=receipt.get('purpose'),formal_paper_evaluation=receipt.get('formal_paper_evaluation')))
    metrics = ('PSNR','SSIM','LPIPS','observable_PSNR','hole_PSNR')
    scene_groups = {r['scene_name']:r['group'] for r in rows}
    by_scene = defaultdict(lambda: defaultdict(list))
    for clip in clip_results:
        for metric in metrics: by_scene[clip['scene_name']][metric].append(clip['metrics'][metric])
    scene_summary = {s:{m:_serial_metric(_mean(by_scene[s][m])) for m in metrics} for s in sorted(by_scene)}
    group_summary = {}
    for group in sorted({r['group'] for r in rows}):
        members = [s for s in scene_summary if scene_groups[s] == group]
        group_summary[group] = {m:_serial_metric(_mean([scene_summary[s][m] for s in members])) for m in metrics}
    required_groups = ('Indoor','Outdoor','UAV')
    scenario_macro = ({m:_serial_metric(_mean([group_summary[g][m] for g in required_groups])) for m in metrics}
                      if all(g in group_summary for g in required_groups) else None)
    result = dict(schema_version=1, protocol_sha256=protocol['protocol_sha256'], setting=setting,
        scope='single_clip_pilot' if clip_id else 'full_setting', num_clips=len(clip_results),
        num_target_frames=sum(x['num_target_frames'] for x in clip_results),
        target_truth_used_for_model=False,target_truth_used_for_metrics=True,
        evaluation_resize=dict(height=HEIGHT,width=WIDTH,interpolation='PIL bilinear'),
        input_range='8-bit grayscale / 255 -> [0,1]',ssim=dict(implementation='pytorch_msssim.ssim',data_range=1.0),
        lpips=dict(backbone='VGG',package_version=getattr(lpips,'__version__','unknown'),grayscale_policy='repeat 3 channels; map [0,1] to [-1,1]'),
        region_mask=dict(source='coarse 3DGS proxy confidence C',observable=f'C > {confidence_threshold}',hole=f'C <= {confidence_threshold}',threshold=confidence_threshold),
        aggregation='mean target-frame metrics within clip; macro over clips within scene; equal macro over scenes within group; equal macro over Indoor/Outdoor/UAV',
        groups=group_summary,scenes=scene_summary,scenario_macro=scenario_macro,receipts=receipt_summaries,clips=clip_results,
        environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,device=str(torch_device)),
        elapsed_seconds=time.time()-start)
    write_json(output,result)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description='Leakage-aware TI-NSD PSNR/SSIM/LPIPS evaluator')
    p.add_argument('--protocol-root',default=str(ROOT/'protocol_v1'))
    p.add_argument('--dataset-root',default=os.environ["OBS_ROOT"])
    p.add_argument('--receipts-root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--setting',required=True,choices=['easy','hard'])
    p.add_argument('--clip-id')
    p.add_argument('--device',default='cuda')
    p.add_argument('--confidence-threshold',type=float,default=.5)
    a=p.parse_args()
    print(json.dumps(evaluate(protocol_root=a.protocol_root,dataset_root=a.dataset_root,receipts_root=a.receipts_root,
        output=a.output,setting=a.setting,clip_id=a.clip_id,device=a.device,confidence_threshold=a.confidence_threshold),
        indent=2,allow_nan=False))
