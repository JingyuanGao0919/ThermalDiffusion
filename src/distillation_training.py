                                                                             

                                                                               
                                                                   
   
from __future__ import annotations
import os
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from .distillation import distillation_loss
from .initialization import read_bundle
from .io import checked, load_json, sha256_file, write_json
from .protocol import validate_saved
from .raster import make_camera, render_thermal
from .sparse_proxy import verify_sparse_contract, OPT

ROOT = Path(os.environ["THERMAL_ROOT"])


def verify_generated_contract(bundle, clip, proxy, generated):
    verify_sparse_contract(bundle, clip)
    if len(clip['frame_ids']) != 16 or len(set(clip['frame_ids'])) != 16:
        raise ValueError('Expected sixteen unique trajectory frames')
    if proxy['source_bundle_sha256'] != bundle['bundle_sha256']:
        raise ValueError('Proxy/bundle source mismatch')
    if generated['clip_id'] != clip['clip_id'] or proxy['clip_id'] != clip['clip_id']:
        raise ValueError('Diffusion/proxy clip mismatch')
    if generated.get('target_truth_used') is not False or proxy.get('target_truth_used') is not False:
        raise ValueError('Unverified target-truth boundary')
    if generated['source_controls_sha256'] != proxy['controls_sha256']:
        raise ValueError('Diffusion used different proxy controls')
    if generated['training_input_frame_ids'] != clip['input_frame_ids']:
        raise ValueError('Diffusion observations differ from the sparse protocol')
    frames = generated.get('frames', [])
    if [x['frame_id'] for x in frames] != clip['frame_ids']:
        raise ValueError('Missing, duplicated, or reordered diffusion frames')
    if proxy['rendered_frame_ids'] != clip['frame_ids']:
        raise ValueError('Confidence/camera trajectory mismatch')
    if any(x.get('source') != 'diffusion' for x in frames):
        raise ValueError('Generated targets must be frozen diffusion outputs')


def fit(sparse_dir, sampling_dir, output, *, iterations=4000):
    if iterations not in (10, 4000):
        raise ValueError('Use 10 steps for component smoke or the full 4000-step pilot')
    sparse_dir = checked(sparse_dir, root=ROOT)
    sampling_dir = checked(sampling_dir, root=ROOT)
    output = checked(output, root=ROOT, must_exist=False)
    if output.exists():
        raise FileExistsError(output)
    valid = validate_saved(ROOT / 'protocol_v1')
    proxy = load_json(sparse_dir / 'proxy/receipt.json')
    generated = load_json(sampling_dir / 'receipt.json')
    setting = proxy['clip_id'].split('_', 1)[0]
    bundle = read_bundle(ROOT / 'inputs_v1/test' / setting / proxy['clip_id'] / 'bundle.json')
    clip = next(x for x in load_json(ROOT / 'protocol_v1' / ('test_' + setting + '.json'))['clips'] if x['clip_id'] == proxy['clip_id'])
    verify_generated_contract(bundle, clip, proxy, generated)
    if proxy['protocol_sha256'] != valid['protocol_sha256'] or generated['protocol_sha256'] != valid['protocol_sha256']:
        raise ValueError('Protocol mismatch')
    ply = sparse_dir / 'proxy/point_cloud.ply'
    controls_path = sparse_dir / 'proxy/controls.npz'
    if sha256_file(ply) != proxy['point_cloud_sha256'] or sha256_file(controls_path) != proxy['controls_sha256']:
        raise ValueError('Changed coarse proxy or confidence controls')
    generated_paths = {}
    for row in generated['frames']:
        p = checked(row['path'], root=sampling_dir)
        if sha256_file(p) != row['sha256']:
            raise ValueError('Changed diffusion frame')
        generated_paths[row['frame_id']] = p
    for row in bundle['observations']:
        if sha256_file(row['image_path']) != row['source_sha256']:
            raise ValueError('Changed observed input')
    q = ROOT / 'runs/pseudo_adaptation_v1'
    with (q / 'gpu_auxiliary.lock').open('a') as auxiliary, (q / 'gpu_inference.lock').open('a') as slot:
        fcntl.flock(auxiliary, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
        free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        if free < 16 * 1024:
            raise RuntimeError('Need 16 GiB free for isolated distillation')
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(.15)
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(42)
        sys.path.insert(0, str(checked(os.environ["GUIDEDVD_ROOT"])))
        from scene.gaussian_model import GaussianModel
        model = GaussianModel(SimpleNamespace(sh_degree=3, use_color=True, train_bg=False, prune_from_iter=500, dist_thres=10.))
        model.load_ply(str(ply))
        centers = np.stack([-np.array(x['camera']['R']).T @ x['camera']['T'] for x in bundle['observations']])
        extent = max(float(np.linalg.norm(centers - centers.mean(0), axis=1).max()) * 1.1, .01)
        model.spatial_lr_scale = extent
        model.training_setup(SimpleNamespace(**OPT))
        lookup = {x['frame_id']: x['camera'] for x in bundle['observations'] + bundle['target_cameras']}
        cameras = {fid: make_camera(fid, lookup[fid]) for fid in clip['frame_ids']}
        def pixels(path):
            with Image.open(checked(path)) as image:
                if image.size != (512, 320):
                    raise ValueError('Generated supervision must already match the calibrated render size')
                values = np.asarray(image.convert('L'), dtype=np.float32).copy() / 255
            return torch.tensor(values, device='cuda').unsqueeze(0)
        observations = {}
        for row in bundle['observations']:
            with Image.open(checked(row['image_path'])) as image:
                values = np.asarray(image.convert('L').resize((512, 320), Image.Resampling.BILINEAR), dtype=np.float32).copy() / 255
            observations[row['frame_id']] = dict(camera_id=row['frame_id'], image=torch.tensor(values, device='cuda').unsqueeze(0))
        with np.load(controls_path) as fields:
            confidence = torch.tensor(fields['C'], device='cuda')
        if confidence.shape != (16, 1, 320, 512):
            raise ValueError('Unexpected frozen confidence shape')
        targets = {fid: dict(camera_id=fid, image=pixels(generated_paths[fid]), confidence=confidence[i].detach(), source='diffusion')
                   for i, fid in enumerate(clip['frame_ids']) if fid in clip['target_frame_ids']}
        if len(observations) != 3 or len(targets) != 13:
            raise ValueError('Distillation must preserve three observations and thirteen generated targets')
        def render(fid, camera_id):
            if fid != camera_id:
                raise ValueError('Camera identity mismatch')
            return dict(frame_id=fid, camera_id=camera_id, image=render_thermal(cameras[fid], model)['image'])
        output.mkdir(parents=True)
        state = dict(stage='distillation', iteration=0, total_iterations=iterations, failed=None,
            formal_paper_evaluation=False, target_truth_used=False, purpose='single_clip_distillation_pilot',
            coarse_iterations=proxy['iterations'], optimizer_reset_at_distillation=True,
            optimizer_convention='new Adam from saved coarse weights; prior optimizer was not serialized',
            densification=False, lambda_gen=.3, observed_views=3, generated_views=13)
        def status(stage):
            state.update(stage=stage, updated_unix=time.time(), peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
            write_json(output / 'status.json', state, overwrite=True)
        started = time.monotonic()
        try:
            with (output / 'steps.jsonl').open('x') as log:
                for iteration in range(1, iterations + 1):
                    model.update_learning_rate(proxy['iterations'] + iteration)
                    loss = distillation_loss(observations, targets, render, lambda_gen=.3)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite distillation loss')
                    loss.backward()
                    grads = [p.grad for group in model.optimizer.param_groups for p in group['params'] if p.grad is not None]
                    if not grads or not all(torch.isfinite(g).all() for g in grads):
                        raise RuntimeError('Absent/nonfinite distillation gradients')
                    model.optimizer.step()
                    model.optimizer.zero_grad(set_to_none=True)
                    if iteration % 1000 == 0:
                        for key in ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity'):
                            if not torch.isfinite(getattr(model, key)).all():
                                raise RuntimeError('Nonfinite intermediate proxy parameters')
                        partial = output / ('point_cloud_%04d.ply' % iteration)
                        model.save_ply(str(partial))
                        write_json(output / ('checkpoint_%04d.json' % iteration),
                            dict(iteration=iteration, point_cloud_sha256=sha256_file(partial),
                                 parameters_finite=True, optimizer_state_saved=False,
                                 resumable_optimizer_checkpoint=False))
                    if iteration == 1 or iteration % 10 == 0 or iteration == iterations:
                        state.update(iteration=iteration, last_loss=float(loss))
                        status('distillation')
                        row = dict(iteration=iteration, loss=float(loss), elapsed_seconds=time.monotonic()-started)
                        log.write(json.dumps(row)+'\n'); log.flush(); print(json.dumps(row), flush=True)
            for key in ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity'):
                if not torch.isfinite(getattr(model, key)).all():
                    raise RuntimeError('Nonfinite distilled proxy parameters')
            final = output / 'point_cloud.ply'
            model.save_ply(str(final))
            (output / 'frames').mkdir()
            predicted = []; raw_frames = []
            with torch.no_grad():
                for i, fid in enumerate(clip['frame_ids']):
                    array = render_thermal(cameras[fid], model)['image'].cpu().numpy()
                    if not np.isfinite(array).all():
                        raise RuntimeError('Nonfinite distilled rendering')
                    raw_frames.append(array)
                    path = output / 'frames' / ('%02d.png' % i)
                    Image.fromarray(np.rint(np.clip(array[0], 0, 1)*255).astype(np.uint8)).save(path)
                    predicted.append(dict(frame_id=fid, path=str(path), sha256=sha256_file(path)))
            np.savez_compressed(output / 'predictions.npz', frames=np.stack(raw_frames))
            result = dict(state, stage='distillation_pilot_complete', protocol_sha256=valid['protocol_sha256'], clip_id=clip['clip_id'],
                source_proxy_receipt_sha256=sha256_file(sparse_dir/'proxy/receipt.json'),
                source_diffusion_receipt_sha256=sha256_file(sampling_dir/'receipt.json'),
                point_cloud_sha256=sha256_file(final), gaussian_count=len(model.get_xyz),
                parameters_finite=True, optimizer_state_saved=False, predicted_frames=predicted,
                raw_predictions_sha256=sha256_file(output/'predictions.npz'), preview_clamp_range=[0, 1],
                generated_target_frame_ids=clip['target_frame_ids'], training_observation_frame_ids=clip['input_frame_ids'],
                elapsed_seconds=time.monotonic()-started, complete_4000_steps=(iterations == 4000))
            write_json(output / 'receipt.json', result)
            status('distillation_pilot_complete')
            return result
        except Exception as error:
            state['failed'] = repr(error); status('failed'); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sparse-dir', required=True)
    p.add_argument('--sampling-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--iterations', type=int, default=4000, choices=[10, 4000])
    a = p.parse_args()
    print(json.dumps(fit(a.sparse_dir, a.sampling_dir, a.output, iterations=a.iterations), indent=2))
