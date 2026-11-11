                                                                                  
from __future__ import annotations
import gc
import os
from pathlib import Path
import sys
import torch
from .io import checked, load_json, sha256_file, canonical_hash

ROOT = Path(os.environ["THERMAL_ROOT"])
AUDIT = ROOT / 'runs/backbone_conditioning_audit_v1'


def load_official():
    receipt = load_json(AUDIT / 'download_status.json')
    if receipt['stage'] != 'official_checkpoint_hash_verified':
        raise ValueError('Official checkpoint is not verified')
    checkpoint = checked(receipt['checkpoint'], root=AUDIT)
    if sha256_file(checkpoint) != receipt['expected_sha256']:
        raise ValueError('Official checkpoint changed')
    source = load_json(AUDIT / 'official_code_source.json')
    if not source.get('source_reviewed'):
        raise ValueError('Official source snapshot is not reviewed')
    code_root = checked(source['root'], root=AUDIT)
    for name, digest in source['files'].items():
        if sha256_file(code_root / name, root=code_root) != digest:
            raise ValueError('Official source changed: ' + name)
    validation = load_json(AUDIT / 'component_validation_v2/receipt.json')
    if not validation.get('strict_state_dict_load') or not validation.get('train_inference_condition_exact_match'):
        raise ValueError('Official backbone component validation is incomplete')
    if validation['checkpoint_sha256'] != receipt['expected_sha256'] or validation['official_code_snapshot_sha256'] != source['source_snapshot_sha256']:
        raise ValueError('Component validation belongs to different assets')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path.insert(0, str(code_root))
    from omegaconf import OmegaConf
    from utils.utils import instantiate_from_config
    config = code_root / 'configs/inference_512_v1.0.yaml'
    if sha256_file(config) != validation['config_sha256']:
        raise ValueError('Official model configuration changed')
    cfg = OmegaConf.load(config).model
    if cfg.target != 'lvdm.models.ddpm3d.LatentVisualDiffusion' or cfg.params.base_scale != .7:
        raise ValueError('Unexpected official configuration')
    cfg.params.rand_cond_frame = False
    cfg.params.uncond_prob = .05
    cfg.params.unet_config.params.use_checkpoint = True
    cfg.params.cond_stage_config.params.version = None
    cfg.params.img_cond_stage_config.params.version = None
    effective_hash = canonical_hash(OmegaConf.to_container(cfg, resolve=True))
    model = instantiate_from_config(cfg)
    for name, module in list(sys.modules.items()):
        if name.startswith(('lvdm.', 'utils.')) and getattr(module, '__file__', None):
            checked(module.__file__, root=code_root)
    payload = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(payload['state_dict'] if 'state_dict' in payload else payload, strict=True)
    del payload
    gc.collect()
    model.eval().requires_grad_(False).cuda()
    model.cond_stage_model.device = 'cuda'
    model.perframe_ae = True
    if model.parameterization != 'v' or not model.use_dynamic_rescale:
        raise ValueError('Unexpected backbone prediction/scaling')
    provenance = dict(base_checkpoint_sha256=receipt['expected_sha256'],
        base_config_sha256=sha256_file(config), effective_config_sha256=effective_hash,
        official_code_snapshot_sha256=source['source_snapshot_sha256'],
        official_component_validation_sha256=sha256_file(AUDIT / 'component_validation_v2/receipt.json'),
        conditioning_implementation_sha256=sha256_file(Path(__file__).with_name('conditioning.py')),
        base_scale_min=float(model.scale_arr.min()), base_scale_max=float(model.scale_arr.max()))
    return model, provenance
