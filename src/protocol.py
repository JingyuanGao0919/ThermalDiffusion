                                                                                     
from __future__ import annotations

import math
import re
from pathlib import Path
from PIL import Image

from .colmap import read_cameras, read_images
from .io import checked, canonical_hash, load_json, sha256_file, write_json

SCENES = ("UAV1", "UAV2", "UAV3", "UAV4", "UAV5", "UAV6", "apples",
          "basketball_court", "bicycle", "bridge", "car", "chair", "corridor",
          "heated", "merge", "sitting", "soccer_goal", "standing", "tall_building", "wall")
OUTDOOR = {"bridge", "car", "soccer_goal", "tall_building", "wall"}
POLICY = {
    "version": "acmmm26-v1", "adapt_fraction": "65/100", "split_rounding": "floor",
    "chronology": "natural image filename order (not COLMAP image ID)",
    "clip_length": 16, "real_clip_step": 8, "input_indices": [0, 7, 15],
    "test_strides": {"easy": 1, "hard": 3},
    "test_clip_step_in_sampled_frames": 16, "test_tail": "drop incomplete clip",
    "test_window_status": "explicit reproduction convention; paper does not specify overlap",
    "target_size_hw": [320, 512], "expected_real_clips": 511,
    "expected_pseudo_clips": 3957, "expected_pseudo_frames_paper": 32278,
    "pseudo_strategies": ["slerp", "lookat_perturb", "lateral_shift"],
}


def natural_key(name):
    return tuple((0, int(x)) if x.isdigit() else (1, x.lower())
                 for x in re.split(r"(\d+)", name))


def group(scene):
    return "UAV" if scene.startswith("UAV") else "Outdoor" if scene in OUTDOOR else "Indoor"


def windows(n, split, *, stride=1, step=16):
    span = 1 + 15*stride
    for start in range(split, n-span+1, step*stride):
        yield list(range(start, start+span, stride))


def collect_scene(root, scene, *, verify_pixels=True):
    root = checked(root)
    scene_root = checked(root/scene, root=root)
    image_dir = checked(scene_root/"images", root=root)
    sparse = checked(scene_root/"sparse"/"0", root=root)
    cameras = read_cameras(checked(sparse/"cameras.bin", root=root))
    poses = read_images(checked(sparse/"images.bin", root=root))
    paths = sorted((p for p in image_dir.iterdir() if p.suffix.lower() in
                    {".jpg", ".jpeg", ".png"}), key=lambda p: natural_key(p.name))
    if not paths:
        raise ValueError(f"Empty scene {scene}")
    if set(p.name for p in paths) != set(poses):
        raise ValueError(f"Image/calibration names differ in {scene}: "
                         f"unregistered={sorted(set(p.name for p in paths)-set(poses))[:5]}, "
                         f"missing={sorted(set(poses)-set(p.name for p in paths))[:5]}")
    split = len(paths)*65//100
    frames = []
    for index, path in enumerate(paths):
        resolved = checked(path, root=root)
        pose = poses[path.name]
        if pose["camera_id"] not in cameras:
            raise ValueError(f"Missing camera {pose['camera_id']}")
        camera = {**cameras[pose["camera_id"]], "R": pose["R"], "T": pose["T"],
                  "convention": "COLMAP world_to_camera: x_cam=R*x_world+T"}
        with Image.open(resolved) as image:
            if image.size != (camera["width"], camera["height"]):
                raise ValueError(f"Image/calibration size mismatch: {path}")
            if verify_pixels:
                image.load()                                                            
        frames.append(dict(frame_id=f"{scene}/{path.name}", scene_name=scene,
                           frame_index=index, image_name=path.name, path=str(resolved),
                           sha256=sha256_file(resolved, root=root),
                           size_bytes=resolved.stat().st_size, camera=camera,
                           split="adapt" if index < split else "heldout"))
    return dict(scene_name=scene, group=group(scene), split_index=split,
                num_frames=len(frames), frames=frames,
                calibration_sha256={name: sha256_file(sparse/name, root=root)
                                    for name in ("cameras.bin", "images.bin")})


def build_manifests(scenes):
    real, clips, adaptation = [], {"easy": [], "hard": []}, []
    inventory = {f["frame_id"]: f for s in scenes for f in s["frames"]}
    for scene in scenes:
        frames, split = scene["frames"], scene["split_index"]
        name = scene["scene_name"]
        adapt = frames[:split]
        adaptation.append(dict(scene_name=name, group=scene["group"],
                               frame_ids=[f["frame_id"] for f in adapt]))
        for k, start in enumerate(range(0, split-15, 8)):
            selected = adapt[start:start+16]
            real.append(dict(clip_id=f"tinsd_{name}_real_{k:03d}", scene_name=name,
                             dataset_type="tinsd", clip_type="real",
                             frame_ids=[f["frame_id"] for f in selected],
                             frame_paths=[f["path"] for f in selected],
                             cameras=[f["camera"] for f in selected]))
        for setting, stride in POLICY["test_strides"].items():
            for k, indices in enumerate(windows(len(frames), split, stride=stride)):
                ids = [frames[i]["frame_id"] for i in indices]
                observed = [ids[j] for j in POLICY["input_indices"]]
                targets = [x for x in ids if x not in observed]
                clips[setting].append(dict(clip_id=f"{setting}_{name}_{k:03d}",
                    scene_name=name, group=scene["group"], stride=stride,
                    frame_ids=ids, input_frame_ids=observed, target_frame_ids=targets,
                                                                                             
                    cameras=[inventory[x]["camera"] for x in ids]))
    return inventory, real, clips, adaptation


def validate_protocol(inventory, real, tests):
    adapt = {x for x, f in inventory.items() if f["split"] == "adapt"}
    heldout = set(inventory)-adapt
    if not adapt or not heldout:
        raise ValueError("Empty adaptation or held-out region")
    for row in real:
        ids = row["frame_ids"]
        if len(ids) != 16 or not set(ids) <= adapt:
            raise ValueError(f"Adaptation leakage: {row['clip_id']}")
        if [inventory[x]["path"] for x in ids] != row["frame_paths"]:
            raise ValueError("Frame/path mapping changed")
        if [inventory[x]["camera"] for x in ids] != row["cameras"]:
            raise ValueError("Frame/camera mapping changed")
    for setting, rows in tests.items():
        used = set()
        for row in rows:
            ids = row["frame_ids"]
            inputs, targets = row["input_frame_ids"], row["target_frame_ids"]
            if len(ids) != 16 or len(set(ids)) != 16 or not set(ids) <= heldout:
                raise ValueError(f"Bad test window: {row['clip_id']}")
            if inputs != [ids[x] for x in (0, 7, 15)]:
                raise ValueError("Wrong sparse input positions")
            if targets != [x for x in ids if x not in inputs] or len(targets) != 13:
                raise ValueError("Wrong target set")
            if row["cameras"] != [inventory[x]["camera"] for x in ids]:
                raise ValueError("Target camera mismatch")
            indices = [inventory[x]["frame_index"] for x in ids]
            if any(b-a != POLICY["test_strides"][setting] for a,b in zip(indices, indices[1:])):
                raise ValueError("Wrong temporal stride")
            if used.intersection(ids):
                raise ValueError("Overlapping windows under declared non-overlap convention")
            used.update(ids)
                                                                                  
    adapt_hashes = {inventory[x]["sha256"] for x in adapt}
    overlap = [x for x in heldout if inventory[x]["sha256"] in adapt_hashes]
    if overlap:
        raise ValueError(f"Identical files across time split: {overlap[:5]}")


def prepare(dataset_root, destination, *, scenes=SCENES, verify_pixels=True,
            expected_real=511):
    destination = checked(destination, must_exist=False)
    if destination.exists():
        raise FileExistsError(f"Immutable protocol directory already exists: {destination}")
                                                                          
    collected = []
    for name in scenes:
        print(f"Validate images and calibration: {name}", flush=True)
        collected.append(collect_scene(dataset_root, name, verify_pixels=verify_pixels))
    inventory, real, tests, adaptation = build_manifests(collected)
    validate_protocol(inventory, real, tests)
    if expected_real is not None and len(real) != expected_real:
        raise ValueError(f"Real clip count {len(real)} != {expected_real}; do not pad samples")
    digest = canonical_hash({"policy": POLICY, "scenes": collected})
    metadata = dict(schema_version=1, protocol_sha256=digest,
                    policy=POLICY, source_scenes=list(scenes))
    destination.mkdir(parents=True)
    write_json(destination/"inventory.json", {**metadata, "frames": inventory,
        "scenes": [{k:v for k,v in s.items() if k != "frames"} for s in collected]})
    write_json(destination/"real_16.json", {**metadata, "clip_length":16,
        "clips":real, "statistics":{"real_clips":len(real), "pseudo_clips":0}})
    write_json(destination/"adaptation_sources.json", {**metadata, "scenes":adaptation})
    for setting, rows in tests.items():
        write_json(destination/f"test_{setting}.json", {**metadata, "clips":rows})
    report = {**metadata, "status":"cpu_protocol_validated",
        "num_scenes":len(collected), "num_frames":len(inventory),
        "real_clips":len(real), "adapt_frames":sum(s["split_index"] for s in collected),
        "heldout_frames":sum(s["num_frames"]-s["split_index"] for s in collected),
        "test_clips":{k:len(v) for k,v in tests.items()},
        "test_target_occurrences":{k:13*len(v) for k,v in tests.items()},
        "all_pixels_decoded":verify_pixels, "pseudo_manifest_ready":False,
        "gpu_validated":False}
    write_json(destination/"validation.json", report)
    print(f"Protocol {digest}: {len(real)} real clips; "
          f"tests={report['test_clips']}", flush=True)
    return report


def validate_saved(path, *, verify_files=False):
    path = checked(path)
    inventory_doc = load_json(path/"inventory.json")
    real = load_json(path/"real_16.json")
    test_docs = {k:load_json(path/f"test_{k}.json") for k in ("easy","hard")}
    digest = inventory_doc["protocol_sha256"]
    if inventory_doc["policy"] != POLICY:
        raise ValueError("Unsupported protocol policy")
    for doc in [real, *test_docs.values(), load_json(path/"adaptation_sources.json")]:
        if doc["protocol_sha256"] != digest or doc["policy"] != POLICY:
            raise ValueError("Mixed protocol versions")
    frames = inventory_doc["frames"]
    scenes = [{**s, "frames":[f for f in frames.values() if f["scene_name"]==s["scene_name"]]}
              for s in inventory_doc["scenes"]]
    if canonical_hash({"policy":POLICY,"scenes":scenes}) != digest:
        raise ValueError("Protocol inventory integrity failure")
    _, expected_real, expected_tests, expected_adaptation = build_manifests(scenes)
    if real["clips"] != expected_real:
        raise ValueError("Real manifest is incomplete, duplicated, or changed")
    for setting, expected in expected_tests.items():
        if test_docs[setting]["clips"] != expected:
            raise ValueError(f"Test {setting} manifest changed or incomplete")
    if load_json(path/"adaptation_sources.json")["scenes"] != expected_adaptation:
        raise ValueError("Adaptation source manifest changed or incomplete")
    validate_protocol(frames, real["clips"], {k:v["clips"] for k,v in test_docs.items()})
    if verify_files:
        for f in frames.values():
            if sha256_file(f["path"]) != f["sha256"]:
                raise ValueError(f"Source image changed: {f['frame_id']}")
    return dict(protocol_sha256=digest, real_clips=len(real["clips"]),
                test_clips={k:len(v["clips"]) for k,v in test_docs.items()})
