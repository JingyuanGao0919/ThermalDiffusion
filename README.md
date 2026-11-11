# ThermalDiffusion: Confidence-Aware Diffusion for Sparse-View Thermal Infrared Novel View Synthesis

This repository accompanies **ThermalDiffusion**, a method that combines thermal-domain adaptation of a pretrained latent video diffusion model with confidence-aware sampling guidance derived from a coarse 3D Gaussian Splatting (3DGS) proxy.

## Pseudo-Thermal Dataset Constructed from TI-NSD

The pseudo-thermal dataset constructed from TI-NSD is available on [Google Drive](https://drive.google.com/drive/folders/1Es67NnCURHJXnF6idNvglb-XB36NnHED?usp=sharing).

### Acknowledgments

We thank Qian Chen, Shihao Shu, and Xiangzhi Bai for introducing and releasing **TI-NSD** in their ECCV 2024 paper, [*Thermal3D-GS*](https://github.com/mzzcdf/Thermal3DGS). Our pseudo-thermal dataset builds upon TI-NSD. If you use it, please also cite the original Thermal3D-GS paper.

## Repository Contents

The repository currently contains the following source files:

```text
src/
├── __init__.py
├── benchmark_evaluation.py
├── colmap.py
├── conditioning.py
├── ddim.py
├── distillation.py
├── distillation_training.py
├── evaluation.py
├── guided_inference.py
├── initialization.py
├── initialization_cache.py
├── io.py
├── lora.py
├── official_backend.py
├── physics.py
├── protocol.py
├── raster.py
├── source_fields.py
└── sparse_proxy.py
```

### Main Components

| Component | Corresponding Modules | Description |
|---|---|---|
| Data Protocol and Camera I/O | `protocol.py`, `colmap.py`, `io.py` | Dataset splitting, camera data handling, validation, and file integrity checks. |
| Initialization and Coarse Proxy | `initialization.py`, `initialization_cache.py`, `sparse_proxy.py`, `raster.py` | Geometric initialization and rendering of coarse thermal images, depth maps, and confidence maps. |
| Guidance Mechanisms | `physics.py`, `source_fields.py`, `conditioning.py`, `ddim.py` | Reliability estimation, source-field optimization, radiometric evidence anchoring (REA), depth-aware screened-Poisson guidance (DSPG), and guided DDIM sampling. |
| Adaptation and Inference Support | `lora.py`, `official_backend.py`, `guided_inference.py` | LoRA utilities, pretrained backbone loading, and guided inference. |
| Distillation and Evaluation | `distillation.py`, `distillation_training.py`, `evaluation.py`, `benchmark_evaluation.py` | Refined 3DGS optimization and quantitative evaluation. |

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{zhang2026thermaldiffusion,
  title={ThermalDiffusion: Confidence-Aware Diffusion for Sparse-View Thermal Infrared Novel View Synthesis},
  author={Zhang, Mingjin and Lu, FengQin and Gao, Jingyuan and Gao, Fei and Zhang, Qiming},
  booktitle={Proceedings of the 34th ACM International Conference on Multimedia},
  year={2026},
  doi={10.1145/3767308.3835148}
}
```

### Original TI-NSD Dataset

```bibtex
@inproceedings{chen2024thermal3dgs,
  title={Thermal3D-GS: Physics-Induced 3D Gaussians for Thermal Infrared Novel-View Synthesis},
  author={Chen, Qian and Shu, Shihao and Bai, Xiangzhi},
  booktitle={European Conference on Computer Vision},
  year={2024},
  doi={10.1007/978-3-031-73383-3_15}
}
```
