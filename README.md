# PoseMatch-TDCM

This repository provides the official implementation of our paper:

**"An Efficient Deep Template Matching and In-Plane Pose Estimation Method via Template-Aware Dynamic Convolution"**  
*Published in [Your Target Journal/Conference]*

## 🔍 Highlights

- **One-shot template matching** without requiring per-template retraining
- **Direct pose estimation**: center, rotation angle, and scale
- **Template-Aware Dynamic Convolution (TDCM)** for efficient structure-aware fusion
- **Self-supervised training** using affine transformations + pseudo labels
- **Supports multi-object and small template matching**
- **Up to 55× faster** than traditional industrial software (Halcon SHM)

## 📦 Features

- End-to-end network for template matching and pose regression
- Lightweight architecture using ConvNeXt-V2 Stage-1, depthwise separable convolutions, and pixel shuffle
- Optional geometric refinement module to improve rotation/scale accuracy
- No annotation needed – fully self-supervised training pipeline provided
- Inference-ready on real-world and synthetic datasets

## 🏗️ Architecture Overview

- **Backbone**: ConvNeXt-V2-large stage-1
- **TDCM Module**: Injects template as depthwise convolution kernels
- **Decoder**: Predicts center heatmap and geometric parameters
- **Refine Module**: Locally optimizes angle-scale estimates

## 🧪 Benchmark

| Setting | Precision (mIoU) | Speed |
|--------|------------------|-----------------------|
| S2 (0.5–2 scale + rotation) | 0.916 | 119ms |

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@unpublished{zhou2025tdcm,
  title={An Efficient Deep Template Matching and In-Plane Pose Estimation Method via Template-Aware Dynamic Convolution},
  author={Zhou, Ji},
  journal={Preprint},
  year={2025}
}
