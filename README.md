# PoseMatch-TDCM

This repository provides the official implementation of our paper:

**"An Efficient Deep Template Matching and In-Plane Pose Estimation Method via Template-Aware Dynamic Convolution"**  
*Published in ...*

## 🔍 Highlights

• End-to-end estimation of 2D geometric pose for planar template matching.
• TDCM enables strong generalization to unseen targets with efficient
matching.
• Consistently high precision and speed under complex geometric transformations.
• A refinement module improves angle-scale estimation via local geometric fitting.
• Structure-aware pseudo labels enable self-supervised training without
annotations.

## 🧪 Benchmark
🚀 Matching Performance
All results are based on a standard template size of **36×36**, unless otherwise specified.

| Setting   | Description                          | Precision (mIoU ↑) | Inference Time ↓ |
| --------- | ------------------------------------ | ------------------ | ---------------- |
| **S1**    | Rotation only                        | **0.965**          | **101 ms**       |
| **S1.5**  | Rotation + mild scaling (0.8–1.5×)   | **0.936**          | **115 ms**       |
| **S2**    | Rotation + moderate scaling (0.5–2×) | **0.916**          | **119 ms**       |
| **S2.5**  | Rotation + large scaling (0.4–2.5×)  | **0.902**          | **123 ms**       |


### 📦 Model Download
Choose your preferred source:  

| Host         | 🔗 Link                                                                  | 🔑 Access Code       | 📁 Recommended Path |
|--------------|---------------------------------------------------------------------------|----------------------|---------------------|
| **Google Drive** | [Download](https://drive.google.com/drive/folders/14hvIaluqEBXuT3vS9cBwEydYo3d4JO6y?usp=drive_link) | - | `./dict/` |
| **Baidu Netdisk** | [Download](https://pan.baidu.com/s/1CHkGL0jkFk68T8mf3Sr34A?pwd=tdcm) | `tdcm` | `./dict/` |


## 📄 Citation

If you find this work useful, please cite:

```bibtex
@unpublished{zhou2025tdcm,
  title={An Efficient Deep Template Matching and In-Plane Pose Estimation Method via Template-Aware Dynamic Convolution},
  author={Zhou, Ji},
  journal={Preprint},
  year={2025}
}
