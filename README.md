# PoseMatch-TDCM

This repository provides the official implementation of our paper:

**"An Efficient Deep Template Matching and In-Plane Pose Estimation Method via Template-Aware Dynamic Convolution"**  
*Published in ...*

## 🔧 Framework Architecture

![Framework](framework.png)



## 🔍 Highlights

• End-to-end estimation of 2D geometric pose for planar template matching.  
• TDCM enables strong generalization to unseen targets with efficient matching.  
• Consistently high precision and speed under complex geometric transformations.  
• A refinement module improves angle-scale estimation via local geometric fitting.  
• Structure-aware pseudo labels enable self-supervised training without annotations.  

## 🧪 Benchmark
🚀 Matching Performance
All results are based on a standard template size of **36×36**, unless otherwise specified.

| Setting   | Description                          | Precision (mIoU ↑) | Time(CPU E5-2680 v4) ↓ |
| --------- | ------------------------------------ | :----------------: | :-----------------:    |
| **S1**    | Rotation only                        |     **0.965**      |     **58.6 ms**        |
| **S1.5**  | Rotation + mild scaling (0.8–1.5×)   |     **0.936**      |     **67.2 ms**        |
| **S2**    | Rotation + moderate scaling (0.5–2×) |     **0.916**      |     **68.7 ms**        |
| **S2.5**  | Rotation + large scaling (0.4–2.5×)  |     **0.902**      |     **72.1 ms**        |


## 🖥️ Usage

`test.py` contains example usage of the PoseMatch-TDCM matcher.  
To try with your own images, edit the file and set:

```python
query_image_path = './res/image.jpg'
template_image_path = './res/template.jpg'
```
   

## 📦 Checkpoints

Currently, only the model trained for the **S2** setting is provided under the `dict/` directory.
For other settings (e.g., S1, S1.5, S2.5, T12, T28-52, T124), please download them manually from the links below:

| Host              | 🔗 Link                                                                                          | 🔑 Access Code |
| ----------------- | ------------------------------------------------------------------------------------------------ | -------------- |
| **Google Drive**  | [Download](https://drive.google.com/drive/folders/14hvIaluqEBXuT3vS9cBwEydYo3d4JO6y?usp=sharing) | –              |
| **Baidu Netdisk** | [Download](https://pan.baidu.com/s/108eWI0Eo-Q4_gNmO88HouQ?pwd=tdcm)                             | `tdcm`         |

