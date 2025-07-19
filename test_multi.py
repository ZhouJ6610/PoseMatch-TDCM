import warnings
warnings.filterwarnings("ignore")

import cv2, torch, time
import numpy as np
from utils import CoCo_Dataset_multi_target
from model import Model
from utils import refine_angle_bisection, refine_scale_y, refine_scale_x, refine_angle
from utils import getIOU

import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

def get_centers_from_scoremap(score_map, threshold=0.2, min_area=3):
    binary = (score_map > threshold).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    centers = []
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask = (labels == i)
            weights = score_map[mask]
            y_coords, x_coords = np.where(mask)
            cx = np.average(x_coords, weights=weights)
            cy = np.average(y_coords, weights=weights)
            centers.append((cx, cy))
    return centers


def postprocess_output(outputs, threshold, isScale=True):
    pred_score = outputs[0][0, 0, :, :]
    pred_sign  = outputs[1][0, 0, :, :]
    pred_cos   = outputs[2][0, 0, :, :]
    if not isScale:
        pred_sx    = np.ones_like(pred_score)
        pred_sy    = np.ones_like(pred_score)
    else:
        pred_sx    = outputs[3][0, 0, :, :]
        pred_sy    = outputs[4][0, 0, :, :]
    centers = get_centers_from_scoremap(pred_score, threshold)

    results = []
    for CX, CY in centers:
        cx = int(CX); cy = int(CY)
        angle_cos = pred_cos[cy, cx]
        angle_deg = np.arccos(angle_cos) * 180 / np.pi
        if pred_sign[cy, cx] < 0.5:
            angle_deg = -angle_deg
        scale_x = pred_sx[cy, cx]
        scale_y = pred_sy[cy, cx]
        score = pred_score[cy, cx]

        results.append({
            'x': CX,
            'y': CY,
            'angle': angle_deg,
            'scale_x': max(min(scale_x, 1.5), 0.8),
            # 'scale_x': scale_x,
            'scale_y': max(min(scale_y, 1.5), 0.8),
            # 'scale_y': scale_y,
            'score': score
        })
    return results


th, tw = 36, 36
img_dir = '/workspace/zhouji/dataSets/MS-CoCo/val2017/'
csv_file = 'data/S1.5/val.csv'
device = torch.device('cpu')
model = Model(None, device).to(device)
state_dict = torch.load('dict/model.pth', map_location=device)
model.load_state_dict(state_dict)
model.eval()

dataset = CoCo_Dataset_multi_target(csv_file, img_dir, (360, 360))

result_file = 'results-multi.txt'
with open(result_file, 'w', encoding='utf-8') as rf:
    rf.write("===== 多目标匹配结果 =====\n\n")

    matched_total = 0
    pred_total = 0
    gt_total = 0
    missed_total = 0
    false_positive_total = 0

    sample_index = 0
    max_samples = 200
    
    # 所有样本统计值
    sum_IoU = 0.0
    sum_time = 0.0
    with torch.no_grad():
        for i in range(min(len(dataset), max_samples)):
            image, template, label = dataset[i]
            image = image.to(device)
            template = template.to(device)

            sample_index += 1
            if sample_index > max_samples:
                break
            print(f'Processing sample {sample_index}/{min(max_samples, len(dataset))}  ...')
            rf.write(f"样本 {sample_index},  {label['nameS']}\n")

            image, template = image.unsqueeze(0), template.unsqueeze(0)

            template_infos = label['templates']
            if len(template_infos) == 0:
                rf.write(f" 空目标 \n")
                rf.write("--------------------------------------------------\n\n")
                continue
            
            start_time = time.time()
            outputs = model(image, template)
            inference_time = time.time() - start_time
            
            outputs = [output.cpu().numpy() for output in outputs]
            predictions = postprocess_output(outputs, threshold=0.01)

            used_gt = set()
            used_pred = set()
            pred_total += len(predictions)
            gt_total += len(template_infos)
            
            post_time = 0.0
            for pi, pred in enumerate(predictions):
                px, py = pred['x'], pred['y']
                score = pred['score']
                min_dist = float('inf')
                best_idx = -1

                # 寻找最近的GT
                for gi, gt in enumerate(template_infos):
                    if gi in used_gt:
                        continue
                    gx, gy = gt['x'], gt['y']
                    dist = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = gi

                # 角度细化（无论是否匹配都进行，为了获得细化得分）
                start_time = time.time()
                refined_angle, ref_score = refine_angle_bisection(
                    cv2.cvtColor(image[0].cpu().permute(1, 2, 0).numpy(),cv2.COLOR_BGR2GRAY), # 转为HWC格式
                    cv2.cvtColor(template[0].cpu().permute(1, 2, 0).numpy(),cv2.COLOR_BGR2GRAY),  # 取当前模板
                    pred_x=px, pred_y=py,
                    pred_scale_x=pred['scale_x'],
                    pred_scale_y=pred['scale_y'],
                    pred_angle=pred['angle'],
                    initial_range=30,
                    coarse_threshold=2,
                    threshold=0.75
                )
                pred['scale_x'], _ = refine_scale_x(
                    cv2.cvtColor(image[0].cpu().permute(1, 2, 0).numpy(),cv2.COLOR_BGR2GRAY), # 转为HWC格式
                    cv2.cvtColor(template[0].cpu().permute(1, 2, 0).numpy(),cv2.COLOR_BGR2GRAY),  # 取当前模板
                    pred_x=px,
                    pred_y=py,
                    pred_scale_x=pred['scale_x'],
                    pred_scale_y=pred['scale_y'],
                    refined_angle=refined_angle,
                    threshold=0.8
                )
                pred['scale_y'], _ = refine_scale_y(
                    cv2.cvtColor(image[0].cpu().permute(1, 2, 0).numpy(),cv2.COLOR_BGR2GRAY), # 转为HWC格式
                    cv2.cvtColor(template[0].cpu().permute(1, 2, 0).numpy(),cv2.COLOR_BGR2GRAY),  # 取当前模板
                    pred_x=px,
                    pred_y=py,
                    refined_scale_x=pred['scale_x'],
                    pred_scale_y=pred['scale_y'],
                    refined_angle=refined_angle,
                    threshold=0.8
                )
                post_time += time.time() - start_time
                
                # 判断是否为正确匹配
                is_match = (best_idx >= 0 and min_dist < 20)

                if is_match:
                    gt = template_infos[best_idx]
                    
                    # 计算iou
                    true_P = (gt['x'], gt['y'], gt['scale_x'], gt['scale_y'], gt['rotation'])
                    pred_P = (px, py, pred['scale_x'], pred['scale_y'], refined_angle)
                    iou = getIOU(true_P, pred_P, th, tw)
                    
                    used_gt.add(best_idx)
                    used_pred.add(pi)
                    matched_total += 1
                    
                    rf.write(f"✅ 匹配: 预测 ({px:.1f}, {py:.1f}) vs GT ({gt['x']:.1f}, {gt['y']:.1f}),\tIoU = {iou:.3f}\n")
                    sum_IoU += iou
                else:
                    rf.write(f"❌ 未匹配: 预测 ({px:.1f}, {py:.1f}), "f"得分: {score:.3f}\n")
                    false_positive_total += 1

            missed = len(template_infos) - len(used_gt)
            missed_total += missed
            if missed > 0:
                for gi, gt in enumerate(template_infos):
                    if gi not in used_gt:
                        rf.write(f"⚠️ 漏检目标: GT位置 ({gt['x']:.1f}, {gt['y']:.1f}), rotation={gt['rotation']:.2f}°\n")

            rf.write(f"图像统计: GT={len(template_infos)}, 预测={len(predictions)}, 匹配={len(used_gt)}, 漏检={missed}\n")
            rf.write(f"time(ms): {inference_time * 1000:.2f} + {post_time * 1000:.2f} = {(inference_time+post_time)*1000:.2f}\n")
            
            rf.write("--------------------------------------------------\n\n")
            sum_time += (inference_time + post_time)

    rf.write("\n===== 总结统计 =====\n")
    rf.write(f"图像总数: {max_samples}\n")
    rf.write(f"总GT数: {gt_total}, 总预测数: {pred_total}\n")
    rf.write(f"正检: {matched_total}\n")
    rf.write(f"漏检: {missed_total}\n")
    rf.write(f"错检: {false_positive_total}\n")
    rf.write(f"Precision: {matched_total / pred_total:.3f}\n")
    rf.write(f"Recall: {matched_total / gt_total:.3f}\n")
    
    rf.write(f"平均IoU: {sum_IoU / (matched_total):.3f}\n")
    rf.write(f"平均时间: {sum_time / max_samples * 1000:.2f} ms\n")