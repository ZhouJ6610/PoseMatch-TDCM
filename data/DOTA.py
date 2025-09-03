import cv2
import numpy as np
import os
import glob
import math
from PIL import Image

def draw_polygon(image, points, color, thickness=2):
    """在图像上绘制多边形"""
    int_points = np.array([[int(p[0]), int(p[1])] for p in points], dtype=np.int32)
    cv2.polylines(image, [int_points], isClosed=True, color=color, thickness=thickness)
    return image

def calculate_min_area_rect_angle(corners):
    """使用最小外接矩形计算旋转角度"""
    corners_array = np.array(corners, dtype=np.float32)
    rect = cv2.minAreaRect(corners_array)
    return rect[2]  # 返回旋转角度

def calculate_rotation_angle(corners):
    """根据模板坐标计算旋转角度 - 改进版"""
    if len(corners) < 4:
        return 0
    
    # 方法1: 使用最小外接矩形计算角度
    min_area_angle = calculate_min_area_rect_angle(corners)
    
    # 方法2: 计算两条长边的角度并取平均
    edges = []
    for i in range(4):
        dx = corners[(i+1)%4][0] - corners[i][0]
        dy = corners[(i+1)%4][1] - corners[i][1]
        length = math.sqrt(dx**2 + dy**2)
        angle = math.degrees(math.atan2(dy, dx))
        edges.append((length, angle))
    
    # 按长度排序，取最长的两条边
    edges.sort(key=lambda x: x[0], reverse=True)
    long_edge_angles = [edges[0][1], edges[1][1]]
    
    # 计算平均角度
    avg_angle = np.mean(long_edge_angles)
    
    # 选择最稳定的角度计算方法
    # 使用最小外接矩形的角度，但调整到-90——90度范围
    if min_area_angle < -45:
        min_area_angle += 90
    elif min_area_angle > 45:
        min_area_angle -= 90
    
    return min_area_angle

def rotate_point(point, center, angle_deg):
    """旋转点坐标（围绕中心点，与OpenCV旋转方向一致）"""
    angle_rad = np.deg2rad(angle_deg)
    x, y = point
    cx, cy = center
    
    # 平移点到原点
    x -= cx
    y -= cy
    
    # 旋转
    x_rot = x * np.cos(angle_rad) + y * np.sin(angle_rad)
    y_rot = -x * np.sin(angle_rad) + y * np.cos(angle_rad)
    
    # 平移回原位置
    x_rot += cx
    y_rot += cy
    
    return (x_rot, y_rot)

def process_template_data(image, template_corners):
    """处理模板数据：旋转图像并裁剪模板 - 改进版"""
    # 计算旋转角度
    rotation_angle = calculate_rotation_angle(template_corners)
    
    # 计算旋转中心（使用最小外接矩形的中心）
    corners_array = np.array(template_corners, dtype=np.float32)
    rect = cv2.minAreaRect(corners_array)
    center = rect[0]  # 使用最小外接矩形的中心
    
    # 获取旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    
    # 旋转整个图像
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    
    # 计算旋转后模板的四角坐标
    rotated_corners = [rotate_point(corner, center, rotation_angle) for corner in template_corners]
    
    # 找到旋转后模板的边界框（使用最小外接矩形）
    rotated_corners_array = np.array(rotated_corners, dtype=np.float32)
    rect_rotated = cv2.minAreaRect(rotated_corners_array)
    box_rotated = cv2.boxPoints(rect_rotated)
    box_rotated = np.intp(box_rotated)
    
    # 提取边界框的坐标
    x_coords = box_rotated[:, 0]
    y_coords = box_rotated[:, 1]
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    # 确保边界框在图像范围内
    x_min = max(0, x_min)
    x_max = min(image.shape[1], x_max)
    y_min = max(0, y_min)
    y_max = min(image.shape[0], y_max)
    
    # 计算裁剪后模板的宽度和高度
    template_width = x_max - x_min
    template_height = y_max - y_min
    
    # 确保边界框有效
    if template_width <= 0 or template_height <= 0:
        print(f"警告: 无效的边界框 ({x_min}, {y_min}, {x_max}, {y_max})")
        return None, None, None, None, None, None
    
    # 裁剪模板
    cropped_template = rotated_image[y_min:y_max, x_min:x_max]
    
    # 计算裁剪区域的四角坐标（在旋转后图像中的坐标）
    cropped_corners_rotated = [
        (x_min, y_min),  # 左上
        (x_max, y_min),  # 右上
        (x_max, y_max),  # 右下
        (x_min, y_max)   # 左下
    ]
    
    # 反向旋转裁剪区域的四角坐标，得到在原图中的坐标
    inverse_rotation_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
    
    # 将裁剪区域的四角坐标转换回原图坐标
    original_corners = []
    for corner in cropped_corners_rotated:
        point = np.array([corner[0], corner[1], 1])
        original_point = np.dot(inverse_rotation_matrix, point)
        original_corners.append(tuple(original_point))
    
    # 使用还原后的矩形角点计算中心点
    restored_center_x = np.mean([p[0] for p in original_corners])
    restored_center_y = np.mean([p[1] for p in original_corners])
    restored_center = (restored_center_x, restored_center_y)
    
    return cropped_template, rotation_angle, restored_center, template_width, template_height, original_corners

def process_images_batch(image_dir, label_dir, template_output_dir, output_dir=None):
    """批量处理图像和标签"""
    # 确保输出目录存在
    os.makedirs(template_output_dir, exist_ok=True)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 创建gt.txt文件
    gt_file_path = os.path.join(template_output_dir, "gt.txt")
    gt_file = open(gt_file_path, "w", encoding="utf-8")
    
    # 获取所有图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 处理每个图像
    for image_path in image_files:
        print(f"处理图像: {image_path}")
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            continue
            
        img_height, img_width = image.shape[:2]
        print(f"图像尺寸: {img_width}x{img_height}")
        
        # 构建对应的标签文件路径
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(label_dir, base_name + ".txt")
        
        # 检查标签文件是否存在
        if not os.path.exists(label_path):
            print(f"标签文件不存在: {label_path}")
            continue
        
        # 复制图像用于绘制
        image_with_boxes = image.copy()
        
        # 读取标签文件
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        valid_lines = lines[2:] # 跳过前两行
        print(f"找到 {len(valid_lines)} 个模板")
        
        # 模板计数器
        img_template_count = 0
        
        # 处理每个模板
        for line_idx, line in enumerate(valid_lines):
            parts = line.strip().split()
            if len(parts) < 8:  # 只需要8个坐标值
                print(f"无效的标签行: {line}")
                continue
            
            # 提取坐标
            coords = list(map(float, parts[:8]))
            template_corners = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
            
            # 处理模板数据
            cropped_template, rotation_angle, center, template_width, template_height, restored_corners = process_template_data(image, template_corners)
            
            if cropped_template is None:
                print(f"模板 {line_idx+1} 处理失败，跳过")
                continue
            
            # 检查模板尺寸是否符合要求 (18-72像素)
            if template_width < 18 or template_width > 72 or template_height < 18 or template_height > 72:
                continue
            
            # 在图像上绘制原始模板框（红色）
            image_with_boxes = draw_polygon(image_with_boxes, template_corners, (0, 0, 255), 2)
            
            # 在图像上绘制还原后的矩形框（绿色）
            image_with_boxes = draw_polygon(image_with_boxes, restored_corners, (0, 255, 0), 2)
            
            # 创建模板文件名
            img_template_count += 1
            template_filename = f"{base_name}_template_{img_template_count:04d}.png"
            template_path = os.path.join(template_output_dir, template_filename)
            
            # 保存模板图像
            if len(cropped_template.shape) == 3 and cropped_template.shape[2] == 3:
                template_rgb = cv2.cvtColor(cropped_template, cv2.COLOR_BGR2RGB)
            else:
                template_rgb = cropped_template
            
            template_pil = Image.fromarray(template_rgb)
            template_pil.save(template_path)
            
            # 写入gt.txt文件
            image_name = os.path.basename(image_path)
            center_x, center_y = center
            
            gt_line = f"{image_name} {template_filename} {center_x:.2f} {center_y:.2f} {template_width/36:.3f} {template_height/36:.3f} {rotation_angle:.2f}\n"
            gt_file.write(gt_line)
        
        # 保存绘制后的图像
        if output_dir and img_template_count!=0 :
            output_path = os.path.join(output_dir, os.path.basename(image_path))
            
            # 将BGR转换为RGB
            if len(image_with_boxes.shape) == 3 and image_with_boxes.shape[2] == 3:
                image_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image_with_boxes
            
            output_pil = Image.fromarray(image_rgb)
            output_pil.save(output_path)
            print(f"已保存绘制后的图像: {output_path}")
        
        print("-" * 50)
    
    # 关闭gt.txt文件
    gt_file.close()
    print(f"GT文件已保存: {gt_file_path}")

# 使用示例
if __name__ == "__main__":
    # 设置路径和参数
    image_dir = "/workspace/zhouji/dataSets/DOTA/images/images"  # 替换为你的图像目录
    label_dir = "/workspace/zhouji/dataSets/DOTA/labelTxt" # 替换为你的标签目录
    template_output_dir = "/workspace/zhouji/dataSets/DOTA/templates" # 裁剪模板保存目录
    output_dir = "output_images_with_boxes" # 绘制后的图像保存目录（可选）
    
    # 批量处理
    process_images_batch(image_dir, label_dir, template_output_dir)
