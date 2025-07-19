import torchvision.transforms as transforms
import cv2, json, torch, random, csv, os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from .get_imgR import rotate_crop
import pandas as pd


# 反向归一化
def deNormalize(img):
    if isinstance(img, np.ndarray):
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)  # 形状 (3,1,1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)   # 形状 (3,1,1)
        img = img.transpose(2, 0, 1)  # 转换为 (c, h, w)
        img = img * std + mean
        img = img.transpose(1, 2, 0)  # 转换回 (h, w, c)
        
    elif isinstance(img, torch.Tensor):
        device = img.device
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        img = img * std + mean
        
    return img
    

class CoCo_Dataset(Dataset):
    def __init__(self, imgPath, fileName, tShape):
        self.transform_img = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  
                std=[0.229, 0.224, 0.225]
            )            
        ])
        self.transform_roi = transforms.Compose([
            transforms.Resize(tShape),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  
                std=[0.229, 0.224, 0.225]
            )            
        ])
        self.imgPath   = imgPath
        self.samples   = []
        self.tShape    = tShape
        
        with open(fileName, 'r') as lines:
            for line in lines:
                name, x, y, w, h = line.strip().split(',')
                self.samples.append([
                    name,
                    (int(x), int(y), int(w), int(h))
                ])
        
        if 'train' in fileName:
            self.Directory = 'train2017/'
        elif 'val' in fileName:
            self.Directory = 'val2017/'
        else:
            print("解析出错啦！！！")
            exit(1)
    
    def __len__(self):
        return len(self.samples)
    
    # 数据操作
    def __getitem__(self, idx):
        while(1):
            name, bbox  = self.samples[idx]
            image       = cv2.imread(self.imgPath + self.Directory + name)
            image       = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            x, y, w, h =  [round(v) for v in bbox]
            template   =  image[y:y+h, x:x+w]
             
            # rotate_crop Image
            imgR, corners, angle = rotate_crop(image, bbox)
            if imgR is None:
                idx = random.randint(0, len(self.samples)-1)
                continue

            scale_y    =  h / self.tShape[0]
            scale_x    =  w / self.tShape[1]
            center     = corners.mean(axis=0)  # 计算中心点

            imgR       = self.transform_img(Image.fromarray(imgR))
            template   = self.transform_roi(Image.fromarray(template))
            
            return imgR, template, center[1], center[0], scale_y, scale_x, angle


def affine_transform(image, angle, scale_x, scale_y):
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2
    angle_rad = np.deg2rad(-angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    M_rs = np.array([[cos_a * scale_x, -sin_a * scale_y],
                     [sin_a * scale_x,  cos_a * scale_y]])
    t = np.array([cx, cy]) - np.dot(M_rs, np.array([cx, cy]))
    M = np.hstack([M_rs, t.reshape(2, 1)])
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    transformed_corners = cv2.transform(np.array([corners]), M)[0]
    x_min, y_min = transformed_corners.min(axis=0)
    x_max, y_max = transformed_corners.max(axis=0)
    new_w = x_max - x_min
    new_h = y_max - y_min
    M[0, 2] -= x_min
    M[1, 2] -= y_min
    transformed = cv2.warpAffine(image, M, (int(np.ceil(new_w)), int(np.ceil(new_h))), flags=cv2.INTER_LINEAR)
    return transformed, new_w, new_h


class CoCo_Dataset_multi_target(Dataset):
    def __init__(self, csv_path, image_dir, image_size=(360, 360), isRotate = True):
        self.df = pd.read_csv(csv_path, header=None, names=["filename", "x", "y", "w", "h"])
        self.grouped = self.df.groupby("filename")
        self.image_names = sorted(self.grouped.groups.keys())
        self.image_dir = image_dir
        self.H, self.W = image_size
        self.isRoate = isRotate
        self.cell_h, self.cell_w = self.H // 4, self.W // 4
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  
                std=[0.229, 0.224, 0.225]
            )            
        ])

    def __len__(self):
        return len(self.image_names) - 1  # 每对数据使用前后图像组合

    def __getitem__(self, idx):
        while(1):
            nameT = self.image_names[idx]
            nameS = self.image_names[idx + 1]
            
            if nameT == nameS:
                idx += 1
                continue

            # 读取图像
            imageT = cv2.imread(os.path.join(self.image_dir, nameT))
            imageT = cv2.cvtColor(imageT, cv2.COLOR_BGR2RGB)
            imageS = cv2.imread(os.path.join(self.image_dir, nameS))
            imageS = cv2.cvtColor(imageS, cv2.COLOR_BGR2RGB)
            imageS = cv2.resize(imageS, (self.W, self.H))

            # 提取模板
            x, y, w, h = self.grouped.get_group(nameT)[["x", "y", "w", "h"]].values[0]
            template = imageT[y:y+h, x:x+w]

            pasted_image = imageS.copy()
            label = {
                "nameS": nameS,
                "nameT": nameT,
                "num": 0,
                "templates": []
            }

            p = random.random()
            for i in range(4):
                for j in range(4):
                    if random.random() > p:
                        continue

                    tx = j * self.cell_w
                    ty = i * self.cell_h
                    offset_x = tx + random.uniform(0.3 * self.cell_w, 0.7 * self.cell_w)
                    offset_y = ty + random.uniform(0.3 * self.cell_h, 0.7 * self.cell_h)
                    
                    angle = random.uniform(-180, 180) if self.isRoate else 0
                        
                    transformed, new_w, new_h = affine_transform(template, angle, 1.0, 1.0)
                    mask = np.ones_like(template, dtype=np.uint8)
                    mask, _, _ = affine_transform(mask, angle, 1.0, 1.0)

                    T = np.float32([[1, 0, offset_x - new_w / 2], [0, 1, offset_y - new_h / 2]])
                    transformed = cv2.warpAffine(transformed, T, (self.W, self.H))
                    mask = cv2.warpAffine(mask, T, (self.W, self.H))

                    pasted_image = (pasted_image * (1 - mask)) + transformed

                    label["templates"].append({
                        "x": float(offset_x),
                        "y": float(offset_y),
                        "scale_x": w / 36,
                        "scale_y": h / 36,
                        "rotation": float(angle),
                    })
                    label["num"] += 1

            pasted_image = self.transform(Image.fromarray(pasted_image))
            template = cv2.resize(template, (36, 36))
            template = self.transform(Image.fromarray(template))

            return pasted_image, template, label
