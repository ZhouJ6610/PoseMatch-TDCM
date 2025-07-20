import csv
from pycocotools.coco import COCO
import math

def filter_data(coco_annotations_path, output_csv_path, tShape=(36, 36), scale_range=(0.5, 2)):
    print(f"筛选完成，共找到 {count} 个满足条件的目标。")



if __name__ == "__main__":
    # COCO 数据集注释文件路径
    annotations_path = "/path-to/MS-CoCo/annotations/instances_train2017.json"
    
    # 输出 CSV 文件路径
    output_path = './data/train.csv'
    
    filter_data(
        annotations_path, 
        output_path, 
        tShape=(36, 36), 
        scale_range=(0.5, 2)
    )
