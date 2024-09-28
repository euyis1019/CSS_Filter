import sys
import os
import tqdm
import torch
import numpy as np
from ram.transform import get_transform_no_Normalize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CSS_Filter.css_dataset import load_dataset_from_config, Dataloader, get_dataset_config
from ram import get_transform

# 加载数据集配置
config = get_dataset_config("VOC")
config.increment_setting.save_stage_image_path = "default"
dataset, eval_dataset = load_dataset_from_config(config, 1, None)

stage_lengths = max(dataset.stage_index_dict.keys())

print("Stage lengths:", stage_lengths)
print(dataset.stage_index_dict[0])  # Stage 0 label
print(dataset.dataset.classes.items())  # dict_items([...])

# 定义变换（不包含归一化）
transform = get_transform_no_Normalize(image_size=384)  # 确保不进行归一化
dataset.dataset.transform = transform

# 设置 Dataloader
batch_size = 8
dataloader = Dataloader(dataset, batch_size=batch_size)

# 初始化统计变量
n_channels = 3
mean = torch.zeros(n_channels)
std = torch.zeros(n_channels)
num_samples = 0

# 遍历整个数据集
pbar = tqdm.tqdm(total=len(dataloader), desc="Computing Mean and Std")
for batch in dataloader:
    images = batch["image"]  # 假设图像的形状为 [batch_size, 3, H, W]
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)  # [batch_size, 3, H*W]
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    num_samples += batch_samples
    pbar.update(1)
pbar.close()

mean /= num_samples
std /= num_samples

print(f"Mean: {mean.tolist()}")
print(f"Std: {std.tolist()}")
