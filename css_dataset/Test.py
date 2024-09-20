import argparse
import numpy as np
import os
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torchvision import transforms
from ram.models import ram_plus_gyf
from ram import get_transform

# 命令行参数
parser = argparse.ArgumentParser(description='Tag2Text Training')
parser.add_argument('--pretrained', metavar='DIR', help='path to pretrained model', default='/root/autodl-tmp/recognize-anything/pretrained/ram_plus_swin_large_14m.pth')
parser.add_argument('--image-size', default=384, type=int, metavar='N', help='input image size (default: 448)')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes =['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv monitor']

# 转换
transform = get_transform(image_size=args.image_size)
 

# 加载模型
model = ram_plus_gyf(pretrained=args.pretrained, image_size=args.image_size, vit='swin_l',classes=classes)
#self.fc_voc = torch.nn.Linear(4586, 20)  # 
model.to(device)

from css_dataset.create_dataset import load_dataset_from_config
from utils import get_dataset_config
from dataset.dataloader import Dataloader
for name, param in model.named_parameters():
    if name not in ['spatial_fc.weight', 'spatial_fc.bias', 'cls_fc.weight', 'cls_fc.bias']:
        param.requires_grad = False
config = get_dataset_config("VOC")
classes_dict = config.dataset_setting.classes
print(config)
user_input = 'yes'
if user_input.lower() == 'yes':
    config.increment_setting.save_stage_image_path = "default"
else:
    print("Configuration not found. Reconfig")
dataset = load_dataset_from_config(config,1,None)
dataset.dataset.transform = transform
dataloader = Dataloader(dataset,batch_size=8)
i=0
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = Adam([
    {'params': model.spatial_fc.parameters()},
    {'params': model.cls_fc.parameters()}
], lr=0.001)
best_loss = float('inf')
epoch = 0
for epoch in range(10): 
    for img,label,text_prompt,label_prompt in dataloader:
        i+=1
        label_formatted = [[item for item in label_str.split('.') if item] for label_str in label_prompt]  # 去掉空项
        logits,attention_mask = model.forward_voc_mask(img,label_formatted)
        attention_mask = attention_mask.float()
        for i, mask in enumerate(attention_mask):
            present_classes = [classes[j] for j in range(len(mask)) if mask[j] == 1]
            print(f"样本 {i + 1} 的类别: {', '.join(present_classes)}")
            print(f"样本 {i + 1} 的标签: {label_formatted[i]}")
            print(f"样本 {i + 1} 的text_prompt: {text_prompt[i]}")#text_prompt
        loss = criterion(logits,attention_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"当前损失: {loss.item():.4f}")
        preds = torch.sigmoid(logits).detach().cpu().numpy() > 0.1
        for i, pred in enumerate(preds):
            present_classes = [classes[j] for j in range(len(pred)) if pred[j] == 1]
            print(f"样本 {i + 1} 的预测类别: {', '.join(present_classes)}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), f'test_voc_mask_{epoch}.pth')
            print(f"保存最佳模型，当前epoch: {epoch}, 最佳损失: {best_loss:.4f}")
        
        if i >= 50:
            print("更新阶段")
            i=0
            dataset.update_stage(1)
            epoch += 1
            print(f"当前epoch: {epoch}")




