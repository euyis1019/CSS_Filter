import argparse
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torchvision import transforms
from ram.models import ram_plus_gyf
from ram import get_transform
from ram.voc_util.voc_utils import VOCClassification, custom_collate_fn
from tqdm import tqdm
root = '/root/autodl-tmp/when-SAM-meets-Continual-Learning/data/VOC2012'
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
model.load_state_dict(torch.load('/root/autodl-tmp/test_voc_mask_4.pth'))  # 'test_voc_mask_best.pth' 是你保存的最佳模型文件
model.eval()  # 切换到评估模式

from css_dataset.create_dataset import load_dataset_from_config
from utils import get_dataset_config
from dataset.dataloader import Dataloader

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
criterion = torch.nn.BCEWithLogitsLoss()
# 存储性能指标
all_preds = []
all_labels = []

with torch.no_grad():  # 禁用梯度计算
    for img, label, text_prompt, label_prompt in dataloader:
        img = img.to(device)  # 将输入移到 GPU（如果可用）
        label_formatted = [[item for item in label_str.split('.') if item] for label_str in label_prompt]
        
        # 前向传播
        logits, attention_mask = model.forward_voc_mask(img, label_formatted)
        print("logits:",logits)
        print("attention_mask:",attention_mask)
        # 计算损失（可选）
        attention_mask = attention_mask.float()
        for i, mask in enumerate(attention_mask):
            present_classes = [classes[j] for j in range(len(mask)) if mask[j] == 1]
            print(f"样本 {i + 1} 的类别: {', '.join(present_classes)}")
            print(f"样本 {i + 1} 的标签: {label_formatted[i]}")
            print(f"样本 {i + 1} 的text_prompt: {text_prompt[i]}")#text_prompt

        loss = criterion(logits, attention_mask)
        print(f"验证损失: {loss.item():.4f}")
        
        # 将预测转换为二进制输出
        preds = torch.sigmoid(logits).cpu().numpy() > 0.2 # 这里preds变成了和attention mask一样维度的one hot向量：
        for i, pred in enumerate(preds):
            present_classes = [classes[j] for j in range(len(pred)) if pred[j] == 1]
            print(f"样本 {i + 1} 的预测类别: {', '.join(present_classes)}")

        # 将标签转换为 numpy 格式
        labels = attention_mask.cpu().numpy()
        all_labels.extend(labels)

# # 计算准确率、精确率、召回率和 F1 分数
# accuracy = accuracy_score(all_labels, all_preds)
# precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro')

# print(f"验证准确率: {accuracy:.4f}")
# print(f"验证精确率: {precision:.4f}")
# print(f"验证召回率: {recall:.4f}")
# print(f"验证 F1 分数: {f1:.4f}")


