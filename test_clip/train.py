import sys
import os
import tqdm
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
import loralib as lora

# Set up paths and imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CSS_Filter.css_dataset import load_dataset_from_config, Dataloader, get_dataset_config
from ram import get_transform,get_transform_no_Normalize,transfomr_384_voc_nor
def text_prompts_to_tensor(text_prompts, class_dict, num_classes, first_text):

    # Create a mapping from class names to indices (1 to 20)
    class_name_to_index = {v: k for k, v in class_dict.items() if 1 <= k <= 20}
    first_index = class_name_to_index[first_text]
    batch_size = len(text_prompts)

    # Initialize the tensor with zeros
    labels_tensor = torch.zeros((batch_size, num_classes), dtype=torch.float32)

    # Populate the tensor
    for i, labels in enumerate(text_prompts):
        for label in labels:
            if label in class_name_to_index:
                class_idx = class_name_to_index[label]
                # Subtract 1 because tensor indices start at 0
                labels_tensor[i][class_idx - first_index] = 1.0

    return labels_tensor

# 加载数据集配置
config = get_dataset_config("VOC")
config.increment_setting.save_stage_image_path = "default"
dataset, eval = load_dataset_from_config(config, 1, None)
dataset.dataset.transform = transfomr_384_voc_nor()
stage_lengths = max(dataset.stage_index_dict.keys())

print("Stage lengths:", stage_lengths)
print(dataset.stage_index_dict[0])  # Stage 0 label
print(dataset.dataset.classes.items())  # dict_items([...])



# Initialize the largest CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
cache_dir = os.getenv("TORCH_HOME", os.path.join(str(os.path.expanduser("~")), ".cache"))

# 获取模型文件路径
model_path = os.path.join(cache_dir, "transformers", model_name)

print(f"模型文件路径：{model_path}")
# 获取模型的参数数量
num_params = sum(p.numel() for p in clip_model.parameters())

# 获取模型的大小（以MB为单位）
model_size = num_params * 4 / (1024 * 1024)  # 4 bytes per float32

print(f"CLIP 模型大小：{model_size:.2f} MB")

# Freeze the text encoder
for param in clip_model.text_model.parameters():
    param.requires_grad = False

# # Apply LoRA to the image encoder's attention layers with rank=32
# for name, module in clip_model.vision_model.named_modules():
#     if isinstance(module, nn.MultiheadAttention) or 'SelfAttention' in name:
#         lora.inject_lora(module, r=32)

# Optimizer for the LoRA parameters
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, clip_model.parameters()), lr=1e-5)

# 或者使用学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Define loss function with label smoothing
class BCEWithLogitsLossWithLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super(BCEWithLogitsLossWithLabelSmoothing, self).__init__()
        self.smoothing = smoothing
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # Apply label smoothing
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        loss = self.bce_loss(logits, targets)
        return loss


smoothing = 0.1  # Adjust as needed
loss_fn = BCEWithLogitsLossWithLabelSmoothing(smoothing=smoothing)

batch_size = 8
num_stages = max(dataset.stage_index_dict.keys())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = clip_model.to(device)

loss_fn = loss_fn.to(device)
start = int(input("Please input the first stage: "))
# 确保模型处于训练模式
clip_model.train()

for current_stage in range(start, num_stages+1):
    print(f"Starting Stage {current_stage}")

    # 更新数据集和数据加载器
    dataset.update_stage(current_stage)
    dataloader = Dataloader(dataset, batch_size=batch_size)

    # 准备当前阶段的文本提示
    stage_text = dataset.class_name
    num_stage_labels = len(stage_text)

    print(num_stage_labels)
    print(len(stage_text))

    # 计算文本嵌入，并使用 detach()
    text_inputs = processor(text=stage_text, return_tensors="pt", padding=True)
    text_inputs = text_inputs.to(device)
    text_embeddings = clip_model.get_text_features(**text_inputs)
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    text_embeddings = text_embeddings.detach().to(device)  # 添加 .detach()
    class_dict = dataset.dataset.classes

    # 训练循环
    pbar = tqdm.tqdm(total=len(dataloader))
    for batch in dataloader:
        images, labels, label_prompts, text_prompts = batch["image"], batch["label_index"], [[class_dict[int(idx)] for idx in indices] for indices in batch["label_index"]], batch["text_prompt"]

        # 将数据移动到设备
        images = images.to(device)
        labels_tensor = text_prompts_to_tensor(text_prompts, class_dict, num_stage_labels,stage_text[0])
        labels = labels_tensor.to(device).float()

        # 反标准化图像
        #images = reverse_normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # 预处理图像并计算图像嵌入
        image_inputs = processor(images=images, return_tensors="pt")
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        image_embeddings = clip_model.get_image_features(**image_inputs)
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)

        # 计算 logits
        logits = image_embeddings @ text_embeddings.T

        # 计算损失
        loss = loss_fn(logits, labels)
        print(logits)
        print(labels)
        print(loss)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.update(1)
    pbar.close()

    # 保存模型
    # 将模型名称中的斜杠替换为下划线，并确保文件路径合法
    model_name_safe = model_name.replace("/", "_")
    torch.save(clip_model.state_dict(), f"{model_name_safe}_stage_{current_stage}_NoLoRA.pt")
    print(f"Completed Stage {current_stage}")
