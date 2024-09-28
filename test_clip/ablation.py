import sys
import os
import tqdm
import torch
import torch.nn as nn
from sympy.codegen import Print
from torch.nn import BCELoss
from transformers import CLIPProcessor, CLIPModel
import loralib as lora

# Set up paths and imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CSS_Filter.css_dataset import load_dataset_from_config, Dataloader, get_dataset_config
from ram import get_transform
from ram import get_transform,get_transform_no_Normalize,transfomr_384_voc_nor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)


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

def load_clip_model():
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    for param in clip_model.text_model.parameters():
        param.requires_grad = False
    return CLIPWithLearnableTemperature(clip_model)

class CLIPWithLearnableTemperature(nn.Module):
    def __init__(self, clip_model):
        super(CLIPWithLearnableTemperature, self).__init__()
        self.clip_model = clip_model
        # 初始化温度参数
        self.temperature = nn.Parameter(torch.tensor(0.3))  # 学习温度参数

    def forward(self, image_inputs, text_embeddings):
        image_embeddings = self.clip_model.get_image_features(**image_inputs)
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)

        # 计算 logits，使用可学习的温度参数
        logits = (image_embeddings @ text_embeddings.T) / self.temperature
        return logits



config = get_dataset_config("VOC")
config.increment_setting.save_stage_image_path = "default"
dataset, eval = load_dataset_from_config(config, 1, None)
#dataset.dataset.transform = transfomr_384_voc_nor()
stage_lengths = max(dataset.stage_index_dict.keys())

print("Stage lengths:", stage_lengths+1)
print(dataset.stage_index_dict[0])  # Stage 0 label
print(dataset.dataset.classes.items())  # dict_items([...])



# 假设你可以访问数据集的标签计数

class BCEWithLogitsLossWithLabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1, pos_weight=None):
        super(BCEWithLogitsLossWithLabelSmoothing, self).__init__()
        self.smoothing = smoothing
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        targets = targets * (1 - self.smoothing) + self.smoothing * (1 - targets)
        loss = self.bce_loss(logits, targets)
        return loss




smoothing = 0.1  # Adjust as needed
#loss_fn = BCELoss(reduction="mean")
loss_fn = BCEWithLogitsLossWithLabelSmoothing(smoothing=smoothing, pos_weight=None).to(device)

batch_size = 16
num_stages = max(dataset.stage_index_dict.keys())


loss_fn = loss_fn.to(device)
start = int(input("Please input the first stage: "))
# 确保模型处于训练模式
def compute_pos_weight(dataset):
    pass
    # num_samples = len(dataset)
    # label_counts = np.zeros(num_classes)
    # for sample in dataset:
    #     labels = sample['labels']  # 根据你的数据集调整
    #     label_counts += labels
    # pos_weight = (num_samples - label_counts) / label_counts
    # pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)
    # return pos_weight

K = int(input("How many epochs?"))
for current_stage in range(start, num_stages+1):
    # 更新数据集和数据加载器
    clip_model = load_clip_model()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, clip_model.parameters()), lr=1e-5)

    dataset.update_stage(current_stage)

    # 准备当前阶段的文本提示
    stage_text = dataset.class_name
    num_stage_labels = len(stage_text)

    print(num_stage_labels)
    print(len(stage_text))

    # 计算文本嵌入，并使用 detach()
    print(stage_text)
    text_inputs = processor(text=stage_text, return_tensors="pt", padding=True,do_rescale=False)
    print(text_inputs)

    text_inputs = text_inputs.to(device)
    print(text_inputs)

    text_embeddings = clip_model.clip_model.get_text_features(**text_inputs)
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    #torch.save(text_embeddings, f'text_embeddings_stage{current_stage}.pt')
    text_embeddings = text_embeddings.detach().to(device)  # 添加 .detach()
    class_dict = dataset.dataset.classes
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    # 训练循环
    for k in range(0, K):
        dataset.update_stage(current_stage)
        print(f"Starting Stage {current_stage}_epoch{k}")
        dataloader = Dataloader(dataset, batch_size=batch_size)
        pbar = tqdm.tqdm(total=len(dataloader))
        for i, batch in enumerate(dataloader):
            images, labels, label_prompts, text_prompts = batch["image"], batch["label_index"], [[class_dict[int(idx)] for idx in indices] for indices in batch["label_index"]], batch["text_prompt"]
            # print("Image tensor max value:", images.max())
            # print("Image tensor min value:", images.min())
            # print("Image tensor shape:", images.shape)
            # 将数据移动到设备
            images = images.to(device)
            labels_tensor = text_prompts_to_tensor(text_prompts, class_dict, num_stage_labels,stage_text[0])
            labels = labels_tensor.to(device).float()


            image_inputs = processor(images=images, return_tensors="pt", do_rescale=False)

            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            image_embeddings = clip_model.clip_model.get_image_features(**image_inputs)
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
            print("image Embeddings", image_embeddings)
            #print("TEXTEMBED", text_embeddings)
            # 计算 logits
            logits = clip_model(image_inputs, text_embeddings)

            print("original", logits)
            logits = torch.sigmoid(logits)
            print("sigmoid", logits)
            # 计算损失
            loss = loss_fn(logits, labels)
            # print("logits",logits)
            print("labels", labels)
            print("loss", loss)
            writer.add_scalar('Loss/train', loss.item(), i)
            writer.add_scalar('Temperature', clip_model.temperature.item(),i)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.update(1)
        pbar.close()
        writer.close()
        model_name_safe = model_name.replace("/", "_")
        torch.save(clip_model.state_dict(), f"{model_name_safe}_stage_{current_stage}_NoLoRA.pt")
        print(f"Completed Stage {current_stage}")
