import sys
import os
import tqdm

print(sys.path)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CSS_Filter.css_dataset import load_dataset_from_config, Dataloader, get_dataset_config
from ram import get_transform

# 加载数据集配置
config = get_dataset_config("VOC")
config.increment_setting.save_stage_image_path = "default"
dataset = load_dataset_from_config(config, 1, None)

# 获取每个 stage 的长度
stage_lengths = dataset.stage_len
print("Stage lengths:", stage_lengths)

# 定义变换
transform = get_transform(image_size=384)
dataset.dataset.transform = transform

# 设置 Dataloader
batch_size = 8
dataloader = Dataloader(dataset, batch_size=batch_size, shuffle = False)

# 使用 tqdm 进度条显示
pbar = tqdm.tqdm(total=len(dataloader))
i = 0
current_stage = 1

# 遍历 stage
while current_stage < len(stage_lengths):
    print(f"Processing stage {current_stage + 1}/{len(stage_lengths)}")

    # 重置 dataloader 来处理当前 stage
    stage_length = stage_lengths[current_stage]
    dataset.update_stage(current_stage)
    dataloader = Dataloader(dataset, batch_size=batch_size)

    for img, label, text_prompt, label_prompt in dataloader:
        i += batch_size
        print(f"Stage {current_stage}, Batch {i}:")
        print("Text prompt:", text_prompt)
        print("Label prompt:", label_prompt)
        pbar.update()

        # 检查是否已经遍历完当前 stage
        if i >= stage_length:
            print(f"Completed stage {current_stage + 1}")
            current_stage += 1  # 进入下一个 stage
            i = 0  # 重置计数器
            break  # 跳出当前循环，开始下一个 stage

pbar.close()
