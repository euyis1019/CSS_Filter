from torch.utils.data import DataLoader as PytorchDataLoader
import torch
from PIL import Image
class Dataloader(PytorchDataLoader):
    def __init__(self, dataset, batch_size=1, transform=None,shuffle=True):
        print("bat", batch_size)
        self.transform = transform
        # 设置collate_fn为自定义函数
        collate_fn = self.collate_fn
        super(Dataloader, self).__init__(dataset, batch_size, collate_fn=collate_fn, shuffle=shuffle)

    def collate_fn(self, batch):
        # 提取batch中的数据、标签、文本和标签提示
        data = [item["data"][0] for item in batch]
        label = [item["data"][1] for item in batch]
        text_prompt = [item["text_prompt"] for item in batch]
        label_prompt = [item["label_prompt"] for item in batch]

        imgs = torch.stack(data)
        #labels = torch.stack(label)
        return imgs, label, text_prompt, label_prompt