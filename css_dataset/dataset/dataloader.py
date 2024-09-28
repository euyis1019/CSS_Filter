from torch.utils.data import DataLoader as PytorchDataLoader
import torch
from PIL import Image
class Dataloader(PytorchDataLoader):
    def __init__(self, dataset, batch_size=1):
        collate_fn = self.collate_fn
        super(Dataloader, self).__init__(dataset, batch_size, collate_fn=collate_fn)

    @staticmethod
    def collate_fn(batch):#.tensor
        data = [item["data"][0].tensor for item in batch]
        label_index = [item["label_index"] for item in batch]
        text_prompt = [item["text_prompt"] for item in batch]
        data = torch.stack(data)
        return {"image": data, "label_index": label_index, "text_prompt": text_prompt}
