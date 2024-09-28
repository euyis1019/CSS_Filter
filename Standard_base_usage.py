# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).parent.parent.absolute()))

# from dataset.base_dataset import BaseIncrement
# from dataset.VOC import Segmentation
# class TestIncrement(BaseIncrement):
#     def __init__(self,**kwargs):
#         super().__init__(**kwargs)
#
#
# dataset_config = get_dataset_config("VOC")
# print(dataset_config)
# # print(dataset_config)
# init_dict = auto_init(TestIncrement,dataset_config.increment_setting)
# print(init_dict)
# init_dict1 = auto_init(Segmentation,dataset_config.dataset_setting)
# print(init_dict1)(
# init_dict["segmentation_config"]=init_dict1
# init_dict["labels"]=dataset_config.training.task_1.index_order[1]
# init_dict["labels_old"]=dataset_config.training.task_1.index_order[0]
#
# dataset = TestIncrement(**init_dict)
# # dataset = Segmentation(**init_dict)
# for i in dataset:
#     print(i["text_prompt"])


# config = get_dataset_config("ADE")
# from dataset import dataset_entrypoints
#
# DATASET=dataset_entrypoints("ADE.Segmentation")
# init_dict = auto_init(DATASET,config.dataset_setting)
#
# dataset = DATASET(**init_dict)
# for i in dataset:
#     print(i["text_prompt"])
#     print(i["scene"])
#
#
# test_list = [13,5]
# print(str(test_list))
from css_dataset import load_dataset_from_config
from css_dataset import Dataloader,get_dataset_config
from ram import get_transform

transform = get_transform(image_size=384)
config = get_dataset_config("VOC")
print(config)
config.increment_setting.save_stage_image_path = "default"

dataset, eval = load_dataset_from_config(config,1,None)
dataset.dataset.transform=transform
dataloader = Dataloader(dataset,batch_size=8)
i=0
for batch in dataloader:
    i+=1
    print("text", batch["text_prompt"])
    print("label", batch["label_index"])
    if i >= 5:
        print("Update")
        dataset.update_stage(1)