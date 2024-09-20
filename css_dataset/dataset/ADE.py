from torch import overrides

from css_dataset.dataset.register import register_training_dataset,register_validation_dataset
from css_dataset.dataset.base_dataset import BaseIncrement,BaseSplit
from typing_extensions import override
import os



def _create_path_list(root_path,save_path):
    image_path = os.path.join(root_path,'images',"validation")
    target_path = os.path.join(root_path,'annotations',"validation")
    image_list = sorted(set(os.listdir(image_path)))
    target_list = sorted(set(os.listdir(target_path)))
    with open(save_path,'w') as f:
        for image,target in zip(image_list,target_list):
            if image.split(".")[-2] == target.split(".")[-2]:
                f.write(f"images/training/{image},annotations/training/{target}\n")

@register_training_dataset
class Split(BaseSplit):
    def __init__(self,scene_path=None,**kwargs):
        super().__init__(**kwargs)
        self.scene_path = scene_path
        if scene_path is not None:
            self.scene_list = self._get_scene(scene_path)

    @override
    def __getitem__(self, index):
        data_dict = super().__getitem__(index)
        if self.scene_path is None:
            data_dict["scene"] = "none"
            return data_dict
        else:
            data_dict["scene"] = self.scene_list[data_dict["path"][0]]
            return data_dict

    def _get_scene(self,scene_path=None):
        scene_list = {}
        scene_path = os.path.join(self.root,"list",scene_path)
        with open(scene_path,'r') as f:
            for lines in f:
                line = lines.strip().split(" ")
                scene_list[os.path.join(self.root,"images","training",line[0]+".jpg").replace(os.sep,"/")]=line[1]
        return scene_list

@register_training_dataset
class Increment(BaseIncrement):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)















if __name__ == '__main__':
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent.joinpath("data","ADEChallengeData2016")
    _create_path_list(str(path),"None")