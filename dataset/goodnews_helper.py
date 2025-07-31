import os
import re
from PIL import Image
# from pymongo import MongoClient
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, AutoImageProcessor
import json

class FieldParser:
    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(args.vllm)
    
    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values[0] 
    
    def transform_with_parse(self, features):
        to_return = {"id": features["id"]}
        ### 读取context
        context = features["conversations"][0]["value"]
        to_return["context"] = context
        ### 读取caption
        caption = features["conversations"][1]["value"]
        to_return["caption"] = caption
        image_path = features["image"]
        ### 读取graph_str
        to_return["graph_str"] = features["graph_str"]
        ### 读取图像
        with Image.open(os.path.join(self.args.base_dir, image_path)).convert("RGB") as pil:
            image = self._parse_image(pil)
            to_return["image"] = image
        
        return to_return
    

class ParseDataset(Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.meta = json.load(open(args.annotation, "r"))
        self.meta = self.meta[split]
        self.parser = FieldParser(args)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index])
        

def create_datasets(args):
    train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'val')
    test_dataset = ParseDataset(args, 'test')
    return train_dataset, dev_dataset, test_dataset