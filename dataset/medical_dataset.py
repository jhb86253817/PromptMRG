import json
import os
import torch
import numpy as np

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from .utils import my_pre_caption
import os

CONDITIONS = [
    'enlarged cardiomediastinum',
    'cardiomegaly',
    'lung opacity',
    'lung lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural effusion',
    'pleural other',
    'fracture',
    'support devices',
    'no finding',
]

SCORES = [
'[BLA]',
'[POS]',
'[NEG]',
'[UNC]'
]


class generation_train(Dataset):
    def __init__(self, transform, image_root, ann_root, tokenizer, max_words=100, dataset='mimic_cxr', args=None):
        
        self.annotation = json.load(open(os.path.join(ann_root),'r'))
        self.ann = self.annotation['train']
        self.transform = transform
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.max_words = max_words      
        self.dataset = dataset
        self.args = args
        with open('./data/mimic_cxr/clip_text_features.json', 'r') as f:
            self.clip_features = np.array(json.load(f))
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = ann['image_path']
        image = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
        image = self.transform(image)
        
        cls_labels = ann['labels']
        prompt = [SCORES[l] for l in cls_labels]
        prompt = ' '.join(prompt)+' '
        caption = prompt + my_pre_caption(ann['report'], self.max_words)
        cls_labels = torch.from_numpy(np.array(cls_labels)).long()
        clip_indices = ann['clip_indices'][:self.args.clip_k]
        clip_memory = self.clip_features[clip_indices]
        clip_memory = torch.from_numpy(clip_memory).float()

        return image, caption, cls_labels, clip_memory
    
class generation_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, tokenizer, max_words=100, split='val', dataset='mimic_cxr', args=None):
        self.annotation = json.load(open(os.path.join(ann_root), 'r'))
        if dataset == 'mimic_cxr':
            self.ann = self.annotation[split]
        else: # IU
            self.ann = self.annotation
        self.transform = transform
        self.max_words = max_words
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.args = args
        with open('./data/mimic_cxr/clip_text_features.json', 'r') as f:
            self.clip_features = np.array(json.load(f))
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        image_path = ann['image_path']
        image = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
        image = self.transform(image)

        caption = my_pre_caption(ann['report'], self.max_words)
        cls_labels = ann['labels']
        cls_labels = torch.from_numpy(np.array(cls_labels))
        clip_indices = ann['clip_indices'][:self.args.clip_k]
        clip_memory = self.clip_features[clip_indices]
        clip_memory = torch.from_numpy(clip_memory).float()

        return image, caption, cls_labels, clip_memory
