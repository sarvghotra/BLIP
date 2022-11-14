import os
import json
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
# from data.utils import pre_question

from torchvision.datasets.utils import download_url

if __name__ == "__main__":

    urls = {'vqa_train':'https://storage.googleapis.com/sfr-vision-language-research/datasets/vqa_train.json',
            'vqa_val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/vqa_val.json',
            'vg_qa':'https://storage.googleapis.com/sfr-vision-language-research/datasets/vg_qa.json'}
        
    annotation = []
    for f in ['vg_qa']:
        download_url(urls[f],"./")
        annotation += json.load(open(os.path.join("./",'%s.json'%f),'r'))

    print(annotation[:2])

    random.shuffle(annotation)

    eval_val_split = annotation[:10000]
    train_val_split = annotation[10000:]

    with open('./eval_vg_qa_split_10k.json', "w") as out_file:
        json.dump(eval_val_split, out_file)
    
    with open('./train_vg_qa_split.json', "w") as out_file2:
        json.dump(train_val_split, out_file2)

    annotation = []
    annotation1 = json.load(open('./eval_vg_qa_split_10k.json','r'))
    print(annotation1[:2])
    annotation2 = json.load(open('./train_vg_qa_split.json','r'))
    print(annotation2[:2])

