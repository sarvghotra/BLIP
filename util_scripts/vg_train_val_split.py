import os
import json
import random
from PIL import Image

import torch
from torch.utils.data import Dataset


from torchvision.datasets.utils import download_url

class vqa_dataset(Dataset):
    def __init__(self, transform, ann_root, vqa_root, vg_root, train_files=[], split="train"):
        self.split = split        

        self.transform = transform
        self.vqa_root = vqa_root
        self.vg_root = vg_root
        
        if split=='train':
            urls = {'vqa_train':'https://storage.googleapis.com/sfr-vision-language-research/datasets/vqa_train.json',
                    'vqa_val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/vqa_val.json',
                    'vg_qa':'https://storage.googleapis.com/sfr-vision-language-research/datasets/vg_qa.json'}
        
            self.annotation = []
            for f in train_files:
                download_url(urls[f],ann_root)
                self.annotation += json.load(open(os.path.join(ann_root,'%s.json'%f),'r'))
        else:
            download_url('https://storage.googleapis.com/sfr-vision-language-research/datasets/vqa_test.json',ann_root)
            self.annotation = json.load(open(os.path.join(ann_root,'vqa_test.json'),'r'))    
            
            download_url('https://storage.googleapis.com/sfr-vision-language-research/datasets/answer_list.json',ann_root)
            self.answer_list = json.load(open(os.path.join(ann_root,'answer_list.json'),'r'))    
                
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        if ann['dataset']=='vqa':
            image_path = os.path.join(self.vqa_root,ann['image'])    
        elif ann['dataset']=='vg':
            image_path = os.path.join(self.vg_root,ann['image'])  
            
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        if self.split == 'test':
            question = pre_question(ann['question'])   
            question_id = ann['question_id']            
            return image, question, question_id


        elif self.split=='train':                       
            
            question = pre_question(ann['question'])        
            
            if ann['dataset']=='vqa':               
                answer_weight = {}
                for answer in ann['answer']:
                    if answer in answer_weight.keys():
                        answer_weight[answer] += 1/len(ann['answer'])
                    else:
                        answer_weight[answer] = 1/len(ann['answer'])

                answers = list(answer_weight.keys())
                weights = list(answer_weight.values())

            elif ann['dataset']=='vg':
                answers = [ann['answer']]
                weights = [0.2]  

            return image, question, answers, weights
        
        
def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n        


if __name__ == "__main__":

    # download_url('https://storage.googleapis.com/sfr-vision-language-research/datasets/vg_qa.json', './')
    ann = json.load(open('/home/mila/s/sarvjeet-singh.ghotra/scratch/data/img/val/v2_Questions_Val_mscoco/vg_qa.json', 'r'))

    # with open('val_ids.json', 'r', encoding='utf-8') as f:
    #     for l in fi:
    #         val_ids = [int(id) for id in l.split(',')] 
    val_ids = json.load(open('/home/mila/s/sarvjeet-singh.ghotra/scratch/data/img/val/vg/val_ids.json', 'r'))
    
    print("no of val ids: ", len(val_ids))
    val_ids = set(val_ids)
    print("set size: ", len(val_ids))

    # print(type(ann))


    # val = []
    # train = []
    # for entry in ann:
    #     id = int(entry['question_id'])

    #     if id in val_ids:
    #         val.append(entry)
    #     else:
    #         train.append(entry)
    
    # with open('val_vg.json', 'w') as f:
    #     json.dump(val, f)
    
    # with open('train_vg.json', 'w') as f:
    #     json.dump(train, f)

    val_split = json.load(open("val_vg.json",'r'))
    print(val_split[:2])
    print()
    train_split = json.load(open("train_vg.json",'r'))
    print(train_split[:2])



    print(len(val_split))
    print(len(train_split))
