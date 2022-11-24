import os
import json
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from data.utils import pre_question

from torchvision.datasets.utils import download_url

class dataset(Dataset):
    def __init__(self, transform, ann_root_files, imgs_path, split="train"):
        self.split = split

        self.transform = transform
        self.imgs_path = imgs_path
        self.split = split

        self.annotation = []
        for f in ann_root_files:
            self.annotation += json.load(open(f,'r'))

        self.annotation = self.annotation[:1200]

        if split != 'train':

            self.ques_id_to_ans = {}
            for idx, ann in enumerate(self.annotation):

                if type(ann['answer']) == list:
                    # VQAv2 dataset with multiple answers
                    ans = {'answers': [{'answer': x} for x in ann['answer']]}
                else:
                    # VG dataset
                    ans = {'answers': [{'answer': ann['answer']}]}

                self.ques_id_to_ans[ann['question_id']] = ans

    def get_ques_id_to_ans(self):
        return self.ques_id_to_ans

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]

        # if ann['dataset']=='vqa':
        image_path = os.path.join(self.imgs_path,ann['image'])
        # elif ann['dataset']=='vg':
        #    image_path = os.path.join(self.imgs_path,ann['image'])

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
                weights = [1.0]

            return image, question, answers, weights

    def evaluate(self, predictions):
        ############## FIXME ##############
        # assert len(predictions) == len(self.annotation), \
        #     "Number of predictions {} should be equal to the dataset size {}.".format(len(predictions), len(self.annotation))

        correct = 0
        for pred in predictions:
            ques_id = pred['question_id']
            pred_answer = pred['answer']

            ann = self.annotation[self.ques_id_to_idx_annotation[ques_id]]
            if ann['dataset']=='vqa':
                answers = ann['answer']
            elif ann['dataset']=='vg':
                answers = [ann['answer']]

            answers = [a.lower() for a in answers]
            if pred_answer.lower() in answers:
                correct += 1
                break

        return correct/len(self.annotation)


def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n