'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import numpy as np
import torch
import argparse
import os
try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml
import random
import time
import datetime
import json
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.blip_vqa import blip_vqa
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
from data.vqa_dataset import vqa_collate_fn
from data.utils import save_result
from util_vqa.eval_vqav2 import VQAEval
import wandb

# import multiprocessing

# import cv2
# cv2.setNumThreads(0)

def save_ckpt(train_stats, epoch, step, model_without_ddp, optimizer, config):
    if utils.is_main_process():     
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'step': step,
                        'epoch': epoch
                    }                
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write(json.dumps(log_stats) + "\n")                        
                
        save_obj = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
            'step': step,
            'epoch': epoch
        }
        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%d.pth'%step)) 
        print("Saved ckpoint: ", 'checkpoint_%d.pth'%step)


@torch.no_grad()
def evaluation(model, data_loader, device, config) :
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 50
    
    result = []
    
    if config['inference']=='rank':   
        answer_list = data_loader.dataset.answer_list
        answer_candidates = model.tokenizer(answer_list, padding='longest', return_tensors='pt').to(device)    
        answer_candidates.input_ids[:,0] = model.tokenizer.bos_token_id
    
    step = 0
    for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header, step, len(data_loader))):        
        image = image.to(device,non_blocking=True)             

        if config['inference']=='generate':
            answers = model(image, question, train=False, inference='generate') 
            
            for answer, ques_id in zip(answers, question_id):
                ques_id = int(ques_id.item())       
                result.append({"question_id":ques_id, "answer":answer})             
            
        elif config['inference']=='rank':    
            answer_ids = model(image, question, answer_candidates, train=False, inference='rank', k_test=config['k_test'])      

            for ques_id, answer_id in zip(question_id, answer_ids):
                result.append({"question_id":int(ques_id.item()), "answer":answer_list[answer_id]})  
        step += 1 

    return result

@torch.no_grad()
def distributed_evaluation(model, data_loader, device, config):
    result = evaluation(model, data_loader, device, config)
    world_size = dist.get_world_size()
    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, result)

    all_results = []
    for gath_res in output:
        all_results.extend(gath_res)
    
    # print("After all_gather_object, size of output: ", len(output), " rank: ", dist.get_rank())
    # print("After all_gather_object, size of all_results: ", len(all_results), " rank: ", dist.get_rank())

    # print("output: ", output)
    # print("all_results: ", all_results)

    return all_results


def list_to_dict(lst):
    result = {}
    for item in lst:
        result[item['question_id']] = {'answer': item['answer']}
    return result

def eval_acc(model, data_loader, device, config):
    result = distributed_evaluation(model, data_loader, device, config)
    result = list_to_dict(result)
    ans = data_loader.dataset.get_ques_id_to_ans()

    vqaEval = VQAEval(result, ans, n=2)   #n is precision of accuracy (number of places after decimal), default is 2
    acc = vqaEval()
    model.train()
    return acc


def train(model, data_loader, val_data_loader, ood_val_data_loader, optimizer, scaler, step, epoch, device, start_step, is_training_resumed, total_steps, nb_acc_steps):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    saved_for_this_step = False
    eval_for_this_step = False
    header = 'Train Epoch: [{}]'.format(epoch)

    wandb.log({'epoch': epoch})
    
    print_freq = config['print_freq']
    CKPT_SAVE_FREQ = config['ckpt_save_freq']
    EVAL_FREQ = config['eval_freq']
    
    optimizer.zero_grad()
    for i,(image, question, answer, weights, n) in enumerate(metric_logger.log_every(data_loader, print_freq, header, step, total_steps)):
        image, weights = image.to(device,non_blocking=True), weights.to(device,non_blocking=True)      

        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            loss = model(image, question, answer, train=True, n=n, weights=weights)        
        
        scaler.scale(loss / nb_acc_steps).backward()

        if (i+1) % nb_acc_steps == 0 or (i+1) == len(data_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            step += 1
            wandb.log({'step': step})
            saved_for_this_step = False
            eval_for_this_step = False

        # loss.backward()
        # optimizer.step()    

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if step > 0 and step % CKPT_SAVE_FREQ == 0 and not saved_for_this_step:
            model_without_ddp = model
            if args.distributed:
                model_without_ddp = model.module
            
            train_stats = {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
            save_ckpt(train_stats, epoch, step, model_without_ddp, optimizer, config)
            saved_for_this_step = True

        if step > 0 and step % EVAL_FREQ == 0 and not eval_for_this_step:
            # evaluate
            acc = eval_acc(model, val_data_loader, device, config)
            print("Validation accuracy: ", acc)
            wandb.log({"val_acc": acc, "step": step})
            acc = eval_acc(model, ood_val_data_loader, device, config)
            print("OOD Validation accuracy: ", acc)
            wandb.log({"ood_val_acc": acc})
            eval_for_this_step = True

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}, step


def find_latest_ckpt(path):
    latest_step = 0
    latest_ckpt = None
    for ckpt_file in os.listdir(path):
        if "checkpoint_" in ckpt_file:
            ckpt_step = int(ckpt_file.split('_')[1][:-4])

            if ckpt_step > latest_step:
                latest_ckpt = ckpt_file
                latest_ckpt = os.path.join(path, latest_ckpt)
    return latest_ckpt


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    wandb.init(project="transfer-bottleneck",
                    entity="sarvghotra",
                    # notes="VQAv2 exp",
                    # tags=["debug", "gdss"],
                    name=config['exp_name'],)
    wandb.config = {
        **config,
    }

    #### Dataset #### 
    print("Creating vqa datasets")
    datasets = create_dataset('ood_vqa', config)   
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]
    
    train_loader, test_loader, test_loader_ood = create_loader(datasets,samplers,
                                              batch_size=[config['batch_size_train'],config['batch_size_test'], config['batch_size_test']],
                                              num_workers=[8,4,4],is_trains=[True, False, False], 
                                              collate_fns=[vqa_collate_fn, None, None]) 
    
    print("Train size: ", len(train_loader))
    print("test size: ", len(test_loader))
    print("OOD test size: ", len(test_loader_ood))
    wandb.config.update({
        "train_size": len(train_loader),
        "test_size": len(test_loader),
        "ood_test_size": len(test_loader_ood),
    })
    #### Model #### 
    print("Creating model")
    model = blip_vqa(pretrained=config['pretrained'], image_size=config['image_size'], 
                       vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])

    wandb.config.update({
        'pretrained_ckpt': config['pretrained']
    })
    scaler = torch.cuda.amp.GradScaler()
    model = model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    step = 0
    start_step = 0
    epoch = 0
    is_training_resumed = False
    total_steps = len(train_loader) * config['max_epoch']
    wandb.config.update({
        "total_steps": total_steps,
        "skipped_steps": 0,
        "resume_step": 0,
        "resume_epoch": 0,
    })

    # batch size // (num_gpus * batch_size_per_gpu)
    total_nb_gpus = torch.distributed.get_world_size() if args.distributed else 1
    NUM_ACCUMULATION_STEPS = config['batch_size_train'] // ( config['batch_size_per_gpu'] * total_nb_gpus)
    print("NUM_ACCUMULATION_STEPS: ", NUM_ACCUMULATION_STEPS)
    wandb.config.update({
        "NUM_ACCUMULATION_STEPS": NUM_ACCUMULATION_STEPS,
    })

    if not args.evaluate:
        latest_ckpt = find_latest_ckpt(args.output_dir)
        if latest_ckpt is not None:
            is_training_resumed = True
            checkpoint = torch.load(latest_ckpt, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epoch']
            step = checkpoint['step']
            
            start_step = step + 1

            print("Loaded checkpoint from: ", latest_ckpt)
            wandb.config.update({
                "resume_from": latest_ckpt,
            })

            for ep in range(0, epoch):
                if not args.evaluate:        
                    if args.distributed:
                        train_loader.sampler.set_epoch(epoch)
            
            # while step + i < start_step:
            skipped = 0
            # if start_step % len(train_loader) != 0:
            #     for i, _ in enumerate(train_loader):
            #         skipped += 1
            #         if ((start_step % len(train_loader)) - i) == 0:
            #             break
            #         if i % 100 == 0:
            #             print("skipped: ", i)
                
            #     if skipped == len(train_loader):
            #         epoch += 1
            
            print("Total datasize: ", len(train_loader))
            print("total steps: ", total_steps)
            print("skipped steps: ", skipped)
            print("Resumed training from step: ", step, " epoch: ", epoch)
            wandb.config.update({
                "skipped_steps": skipped,
                "resume_step": step,
                "resume_epoch": epoch,
            })
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    best = 0
    best_epoch = 0 
       
    print("Start training")
    start_time = time.time()    
    for epoch in range(epoch, config['max_epoch']):
        if not args.evaluate: 
            if not is_training_resumed: 
                # It has been already done in the resume code above
                if args.distributed:
                    train_loader.sampler.set_epoch(epoch)
                    
                cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
                
            train_stats, step = train(model,
                                    train_loader,
                                    test_loader,
                                    test_loader_ood,
                                    optimizer, 
                                    scaler, 
                                    step, 
                                    epoch, 
                                    device, 
                                    start_step, 
                                    is_training_resumed, 
                                    total_steps, 
                                    NUM_ACCUMULATION_STEPS) 

        else:         
            break        
        
        save_ckpt(train_stats, epoch, step, model_without_ddp, optimizer, config)
        dist.barrier()         

    vqa_result = evaluation(model_without_ddp, test_loader, device, config)        
    result_file = save_result(vqa_result, args.result_dir, 'vqa_result')  

    val_acc = eval_acc(model, test_loader, device, config)
    print("val_acc: ", val_acc)
    wandb.log({"val_acc": val_acc})
    ood_val_acc = eval_acc(model, test_loader_ood, device, config)
    print("ood_val_acc: ", ood_val_acc)
    wandb.log({"ood_val_acc": ood_val_acc})
                      
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    # multiprocessing.set_start_method('spawn') 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/vqa.yaml') 
    parser.add_argument('--output_dir', default='output/VQA')
    parser.add_argument('--evaluate', action='store_true')      
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)