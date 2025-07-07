import argparse
import os
import sys
import math

import ruamel.yaml as yaml

import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist


from models.model_grounding import Model

import torchvision


import utils
from dataset import create_dataset, create_sampler, create_loader, build_tokenizer
from scheduler import create_scheduler
from optim import create_optimizer

from utils.metric import getScorer

def train(model, data_loader, optimizer, tokenizer, epoch, device, scheduler, config):
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_bbox', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_giou', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ce', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_aux', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 10
    step_size = 100

    for i, (text, image, gt_box_iter, idx, ref_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        image = torch.tensor(image, device=device).to(non_blocking=True)
        idx = idx.to(device, non_blocking=True)

        if config['text_encoder'] == 'bert':
            out = tokenizer.batch_encode_plus(text, add_special_tokens=True, padding='longest', return_attention_mask=True, return_tensors="pt")
            text_input = out['input_ids'].to(device)
            text_mask = out['attention_mask'].to(device)
        elif config['text_encoder'] == 'clip':
            text_input = tokenizer(text).to(device)
            text_mask = None
        else:
            print('tokenizer error...')
            exit()

        losses = model(image, text_input, text_mask, targets=gt_box_iter, idx=idx, epoch=epoch, training=True, vit_type=config['vit_type'])

        loss = losses['loss_bbox'] + losses['loss_giou'] + losses['loss_aux'] + losses['loss_ce']

        optimizer.zero_grad()

        from torch.nn.utils.clip_grad import clip_grad_norm_
        clip_grad_norm_(model.parameters(), 1.0)

        loss.backward()
        optimizer.step()
        scheduler.step()
       
        # metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_bbox=losses['loss_bbox'].item())
        metric_logger.update(loss_giou=losses['loss_giou'].item())
        metric_logger.update(loss_ce=losses['loss_ce'].item())
        metric_logger.update(loss_aux=losses['loss_aux'].item())

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def batch_box_iou(box1, box2,threshold=0.5,iou_out=False):
    """
    :param box1:  N,4 xywh
    :param box2:  N,4 xywh
    :return: N
    """
    in_h = torch.min(box1[:,2], box2[:,2]) - torch.max(box1[:,0], box2[:,0])
    in_w = torch.min(box1[:,3], box2[:,3]) - torch.max(box1[:,1], box2[:,1])
    in_h=in_h.clamp(min=0.)
    in_w=in_w.clamp(min=0.)
    inter =in_h * in_w
    union = (box1[:,2] - box1[:,0]) * (box1[:,3] - box1[:,1]) + \
            (box2[:,2] - box2[:,0]) * (box2[:,3] - box2[:,1]) - inter
    iou = inter / union
    if iou_out:
        return iou >= threshold,iou
    else:
        return iou>=threshold
    

@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print('Computing features for evaluation...')
    start_time = time.time()
    box_ap = torch.zeros(1).to(device)

    from alive_progress import alive_bar

    with alive_bar(len(data_loader)) as bar:
        for ith_batch, data in enumerate(data_loader):
            bar()
            ref_iter, image_iter, gt_box_iter, iid = data
            image = image_iter.to(device=device, non_blocking=True)

            if config['text_encoder'] == 'bert':
                out = tokenizer.batch_encode_plus(ref_iter, add_special_tokens=True, padding='longest', return_attention_mask=True, return_tensors="pt")
                text_input = out['input_ids'].to(device)
                text_mask = out['attention_mask'].to(device)
            else:
                text_input = tokenizer(ref_iter).to(device)
                text_mask = None
            
            gt_box_iter=gt_box_iter.squeeze(1).to(device)
            gt_box_iter = torchvision.ops.box_convert(gt_box_iter, in_fmt="cxcywh", out_fmt="xyxy")
            
            output = model(image, text_input, text_mask, targets=gt_box_iter, text=ref_iter, idx=iid, vit_type=config['vit_type'])
            
            box = output['pred_box']
            box = torchvision.ops.box_convert(box, in_fmt="cxcywh", out_fmt="xyxy")
            box_iou, box_iou_v = batch_box_iou(gt_box_iter, box, iou_out=True)
            box_ap += (box_iou.type(torch.float32).sum())

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(box_ap, op=torch.distributed.ReduceOp.MAX)
    
    box_ap = box_ap / data_loader.dataset.data_size * 100.

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    return np.round(box_ap.cpu().numpy(), 3)

  


def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    world_size = utils.get_world_size()

    if args.epoch > 0:
        config['schedular']['epochs'] = args.epoch
        print(f"### set epochs to: {args.epoch}", flush=True)

    if args.bs > 0:
        config['batch_size_train'] = args.bs // world_size

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    # torch.backends.cudnn.deterministic=True

    print("Creating retrieval dataset", flush=True)
    
    # d_name = 'refcoco'
    train_dataset, val_dataset_dict, test_dataset_dict = create_dataset(args.dataset, config)

    train_dataset_size = len(train_dataset)

    if utils.is_main_process():
        print(f"### Train Files: {[os.path.basename(rpath) for rpath in config['train_file']]}")
        print(f"### Train data {train_dataset_size}, batch size, {config['batch_size_train']}, world_size {world_size}")
        print(f"### Validation: {[(k, len(dataset)) for k, dataset in val_dataset_dict.items()]}")
        print(f"### Test: {[(k, len(dataset)) for k, dataset in test_dataset_dict.items()]}")

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        train_sampler = create_sampler([train_dataset], [True], num_tasks, global_rank)
    else:
        train_sampler = [None]


    if args.dataset == 'refcoco':
        from dataset.dataset_grounding import collate_fn
    elif args.dataset in ['flickr30k', 'referIt']:
        from dataset.dataset_grounding import referIt_flickr30k_collate_fn as collate_fn


    train_loader = create_loader([train_dataset], train_sampler, batch_size=[config['batch_size_train']],
                                 num_workers=[4],
                                 is_trains=[True],
                                 collate_fns=[collate_fn])[0]

    val_test_loader_set = {}
    for k in val_dataset_dict.keys():
        val_test_loader_set[k] = create_loader([val_dataset_dict[k], test_dataset_dict[k]], [None, None],
                                      batch_size=[config['batch_size_test']] * 2,
                                      num_workers=[4, 4], is_trains=[False, False], collate_fns=[None, None])

    print("Creating model", flush=True)
    model = Model(config=config)
    model = model.to(device)

    print("Start training", flush=True)
    start_time = time.time()
    arg_opt = utils.AttrDict(config['optimizer'])

    optimizer = create_optimizer(arg_opt, model)


    arg_sche = utils.AttrDict(config['schedular'])
    arg_sche['step_per_epoch'] = math.ceil(train_dataset_size/(config['batch_size_train']*world_size))
    lr_scheduler = create_scheduler(arg_sche, optimizer)

    start_epoch = 0
    if args.checkpoint != 'null':
        print('load checkpint ....')
        state_dict = torch.load(args.checkpoint)
        model.load_state_dict(state_dict['model'], strict=False)



    for n, p in model.named_parameters():
        if 'backbone' in n or 'language_model' in n or 'image_model' in n:
            p.requires_grad = False #False
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module


    if config['text_encoder'] == 'clip':
        from clip import tokenize as tokenizer
    elif config['text_encoder'] == 'bert':
        from bert.tokenization_bert import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('/mnt_rela/wangyabing.wyb/ckpt/bert-base-uncased')
    else:
        print('tokenizer error...')
        exit()

    print("### output_dir, ", args.output_dir, flush=True)


   
    # print("Running zero-shot evaluation", flush=True)
    # for language, [val_loader, test_loader] in val_test_loader_set.items():
    #     box_ap = evaluation(model_without_ddp, val_loader, tokenizer, device, config)  # (1000, 1000)
    #     exit()
    #     if utils.is_main_process():
    #         print('======enter')
    #         # test_result = itm_eval_gd(score_test_t2i, test_loader.dataset.txt2img)
    #         print(f"{language}-test: {box_ap}")
    #     dist.barrier()
    # exit()
   


    max_epoch = config['schedular']['epochs']
    best = 0
    best_epoch = 0
    for epoch in range(start_epoch, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, device, lr_scheduler, config)
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        else:
            log_stats = {}

        mAP = 0
        for language, [val_loader, test_loader] in val_test_loader_set.items():
            box_ap = evaluation(model_without_ddp, test_loader, tokenizer, device, config)

            if utils.is_main_process():
                log_stats[f'{language}-test'] = str(box_ap)
                if utils.is_main_process():
                    if language == 'g_val' or language == 'val':
                        mAP = box_ap

            dist.barrier()

        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")

        if args.evaluate:
            break

        if utils.is_main_process():
            if mAP > best:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                best = mAP
                best_epoch = epoch

            elif epoch >= config['schedular']['epochs'] - 1:
            # if epoch >= 3:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, f'checkpoint_{epoch}.pth'))

            print('--------------')
            print(f"best_result: {best} ---- best_epoch: {best_epoch}")

        dist.barrier()
        torch.cuda.empty_cache()

    if utils.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write("best epoch: %d" % best_epoch)

        os.system(f"cat {args.output_dir}/log.txt")

    dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))
    print(f"best_result: {best} ---- best_epoch: {best_epoch}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)  # this script works for both mscoco and flickr30k
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')

    parser.add_argument('--epoch', default=-1, type=int)
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus")
    parser.add_argument('--evaluate', action='store_true')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    

    main(args, config)
