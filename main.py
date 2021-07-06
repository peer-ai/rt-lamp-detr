# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, pdb

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from util.plot_utils import plot_precision_recall
from glob import glob
from sklearn.metrics import confusion_matrix
from test_detr import predict_bbox, visualize_decoder_encoder_att, get_results, get_result_from_target, plot_confusion_matrix

try:
    import wandb
except:
    wandb = None
    
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    
    ## IC
    parser.add_argument('--nonos', action='store_true',
                        help="Non overlapping loss")

    parser.add_argument('--early_stopping', default=50, type=int)
    parser.add_argument('--plot_every', default=10, type=int)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--test', default=False, type=bool, help='test mode or not')
    parser.add_argument('--eval_test', action='store_true', help='eval test after training is complete or not')
    
    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients    
    parser.add_argument('--col_loss_coef', default=1, type=float)
    parser.add_argument('--row_loss_coef', default=1, type=float)
    parser.add_argument('--nono_loss_coef', default=0, type=float)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--custom_data_path', type=str)
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

class Args:
    pass

def log_confusion_acc(preds, targets, epoch, prefix='', wandb=None, class_names=['COVID19 Positive', 'IC Positive', 'Negative', 'No Object']):
    # Confusion matrix
    log = {}
    conf_mat=confusion_matrix(targets, preds)
    # print('conf_mat', conf_mat)
    for i,ci in enumerate(conf_mat):
        for j,cj in enumerate(ci):
            log["{}_confusion_matrix_{}_{}".format(prefix, class_names[i], class_names[j])] = cj

    # Per-class accuracy
    total = conf_mat.sum(1)
    total[total==0] = 1
    class_accuracy = 100*conf_mat.diagonal()/total
    
    for i,acc in enumerate(class_accuracy):
        log["{}_accuracy_{}".format(prefix, class_names[i])] = acc
        
    if wandb is not None:
        wandb.log(log, step=epoch)
        wandb.log({"{}_confusion_matrix".format(prefix) : wandb.plot.confusion_matrix(probs=None,
                y_true=targets, preds=preds, class_names=class_names)}, step=epoch)
    return log

def get_acc(preds, targets):
    conf_mat=confusion_matrix(targets, preds)
    # Per-class accuracy
    total = conf_mat.sum(1)
    total[total==0] = 1
    class_accuracy = 100*conf_mat.diagonal()/total
    return sum(class_accuracy[:3])

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=25, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_loss):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0
            
def main(**kwargs):
    args = Args()
    args.__dict__ = kwargs
    
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)
    

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if not args.test:
        dataset_train = build_dataset(image_set='train', args=args)
        dataset_val = build_dataset(image_set='val', args=args)

        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=args.num_workers)
        
        data_loader_train_eval = DataLoader(dataset_train, args.batch_size, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        
        data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

        # if args.dataset_file == "coco_panoptic":
        #     # We also evaluate AP during panoptic training, on original coco DS
        #     coco_val = datasets.coco.build("val", args)
        #     base_ds = get_coco_api_from_dataset(coco_val)
        # else:
        #     base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            print('args.resume', args.resume)
            checkpoint = torch.load(args.resume, map_location='cpu')
            
            
        del checkpoint["model"]["class_embed.weight"]
        del checkpoint["model"]["class_embed.bias"]
        del checkpoint["model"]["query_embed.weight"]
        
        model_without_ddp.load_state_dict(checkpoint['model'],strict=False)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        base_ds = get_coco_api_from_dataset(dataset_val)
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    def run_test(model, epoch, wandb=None, image_set='test'):
#         dataset_test = build_dataset(image_set='test', args=args)
#         sampler_test = torch.utils.data.SequentialSampler(dataset_test)
#         data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
#                                      drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        # base_ds = get_coco_api_from_dataset(dataset_test)

        #test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
        #                                      data_loader_test, base_ds, device, args.output_dir)
        #
        #log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
        #             'n_parameters': n_parameters}
        #log_stats['test_mAP'] = log_stats['test_coco_eval_bbox'][1]
        #if wandb is not None:
        #    wandb.log({"test_log_stats": log_stats})
                
        if args.output_dir:
            output_dir_test = (output_dir / '{}'.format(image_set))
            output_dir_test.mkdir(exist_ok=True)
            #utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir_test / "test.pth")
        
            # plot precision recall
            fns = [output_dir_test / "{}.pth".format(image_set)]
            fig, axs = plot_precision_recall(fns)
            fig.savefig(output_dir_test / '{}_pr.pdf'.format(image_set))
            if wandb is not None:
                wandb.log({'{}_pr.jpg'.format(image_set): wandb.Image(fig)}, step=epoch)
        
            # logs  
            # with (output_dir_test / "log.txt").open("a") as f:
            #     f.write(json.dumps(log_stats) + "\n")
        
            # split bbox prediction
            preds = []
            targets = []
            for i,(img,target) in enumerate(build_dataset(image_set=image_set, args=args, noTransforms=True)):
                # print(samples.shape)
                fig = predict_bbox(model, img, device, threshold=0.7)
                fig.savefig(output_dir_test / '{}_bbox_{}.pdf'.format(image_set, i))
                if wandb is not None:
                    wandb.log({'{}_bbox_{}.jpg'.format(image_set, i): wandb.Image(fig)}, step=epoch)
                 
                fig, enfig = visualize_decoder_encoder_att(model, img, device)
                fig.savefig(output_dir_test / '{}_decoder_att_{}.pdf'.format(image_set, i))
                
                enfig.savefig(output_dir_test / '{}_encoder_att_{}.jpg'.format(image_set, i))
                if wandb is not None:
                    wandb.log({'{}_decoder_att_{}.jpg'.format(image_set, i): wandb.Image(fig)}, step=epoch)
                    wandb.log({'{}_encoder_att_{}.jpg'.format(image_set, i).format(i): wandb.Image(enfig)}, step=epoch)
                
                
                results = get_results(model, img, device, threshold=0.7)
                tresults = get_result_from_target(target)
                preds += results.values()
                targets += tresults.values()
            
            fig = plot_confusion_matrix(targets, preds)
            fig.savefig(output_dir_test / f'{image_set}_confusion_matrix.svg')
            if wandb is not None:
                wandb.log({f'{image_set}_confusion_matrix.jpg'.format(image_set, i): wandb.Image(fig)}, step=epoch)
                
    if args.test:
        run_test(model)
        return
    
    print("Start training")
        
    # [WANDB] Start a new run
    wandb.init(project='chroml-detr', entity='peer-ai', name=args.name, config=kwargs)
    # [WANDB] Log gradients and model parameters
    if wandb is not None:
        wandb.watch(model)
    
    earlyStopping = EarlyStopping(args.early_stopping)

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if wandb is not None:
            wandb.log({'config': wandb.Table(data=[[str(k),str(v)] for (k,v) in kwargs.items()], columns=['Param', 'Value'])}, step=epoch)

        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        
        base_ds = get_coco_api_from_dataset(dataset_train)
        train_eval_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_train_eval, base_ds, device, args.output_dir
        )
        
        base_ds = get_coco_api_from_dataset(dataset_val)
        val_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )
        
        
        # validation accuracy
        # curr_acc = get_acc(val_stats['preds'], val_stats['targets'])
        curr_acc = -val_stats['loss']
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
              
            best_acc = -np.Inf
            if (output_dir/'best_accuracy.txt').exists():
                with open(str(output_dir/'best_accuracy.txt')) as f:
                    best_acc = float(""+f.read())
                    print('best_acc', best_acc)

            if curr_acc > best_acc:
                for bcheckf in glob(str(output_dir)+"/best_accuracy_checkpoint*.pth"):
                    Path(bcheckf).unlink()
                checkpoint_paths.append(output_dir / f'best_accuracy_checkpoint{epoch:04}.pth')
                checkpoint_paths.append(output_dir / f'best_accuracy_checkpoint.pth')
                
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        print('curr_val_loss', curr_acc)
        with open(str(output_dir/'best_accuracy.txt'), 'w') as f:
            f.write(str(curr_acc))

        # draw box on val set
        if epoch % args.plot_every == args.plot_every-1:
            if args.output_dir:
                output_dir_val = (output_dir / 'val')
                output_dir_val.mkdir(exist_ok=True)
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir_val / "val.pth")
                run_test(model, epoch, wandb, 'val')            
        
        if args.eval_test and epoch % args.plot_every == args.plot_every-1:
            dataset_test = build_dataset(image_set='test', args=args)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
            data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
            
            base_ds = get_coco_api_from_dataset(dataset_test)
            test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                                  data_loader_test, base_ds, device, args.output_dir)

#             log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
#                          'n_parameters': n_parameters}
#             log_stats['test_mAP'] = log_stats['test_coco_eval_bbox'][1]
            
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'train_eval_{k}': v for k, v in train_eval_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            
            log_stats['train_eval_mAP'] = log_stats['train_eval_coco_eval_bbox'][1]
            log_stats['val_mAP'] = log_stats['val_coco_eval_bbox'][1]
            log_stats['test_mAP'] = log_stats['test_coco_eval_bbox'][1]
            
            for i in range(12):
                log_stats['train_eval_AP{}'.format(i)] = log_stats['train_eval_coco_eval_bbox'][i]
                log_stats['val_AP{}'.format(i)] = log_stats['val_coco_eval_bbox'][i]
                log_stats['test_AP{}'.format(i)] = log_stats['test_coco_eval_bbox'][i]
            
            if args.output_dir:
                output_dir_test = (output_dir / 'test')
                output_dir_test.mkdir(exist_ok=True)
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir_test / "test.pth")
            
            log_confusion_acc(test_stats['preds'], test_stats['targets'], epoch, 'test', wandb)
        
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'train_eval_{k}': v for k, v in train_eval_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            # 12 types of AP/AR from cocoeval mAP is at index [1]
            # stats[0] = _summarize(1)
            # stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            # stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            # stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            # stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            # stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            # stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            # stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            # stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            # stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            # stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            # stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])        
            log_stats['train_eval_mAP'] = log_stats['train_eval_coco_eval_bbox'][1]
            log_stats['val_mAP'] = log_stats['val_coco_eval_bbox'][1]
            
            for i in range(12):
                log_stats['train_eval_AP{}'.format(i)] = log_stats['train_eval_coco_eval_bbox'][i]
                log_stats['val_AP{}'.format(i)] = log_stats['val_coco_eval_bbox'][i]
                
        log_confusion_acc(train_stats['preds'], train_stats['targets'], epoch, 'train', wandb)
        log_confusion_acc(val_stats['preds'], val_stats['targets'], epoch, 'val', wandb)
        if wandb is not None:
            wandb.log(log_stats, step=epoch)
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)
                    
                    #fns = [Path(f'{epoch:03}.pth')]
                    fns = [Path('{}/eval/latest.pth'.format(output_dir))]
                    fig, axs = plot_precision_recall(fns)
                    wandb.log({'eval_pr': wandb.Image(fig)}, step=epoch)
                    
                    # plot progression
                    #fns = [Path(f) for f in sorted(glob('outputs/{}/eval/*.pth'.format(args.name)))]
                    fns = [Path(f) for f in sorted(glob('{}/eval/*.pth'.format(output_dir)))]
                    fig, axs = plot_precision_recall(fns)
                    wandb.log({'eval_pr_progress': wandb.Image(fig)}, step=epoch)

        total_time = time.time() - start_time
        if wandb is not None:
            wandb.log({'training_time_s': int(total_time)}, step=epoch)
              
        if earlyStopping(curr_acc):
#         if earlyStopping(-log_stats['val_loss']):
            for checkpoint_path in [output_dir / 'checkpoint_early_stop.pth']:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
            break
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    if args.output_dir:
        with open(os.path.join(args.output_dir,'training_time.txt'), 'w') as f:
            f.write(str(total_time_str))
            
    if args.eval_test:
        # test on the best accuracy model
        temp = torch.load(output_dir / 'best_accuracy_checkpoint.pth')
        args = temp['args']
        model, _, _ = build_model(args)
        model.to(device)
        model.load_state_dict(temp['model'])
        model.eval()

        run_test(model, epoch, wandb)
        
    if wandb is not None:
        wandb.finish()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(**vars(args))
