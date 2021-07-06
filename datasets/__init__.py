# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args, noTransforms=False, noNormalize=False):
    if args.dataset_file == 'LAMP_Covid':
        from .LAMP_Covid import build as build_LAMP_Covid
        return build_LAMP_Covid(image_set,args,noTransforms,noNormalize)
    raise ValueError(f'dataset {args.dataset_file} not supported')
