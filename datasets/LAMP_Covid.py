# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as LAMP_Covid_mask

import datasets.transforms as T

import pdb

class LAMP_CovidDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(LAMP_CovidDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertLAMP_CovidPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(LAMP_CovidDetection, self).__getitem__(idx)
        # print('target', target)
        #target['labels'] = target['labels'] - 1 # Hack to remove the 0th class (tubes)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
            
        return img, target

def convert_LAMP_Covid_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = LAMP_Covid_mask.frPyObjects(polygons, height, width)
        mask = LAMP_Covid_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertLAMP_CovidPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        
        positions = [obj["pos"] for obj in anno if "pos" in obj]
        positions = torch.tensor(positions, dtype=torch.int64)
        
        rows = [obj["row"] for obj in anno if "row" in obj]
        rows = torch.tensor(rows, dtype=torch.int64)
        
        cols = [obj["col"] for obj in anno if "col" in obj]
        cols = torch.tensor(cols, dtype=torch.int64)
        
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_LAMP_Covid_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["positions"] = positions
        target["rows"] = rows
        target["cols"] = cols
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

import imgaug.augmenters as iaa

def make_LAMP_Covid_transforms(image_set,use_augment=True,noNormalize=False):

    if noNormalize:
        normalize = T.Compose([T.ToTensor()])
    else:
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        if use_augment:
            return T.Compose([
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
                # T.RandomHorizontalFlip(),
                T.RandomResize([800], max_size=1333),
#                 T.RandomSelect(
#                     T.RandomResize(scales, max_size=1333),
#                     T.Compose([
#                         T.RandomResize([400, 500, 600]),
#                         T.RandomSizeCrop(384, 600),
#                         T.RandomResize(scales, max_size=1333),
#                     ])
#                 ),
#                 T.RandomRotate(p=0.35, possible_angles = [-5,5]),
                T.RandomAffine(rotate=(-1, 1), translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)}, scale=(0.99, 1.01), shear=(-1,1)),
#                 iaa.Affine(rotate=(-5, 5), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),
                normalize,
            ])
        else:
            return T.Compose([T.RandomResize([800], max_size=1333),
                              normalize])
    
    if image_set in ['val', 'test']:
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args, noTransforms=False, noNormalize=False):
    
    root = Path(args.custom_data_path)
    assert root.exists(), f'provided LAMP_Covid path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train", root / "train" / f'annotations.json'),
        "val": (root / "val", root / "val" / f'annotations.json'),
        "test": (root / "test", root / "test" / f'annotations.json'),
    }
    
    
    img_folder, ann_file = PATHS[image_set]
    dataset = LAMP_CovidDetection(img_folder, ann_file, transforms=None if noTransforms else make_LAMP_Covid_transforms(image_set,args.augment,noNormalize), return_masks=args.masks)
    return dataset
