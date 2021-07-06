import numpy as np
import time
import math, os
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'retina'

import ipywidgets as widgets
# from IPython.display import display, clear_output
#from torchsummary import summary
from models import build_model

from torch import nn
from torchvision.models import resnet50
import datasets.transforms as T

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

# colors for visualization
#COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
#          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# This set of colors should follow the LAMP solution colors
COLORS = [[194/256.,182/256.,104/256.],[46/256.,126/256.,65/256.],[194/256.,96/256.,209/256.]]

CLASSES = ['COVID-19 Positive','IC Positive','COVID-19 Negative']
# CLASSES = ['+','ic+','-']

normalize = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    normalize,
])

sample_map = {}
sample_map[(0,0,1,0)] = 1
sample_map[(0,1,1,1)] = 2
sample_map[(0,2,1,2)] = 3
sample_map[(0,3,1,3)] = 4
sample_map[(0,4,1,4)] = 5
sample_map[(0,5,1,5)] = 6
sample_map[(0,6,1,6)] = 7
sample_map[(0,7,1,7)] = 8
sample_map[(2,0,3,0)] = 9
sample_map[(2,1,3,1)] = 10
sample_map[(2,2,3,2)] = 11
sample_map[(2,3,3,3)] = 12
sample_map[(2,4,3,4)] = 13
sample_map[(2,5,3,5)] = 14
sample_map[(2,6,3,6)] = 15
sample_map[(2,7,3,7)] = 16

idMap = {}
for k,v in sample_map.items():
    idMap[(k[0],k[1])] = v
    idMap[(k[2],k[3])] = v

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescaled_box_cxcywh_to_xywh(x, size):
    img_w, img_h = size
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w)*img_w, (y_c - 0.5 * h)*img_h,
         w*img_w, h*img_h]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    #b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).cuda()
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def draw_box(p, cl, ax, xmin, ymin):
    # text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
    text = f'{p[cl]:0.2f}'
    ax.text(xmin, ymin-50, text, fontsize=11,
            bbox=dict(facecolor='yellow', alpha=0.5))
    
def draw_pos(row, col, ax, xmin, ymin):
    # text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
    text = f'#{idMap[(row,col)]}\n({row},{col})'
    ax.text(xmin, ymin-120, text, fontsize=11,
            bbox=dict(facecolor='grey', alpha=0.5))
#     text = f''
#     ax.text(xmin, ymin-190, text, fontsize=11,
#             bbox=dict(facecolor='grey', alpha=0.5))
    
def plot_results(pil_img, prob, boxes):
    
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        
        draw_box(p, cl, ax, xmin, ymin)
    plt.axis('off')
    plt.show()
    
def plot_confusion_matrix(targets, preds, labels=['COVID-19 positive', 'COVID-19 negative', 'RNA absent', 'False positive', 'Undetected']):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import pylab
    sns.set(font_scale=1.8)    
    cm = confusion_matrix(targets, preds, labels=labels)
    cmap = sns.color_palette("Blues", as_cmap=True)
    fig = pylab.figure()
    f = sns.heatmap(cm, annot=True, cmap=cmap, xticklabels=labels, yticklabels=labels, square=True,cbar=True,fmt="d")
    pylab.xlabel("Predicted")
    pylab.ylabel("Actual")
    return fig
    
def get_result_from_target(target):
    rows = target['rows']
    cols = target['cols']
    labels = target['labels']
    preds = {}
    results = {}
    for r,c,l in zip(rows,cols,labels):
        preds[(r.item(), c.item())] = l.item()
#     print(target['rows'])

    for k,v in sample_map.items():
        if (k[0],k[1]) in preds and (k[2],k[3]) in preds:
            if preds[(k[0],k[1])] == 0 and preds[(k[2],k[3])] == 1:
                results[v] = 'COVID-19 positive'
            elif preds[(k[0],k[1])] == 2 and preds[(k[2],k[3])] == 1:
                results[v] = 'COVID-19 negative'
            elif preds[(k[0],k[1])] == 2 and preds[(k[2],k[3])] == 2:
                results[v] = 'RNA absent'
            else:
                results[v] = 'False positive'
        else:
            results[v] = 'Undetected'
    return results

def get_result(prob, row_probas, col_probas):
    results = {}
    
    preds = {}
    for p, rp, cp in zip(prob, row_probas, col_probas):
        cl = p.argmax()
        row = rp.argmax().item()
        col = cp.argmax().item()
        preds[(row, col)] = cl.item()
    
    for k,v in sample_map.items():
        if (k[0],k[1]) in preds and (k[2],k[3]) in preds:
            if preds[(k[0],k[1])] == 0 and preds[(k[2],k[3])] == 1:
                results[v] = 'COVID-19 positive'
            elif preds[(k[0],k[1])] == 2 and preds[(k[2],k[3])] == 1:
                results[v] = 'COVID-19 negative'
            elif preds[(k[0],k[1])] == 2 and preds[(k[2],k[3])] == 2:
                results[v] = 'RNA absent'
            else:
                results[v] = 'False positive'
        else:
            results[v] = 'Undetected'
            
    return results

def plot_results_one_color_per_class(pil_img, prob, row_probas, col_probas, boxes):
    w, h = pil_img.size
    fig = plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, rp, cp, (xmin, ymin, xmax, ymax), c in zip(prob, row_probas, col_probas, boxes.tolist(), colors):
        cl = p.argmax()
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=COLORS[cl], linewidth=3))        
        row = rp.argmax().item()
        col = cp.argmax().item()
        draw_pos(row, col, ax, xmin, ymin)            
        draw_box(p, cl, ax, xmin, ymin)    
    
    # ID #1: RESULT
    # COVID-19 positive, COVID-19 negative, RNA absent, false positive
    results = get_result(prob, row_probas, col_probas)
    texts = []
    for k,v in results.items():
        text = f'#{k}: {v}'
        texts.append(text)
    text = "\n".join(texts)
    ax.text(w, 0, text, fontsize=11,
            horizontalalignment='left',
            verticalalignment='top',
            bbox=dict(facecolor='grey', alpha=0.5))
        
    plt.axis('off')
    plt.show()
    return fig
    
def plot_results_specific_classes(pil_img, prob, boxes, target_class_list):
    fig = plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        cl = p.argmax()
        if cl in target_class_list:
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=COLORS[cl], linewidth=3))

            draw_box(p, cl, ax, xmin, ymin)
            
    plt.axis('off')
    plt.show()
    return fig

@torch.no_grad()
def get_results(trained_model, im, device, threshold=0.7, idx_show='all'):
    trained_model.eval()
#     torch.set_grad_enabled(False);
    
    # im = Image.open(curr_img_fullpath)
    width,height = im.size
    img = transform(im, None)[0].unsqueeze(0)
    
    t_start = time.perf_counter()
    
    # propagate through the model
    outputs = trained_model(img.to(device))
    # outputs = trained_model(img)
    
    
    # keep only predictions with 0.7+ confidence
    #pdb.set_trace()
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1] # ignore the last class (background)
    keep = probas.max(-1).values > threshold

    t_end = time.perf_counter()
    print('Duration = ' + str(t_end-t_start))
    
    curr_bboxes = outputs['pred_boxes'][0, keep].cpu().numpy()
    curr_probas = probas[keep].cpu().numpy()
    
    row_probas = outputs['pred_row_logits'].softmax(-1)[0, :, :-1]
    col_probas = outputs['pred_col_logits'].softmax(-1)[0, :, :-1]
    curr_row_probas = row_probas[keep].cpu().numpy()
    curr_col_probas = col_probas[keep].cpu().numpy()
    
    # Apply non-maximal suppression
    from nms import nms
    boxes = rescaled_box_cxcywh_to_xywh(torch.tensor(curr_bboxes), im.size).numpy()
    indices = nms.boxes(boxes, curr_probas.max(1))
    curr_bboxes = curr_bboxes[indices]
    curr_probas = curr_probas[indices]
    curr_row_probas = curr_row_probas[indices]
    curr_col_probas = curr_col_probas[indices]
    
    curr_bboxes = torch.tensor(curr_bboxes)
    curr_probas = torch.tensor(curr_probas)
    curr_row_probas = torch.tensor(curr_row_probas)
    curr_col_probas = torch.tensor(curr_col_probas)
    
    return get_result(curr_probas, curr_row_probas, curr_col_probas)
    
    
@torch.no_grad()
def backbone_features(trained_model, im, device):
    trained_model.eval()
#     torch.set_grad_enabled(False);
    
    # im = Image.open(curr_img_fullpath)
    width,height = im.size
    img = transform(im, None)[0].unsqueeze(0)
    
    t_start = time.perf_counter()
    
    # propagate through the model
    if isinstance(img, (list, torch.Tensor)):
        img = nested_tensor_from_tensor_list(img)
            
    features, pos = trained_model.backbone(img.to(device))
    return features, pos

@torch.no_grad()
def predict_bbox(trained_model, im, device, threshold=0.7, idx_show='all'):
    trained_model.eval()
#     torch.set_grad_enabled(False);
    
    # im = Image.open(curr_img_fullpath)
    width,height = im.size
    img = transform(im, None)[0].unsqueeze(0)
    
    t_start = time.perf_counter()
    
    # propagate through the model
    outputs = trained_model(img.to(device))
    # outputs = trained_model(img)
    
    
    # keep only predictions with 0.7+ confidence
    #pdb.set_trace()
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1] # ignore the last class (background)
    keep = probas.max(-1).values > threshold

    t_end = time.perf_counter()
    print('Duration = ' + str(t_end-t_start))
    
    curr_bboxes = outputs['pred_boxes'][0, keep].cpu().numpy()
    curr_probas = probas[keep].cpu().numpy()
    
    row_probas = outputs['pred_row_logits'].softmax(-1)[0, :, :-1]
    col_probas = outputs['pred_col_logits'].softmax(-1)[0, :, :-1]
    curr_row_probas = row_probas[keep].cpu().numpy()
    curr_col_probas = col_probas[keep].cpu().numpy()
    
    # Apply non-maximal suppression
    from nms import nms
    boxes = rescaled_box_cxcywh_to_xywh(torch.tensor(curr_bboxes), im.size).numpy()
    # print(boxes)
    # print('curr_probas.max(1)', curr_probas.max(1))
    indices = nms.boxes(boxes, curr_probas.max(1))
    # print('indices', indices)
    curr_bboxes = curr_bboxes[indices]
    curr_probas = curr_probas[indices]
    curr_row_probas = curr_row_probas[indices]
    curr_col_probas = curr_col_probas[indices]
    
    curr_bboxes = torch.tensor(curr_bboxes)
    curr_probas = torch.tensor(curr_probas)
    curr_row_probas = torch.tensor(curr_row_probas)
    curr_col_probas = torch.tensor(curr_col_probas)
    
    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(curr_bboxes, im.size)
    
    if idx_show == 'all': # display the bb for all classes
        fig = plot_results_one_color_per_class(im, curr_probas, curr_row_probas, curr_col_probas, bboxes_scaled)
    else: # pick one class
        for idx in idx_show:
            fig = plot_results_specific_classes(im, curr_probas, bboxes_scaled, target_class_list = [idx])
    
    return fig
        
@torch.no_grad()
def visualize_decoder_encoder_att(model, im, device):
#     torch.set_grad_enabled(False);
    
    # im = Image.open(curr_img_fullpath)
    width,height = im.size
    img = transform(im, None)[0].unsqueeze(0)

    # use lists to store the outputs via up-values
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []

    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
            lambda self, input, output: enc_attn_weights.append(output[1])
        ),
        model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
            lambda self, input, output: dec_attn_weights.append(output[1])
        ),
    ]

    # propagate through the model
    outputs = model(img.to(device))
    
    for hook in hooks:
        hook.remove()
        
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1] # ignore the last class (background)
    keep = probas.max(-1).values > 0.7
    keep = torch.where(keep)[0]
    idx = torch.randperm(keep.nelement())
    keep = keep.view(-1)[idx].view(keep.size())

#     print('keep.shape', keep.shape)
#     keepidx = torch.where(keep)
#     keepidx.shuffle()
#     print('keepidx', keepidx[0])
#     print(torch.randperm(len(keepidx[0])))
#     keep = keepidx[torch.randperm(len(keepidx[0])).cpu().numpy()]
    print('keep',keep)
    
    # Apply non-maximal suppression
    curr_bboxes = outputs['pred_boxes'][0, keep].cpu().numpy()
    curr_probas = probas[keep].cpu().numpy()
    
    #curr_bboxes, curr_probas = non_max_suppression_fast(boxes=curr_bboxes, probs=curr_probas, overlapThresh=0.99)
    curr_bboxes = torch.tensor(curr_bboxes)
    curr_probas = torch.tensor(curr_probas)
    
    # convert boxes from [0; 1] to image scales
    print('im.size, img.shape', im.size, img.shape)
    bboxes_scaled = rescale_bboxes(curr_bboxes, im.size)
    bboxes_scaled_img = rescale_bboxes(curr_bboxes, (img.shape[3], img.shape[2]))
    
#     print('bboxes_scaled', bboxes_scaled)
#     print('bboxes_scaled_img', bboxes_scaled_img)

    # don't need the list anymore
    conv_features = conv_features[0]
    enc_attn_weights = enc_attn_weights[0]
    dec_attn_weights = dec_attn_weights[0]
    
    # Truncate the number of objects to be displayed
    target_num_imgs = 4
    # get the feature map shape
    h, w = conv_features['0'].tensors.shape[-2:]

    #fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
    fig, axs = plt.subplots(ncols=target_num_imgs, nrows=2, figsize=(22, 7))
    colors = COLORS * 100
    idxs = []
    count = 0
    
    for idx, ax_i, (xmin, ymin, xmax, ymax), (xmin_img, ymin_img, xmax_img, ymax_img) in zip(keep, axs.T, bboxes_scaled, bboxes_scaled_img):

        # IC: Display parts of the results
        if count==target_num_imgs:
            break
            
#         idxs.append((int((xmin_img+xmax_img).item()/2), int((ymin_img+ymax_img).item()/2)))
        idxs.append((int((ymin_img+ymax_img).item()/2), int((xmin_img+xmax_img).item()/2),idx.item()))

        ax = ax_i[0]
        ax.imshow(dec_attn_weights[0, idx].view(h, w).cpu().numpy())
        ax.axis('off')
        ax.set_title(f'query id: {idx.item()}')
        ax = ax_i[1]
        ax.imshow(im)
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color='blue', linewidth=3))
        ax.axis('off')
        ax.set_title(CLASSES[probas[idx].argmax()])
        count += 1

    fig.tight_layout()
    
    print('idxs', idxs)
    # encoder
    enfig = encoder_att(im, img, conv_features, enc_attn_weights, idxs)
    
    return fig, enfig

@torch.no_grad()
def visualize_decoder_encoder_att_combined(model, im, device):
    width,height = im.size
    img = transform(im, None)[0].unsqueeze(0)

    conv_features, enc_attn_weights, dec_attn_weights = [], [], []

    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
            lambda self, input, output: enc_attn_weights.append(output[1])
        ),
        model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
            lambda self, input, output: dec_attn_weights.append(output[1])
        ),
    ]

    # propagate through the model
    outputs = model(img.to(device))
    
    for hook in hooks:
        hook.remove()
        
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1] # ignore the last class (background)
    keep = probas.max(-1).values > 0.7
    keep = torch.where(keep)[0]
    idx = torch.randperm(keep.nelement())
    keep = keep.view(-1)[idx].view(keep.size())

    
    # Apply non-maximal suppression
    curr_bboxes = outputs['pred_boxes'][0, keep].cpu().numpy()
    curr_probas = probas[keep].cpu().numpy()
    
    #curr_bboxes, curr_probas = non_max_suppression_fast(boxes=curr_bboxes, probs=curr_probas, overlapThresh=0.99)
    curr_bboxes = torch.tensor(curr_bboxes)
    curr_probas = torch.tensor(curr_probas)
    
    # convert boxes from [0; 1] to image scales
    print('im.size, img.shape', im.size, img.shape)
    bboxes_scaled = rescale_bboxes(curr_bboxes, im.size)
    bboxes_scaled_img = rescale_bboxes(curr_bboxes, (img.shape[3], img.shape[2]))
    
    # don't need the list anymore
    conv_features = conv_features[0]
    enc_attn_weights = enc_attn_weights[0]
    dec_attn_weights = dec_attn_weights[0]
    
    # Truncate the number of objects to be displayed
    target_num_imgs = 4
    # get the feature map shape
    h, w = conv_features['0'].tensors.shape[-2:]

    #fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
    fig, axs = plt.subplots(ncols=target_num_imgs, nrows=3, figsize=(9, 7))
    colors = COLORS * 100
    idxs = []
    count = 0
    
    for idx, ax_i, (xmin, ymin, xmax, ymax), (xmin_img, ymin_img, xmax_img, ymax_img) in zip(keep, axs.T, bboxes_scaled, bboxes_scaled_img):

        # IC: Display parts of the results
        if count==target_num_imgs:
            break
            
#         idxs.append((int((xmin_img+xmax_img).item()/2), int((ymin_img+ymax_img).item()/2)))
        idx_o = (int((ymin_img+ymax_img).item()/2), int((xmin_img+xmax_img).item()/2),idx.item())

        ax = ax_i[0]
        ax.imshow(im)
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color='blue', linewidth=3))
        ax.axis('off')
        ax.set_title(CLASSES[probas[idx].argmax()])
        
        ax = ax_i[1]
        fact = 32
        e_idx = (idx_o[0] // fact, idx_o[1] // fact)
        f_map = conv_features['0']
        print("Encoder attention:      ", enc_attn_weights[0].shape)
        print("Feature map:            ", f_map.tensors.shape)

        # get the HxW shape of the feature maps of the CNN
        shape = f_map.tensors.shape[-2:]
        # and reshape the self-attention to a more interpretable shape
        sattn = enc_attn_weights[0].reshape(shape + shape)
        ax.imshow(sattn[..., e_idx[0], e_idx[1]].cpu().numpy(), cmap='cividis', interpolation='nearest')
        ax.axis('off')
        ax.set_title(f'Encoder Attention ID:{idx_o[2]}')
        
        ax = ax_i[2]
        ax.imshow(dec_attn_weights[0, idx].view(h, w).cpu().numpy())
        ax.axis('off')
        ax.set_title(f'Decoder Attention ID: {idx.item()}')
        
        count += 1

    fig.tight_layout()
    
    # print('idxs', idxs)
    # encoder
    # enfig = encoder_att(im, img, conv_features, enc_attn_weights, idxs)
    
    return fig

def encoder_att(im, img, conv_features, enc_attn_weights, idxs=[(300, 180), (500, 320), (770, 620), (990, 700),]):
    # Visualize encoder self-attention weights
    # output of the CNN
    f_map = conv_features['0']
    print("Encoder attention:      ", enc_attn_weights[0].shape)
    print("Feature map:            ", f_map.tensors.shape)

    # get the HxW shape of the feature maps of the CNN
    shape = f_map.tensors.shape[-2:]
    # and reshape the self-attention to a more interpretable shape
    sattn = enc_attn_weights[0].reshape(shape + shape)
    print("Reshaped self-attention:", sattn.shape)
    
    # downsampling factor for the CNN, is 32 for DETR and 16 for DETR DC5
    fact = 32

    # here we create the canvas
    fig = plt.figure(constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))
    # and we add one plot per reference point
    gs = fig.add_gridspec(2, 4)
    axs = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[0, -1]),
        fig.add_subplot(gs[1, -1]),
    ]

    # for each one of the reference points, let's plot the self-attention
    # for that point
    for idx_o, ax in zip(idxs, axs):
        idx = (idx_o[0] // fact, idx_o[1] // fact)
        print(idx)
        ax.imshow(sattn[..., idx[0], idx[1]].cpu().numpy(), cmap='cividis', interpolation='nearest')
        ax.axis('off')
        ax.set_title(f'self-attention id:{idx_o[2]}')

    # and now let's add the central image, with the reference points as red circles
    fcenter_ax = fig.add_subplot(gs[:, 1:-1])
    fcenter_ax.imshow(im)
    for (y, x, _) in idxs:
        scale = im.height / img.shape[-2]
        x = ((x // fact) + 0.5) * fact
        y = ((y // fact) + 0.5) * fact
        fcenter_ax.add_patch(plt.Circle((x * scale, y * scale), fact // 2, color='r'))
        fcenter_ax.axis('off')
    
    return fig