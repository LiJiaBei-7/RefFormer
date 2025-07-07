import torch
import clip
import torch.nn as nn

from PIL import Image
from PIL import ImageFile

import torch.nn.functional as F

import os
import h5py

from utils.misc import rescale_bboxes

import torchvision
import clip
from clip.model import build_model
from models.layers_6 import GA, MLP, CrossAttention, HA #HA2 as HA
from models.Bridger_6_2_1 import Bridger_ViT_1 as Bridger_VL_1
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou



class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        backbone, _ = clip.load('ViT-B/32', resolution=config['image_res'])
        self.backbone = backbone.cuda().float()
        ladder_dim, nhead = 128, 8
        self.bridger = Bridger_VL_1(d_model=ladder_dim, nhead=nhead)
        fpn_in = [768, 768, 512]
        fpn_out = [256, 512, 1024]
        stride = [1, 1, 1]
        self.neck = HA(in_channels=fpn_in, out_channels=fpn_out, stride=stride)
        num_layers = 2
        vis_dim = 512
        num_head = 8
        dim_ffn= 512
        dropout = 0.1
        intermediate = False
        self.decoder = GA(num_layers=num_layers,
                                      d_model=vis_dim,
                                      nhead=num_head,
                                      dim_ffn=dim_ffn,
                                      dropout=dropout,
                                      return_intermediate=intermediate)
        # Projector
        self.bbox_embed = MLP(dim_ffn, dim_ffn, 4, 3)
        self.bbox_coef = config['loss_bbox_weight']
        self.giou_coef = config['loss_giou_weight']
        self.reg_proj = nn.Linear(ladder_dim, dim_ffn)
        self.tgt_proj = nn.Linear(ladder_dim, dim_ffn)
        
    def set_aux_criterion_roi(self, output):
        reg_tokens, targets = output['reg_tokens'], output['targets']
        targets = torchvision.ops.box_convert(targets, in_fmt="cxcywh", out_fmt="xyxy")
        features = output['feat_ls']
        h, w = (features[0].shape)[-2:]
        rois = targets * torch.tensor([w,h,w,h]).unsqueeze(0).cuda()
        batch_indices = torch.arange(rois.size(0)).unsqueeze(1).cuda()
        rois = torch.cat([batch_indices, rois], dim=1)
        losses = torch.tensor(0.).cuda()
        for i, (feat,reg) in enumerate(zip(features, reg_tokens)):
            roi_align = torchvision.ops.RoIAlign(output_size=(7, 7), spatial_scale=1., sampling_ratio=1)
            # feat bsz,d, h,w
            output = roi_align(feat, rois)
            output = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1)).squeeze(-1).squeeze(-1) # bsz, d
            losses += F.l1_loss(reg.squeeze(0), output, reduction='mean')
            # loss = torch.sum(loss, dim=-1).mean()
        return losses
    
    # def set_aux_criterion(self, output):
    #     loss_dict = {}
    #     reg_tokens, targets = output['reg_tokens'], output['targets']
    #     loss_aux = torch.zeros(1).to(targets.device)
    #     for i, rt in enumerate(reg_tokens):
    #         predict_xy = self.bridger.aux_heads[i](rt.squeeze(0)) # B,2  
    #         predict_xy = torch.sigmoid(predict_xy)
    #         loss_aux_l1 = F.l1_loss(predict_xy, targets, reduction='none').sum() / predict_xy.size(0) #* self.bbox_coef
            
    #         _targets = torchvision.ops.box_convert(targets, in_fmt="cxcywh", out_fmt="xyxy")
    #         predict_xy = torchvision.ops.box_convert(predict_xy, in_fmt="cxcywh", out_fmt="xyxy")
    #         loss_aux_giou = (1 - torch.diag(generalized_box_iou(predict_xy, _targets))).sum() / predict_xy.size(0) #* self.giou_coef
    #         loss_aux += loss_aux_l1 + loss_aux_giou
    #     return loss_aux 

    def set_criterion(self, output):
        """
            Compute the losses related to the bounding boxes, 
            including the L1 regression loss and the GIoU loss
            targets, pred_box: cxcywh
        """
        pred_box, targets = output['pred_box'], output['targets']    
        
        batch_size = pred_box.shape[0]
        num_boxes = batch_size
        
        loss_bbox = F.l1_loss(pred_box, targets, reduction='none')
        
        targets = torchvision.ops.box_convert(targets, in_fmt="cxcywh", out_fmt="xyxy")
        pred_box = torchvision.ops.box_convert(pred_box, in_fmt="cxcywh", out_fmt="xyxy")
        loss_giou = 1 - torch.diag(generalized_box_iou(pred_box, targets))

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes #* self.bbox_coef
        losses['loss_giou'] = loss_giou.sum() / num_boxes #* self.giou_coef
        # losses['loss_giou'] = losses['loss_bbox'] = torch.tensor(0.).cuda().detach()
        losses['loss_aux'] = self.set_aux_criterion_roi(output)
        
        return losses
    

    def forward(self, image, text_ids, targets=None, idx=None, text=None, epoch=None, training=False):
        '''
            vis:list
                torch.Size([32, 768, 16, 16])
                torch.Size([32, 768, 16, 16])
                torch.Size([32, 512, 16, 16])
        
        '''     
        # padding mask used in decoder
        pad_mask = torch.zeros_like(text_ids).masked_fill_(text_ids == 0, 1).bool()
        
        vis, word, state, reg_tokens, condition_ls, feat_ls, attn_ls = self.bridger(image, text_ids, self.backbone)
        
        # tgt = self.tgt_proj(reg_tokens[-1])
        bs = state.size(0)
        reg = self.reg_proj(self.bridger.reg_token.expand(-1, bs, -1) + condition_ls[-1]) 
        tgt = torch.ones(reg.shape).cuda()

        fq = self.neck(vis, state)

        fq, attn = self.decoder(tgt, reg, fq, word, pad_mask=pad_mask, query_pos=None, return_attn=True) # torch.Size([32, 512, 257])

        reg_token = fq[...,0]
        pred_box = self.bbox_embed(reg_token).sigmoid() # torch.Size([32, 4])
        
#         if torch.distributed.get_rank() == 0:
#             with h5py.File('attn_data/base_10_1_39.hdf5', 'a') as f:
#                 for i in range(len(idx)):
#                     if str(idx[i].item()) not in f.keys():
#                         _attn = torch.cat([attn_ls[j][i] for j in range(3)], dim=0)
#                         f.create_dataset(f'{idx[i].item()}', data=_attn.detach().cpu())
#                         f.create_dataset(f'ref_{idx[i].item()}', data=text[i])
#                         f.create_dataset(f'bbox_{idx[i].item()}', data=targets[i].detach().cpu())
#                         f.create_dataset(f'pred_{idx[i].item()}', data=pred_box[i].detach().cpu())
            
            
        
#         if torch.distributed.get_rank() == 0:
#             with h5py.File('attn_data/decoder_attn/base_10_1_39.hdf5', 'a') as f:
#                 for i in range(len(idx)):
#                     if str(idx[i].item()) not in f.keys():
#                         f.create_dataset(f'{idx[i].item()}', data=attn[i, 0].detach().cpu())
#                         f.create_dataset(f'ref_{idx[i].item()}', data=text[i])
#                         f.create_dataset(f'bbox_{idx[i].item()}', data=targets[i].detach().cpu())
#                         f.create_dataset(f'pred_{idx[i].item()}', data=pred_box[i].detach().cpu())

        
        # reg_token = reg_tokens[-1]
        # pred_box = self.bridger.aux_heads[-1](reg_token.squeeze(0)).sigmoid()  # B,4 
        
        output = {'pred_box': pred_box}
        
        if training:
            output.update(dict(reg_tokens=reg_tokens))
            output.update(dict(targets=targets))
            output.update(dict(feat_ls=feat_ls))
            losses = self.set_criterion(output)
            return losses
 
        return output





