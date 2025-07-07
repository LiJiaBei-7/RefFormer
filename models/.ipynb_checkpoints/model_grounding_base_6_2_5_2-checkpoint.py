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
from models.layers_6_2_5_2 import GA, MLP, CrossAttention, HA #HA2 as HA
from models.Bridger_6_2_5_2 import Bridger_ViT_1 as Bridger_VL_1
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from scipy.optimize import linear_sum_assignment



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
        self.cls_embed = MLP(dim_ffn, dim_ffn, 2, 3)
        self.bbox_coef = config['loss_bbox_weight']
        self.giou_coef = config['loss_giou_weight']
        self.reg_proj = nn.Linear(ladder_dim, dim_ffn)
        self.tgt_proj = nn.Linear(ladder_dim, dim_ffn)
        
        self.cost_bbox = 5
        self.cost_class = 1
        self.cost_giou = 2
        
    def matcher(self, predict_xy, predict_cls, targets):
        bs, num_queries = predict_cls.shape[:2]
        out_prob = predict_cls.flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = predict_xy.flatten(0, 1)  # [batch_size * num_queries, 4]
        # Also concat the target labels and boxes
        # [3]  idx = 32, 1, 85  concat all labels
        cost_class = -out_prob[:, :1] # BxN,1
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, targets, p=1) # BxN, B

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(torchvision.ops.box_convert(out_bbox, in_fmt="cxcywh", out_fmt="xyxy"), torchvision.ops.box_convert(targets, in_fmt="cxcywh", out_fmt="xyxy")) # BxN, B
        
        # Final cost matrix   [100, 3]  bs*100个预测框分别和3个gt框的损失矩阵
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).detach().cpu()  # [bs, N, bs]

        sizes = [1 for _ in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        # 0 = Tensor[gt_num,]  匹配到的正样本idx       1 = Tensor[gt_num,]  gt的idx
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
   

    def set_aux_criterion(self, output):
        loss_dict = {}
        reg_tokens, targets = output['reg_tokens'], output['targets']
        loss_aux = torch.zeros(1).to(targets.device)
        for i, rt in enumerate(reg_tokens):
            batch_size = len(targets)
            predict_xy = torch.sigmoid(self.bridger.aux_heads[i](rt)).transpose(0,1) # B,N,4  
            predict_cls = self.bridger.aux_cls[i](rt).transpose(0,1) # B,N,2
            
            indices = self.matcher(predict_xy, predict_cls, targets)

            pre_index = torch.tensor([t[0].item() for t in indices]).cuda()
        
            cls_tgt = torch.ones(predict_cls.shape[:2], dtype=torch.long).cuda() #B,N
            cls_tgt[torch.arange(batch_size), pre_index] -= 1

            predict_cls = predict_cls.reshape(-1, 2)
            cls_tgt = cls_tgt.flatten()
            
            loss_ce = F.cross_entropy(predict_cls, cls_tgt)
            
            predict_xy = predict_xy[torch.arange(batch_size), pre_index]

            loss_aux_l1 = F.l1_loss(predict_xy, targets, reduction='none').sum() / predict_xy.size(0) #* self.cost_bbox
            
            _targets = torchvision.ops.box_convert(targets, in_fmt="cxcywh", out_fmt="xyxy")
            predict_xy = torchvision.ops.box_convert(predict_xy, in_fmt="cxcywh", out_fmt="xyxy")
            loss_aux_giou = (1 - torch.diag(generalized_box_iou(predict_xy, _targets))).sum() / predict_xy.size(0) #* self.cost_giou
            loss_aux += loss_aux_l1 + loss_aux_giou + loss_ce
        return loss_aux 

    def set_criterion(self, output):
        """
            Compute the losses related to the bounding boxes, 
            including the L1 regression loss and the GIoU loss
            targets, pred_box: cxcywh
        """
        # B,N,4  and B,N,2 and B,4
        pred_box, pred_cls, targets = output['pred_box'], output['pred_cls'], output['targets']    
        
        batch_size = pred_box.shape[0]
        num_boxes = pred_box.size(1)
        
        indices = self.matcher(pred_box, pred_cls, targets)
        
        pre_index = torch.tensor([t[0].item() for t in indices]).cuda()
        
        cls_tgt = torch.ones(pred_cls.shape[:2], dtype=torch.long).cuda() #B,N
        cls_tgt[torch.arange(batch_size), pre_index] -= 1
        
        pred_cls = pred_cls.reshape(-1, 2)
        cls_tgt = cls_tgt.flatten()
        loss_ce = F.cross_entropy(pred_cls, cls_tgt)
    
        pred_box = pred_box[torch.arange(batch_size), pre_index]
        loss_bbox = F.l1_loss(pred_box, targets, reduction='none')
        
        targets = torchvision.ops.box_convert(targets, in_fmt="cxcywh", out_fmt="xyxy")
        pred_box = torchvision.ops.box_convert(pred_box, in_fmt="cxcywh", out_fmt="xyxy")
        loss_giou = 1 - torch.diag(generalized_box_iou(pred_box, targets))

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / batch_size * self.cost_bbox
        losses['loss_giou'] = loss_giou.sum() / batch_size * self.cost_giou
        losses['loss_ce'] = loss_ce
        # losses['loss_giou'] = losses['loss_bbox'] = torch.tensor(0.).cuda().detach()
        losses['loss_aux'] = self.set_aux_criterion(output)* 0.1 #torch.tensor(0.).cuda().detach() #self.set_aux_criterion(output)
        
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
        
        vis, word, state, reg_tokens, feat_ls, attn_ls = self.bridger(image, text_ids, self.backbone)
        
        # tgt = self.tgt_proj(reg_tokens[-1])
        bs = state.size(0)
        reg = self.reg_proj(reg_tokens[-1]) 
        tgt = torch.zeros(reg.shape).cuda()

        fq = self.neck(vis, state)
        
        reg_token, attn = self.decoder(reg+tgt, fq, word, pad_mask=pad_mask, query_pos=None, return_attn=True) # torch.Size([32, 512, 257])

        pred_box = self.bbox_embed(reg_token).sigmoid() # torch.Size([32, 4])
        pred_cls = self.cls_embed(reg_token)
        
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
        pred_box = pred_box.transpose(0,1)
        pred_cls = pred_cls.transpose(0,1)
        output = {'pred_box': pred_box, 'pred_cls': pred_cls}
        
        if training:
            output.update(dict(reg_tokens=reg_tokens))
            output.update(dict(targets=targets))
            losses = self.set_criterion(output)
            return losses
        else:
            index = torch.argmax(pred_cls.softmax(-1)[:,:,0], dim=-1)
            pred_box = pred_box[torch.arange(len(index)), index]
            output['pred_box'] = pred_box
 
        return output





