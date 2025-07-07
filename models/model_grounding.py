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
from models.layers_grounding_final import GA, MLP, CrossAttention, HA, HA_swin #HA2 as HA
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from scipy.optimize import linear_sum_assignment



class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        vit_type = config['vit_type']
        if vit_type == 'vit_32':
            vit_path = '/mnt_rela/wangyabing.wyb/ckpt/clip/ViT-B-32.pt'
        elif vit_type == 'vit_16':
            vit_path = '/mnt_rela/wangyabing.wyb/ckpt/clip/ViT-B-16.pt'
        elif vit_type == 'vit_large':
            vit_path = '/mnt_rela/wangyabing.wyb/ckpt/clip/ViT-L-14.pt'
        elif vit_type == 'swin':
            vit_path = '/mnt_rela/wangyabing.wyb/ckpt/swin_transformer/swin_base_patch4_window12_384_22k.pth'
            ck_bert = '/mnt_rela/wangyabing.wyb/ckpt/bert-base-uncased'
        else:
            print('vit_type (ckpt) Error!')
            exit()
        
        
        num_reg = config['num_reg']
        bridger_stages = config['bridger_stages']
        aggregate_layers = config['aggregate_layers']

        
        ladder_dim, nhead = 128, 8
        fpn_in = [768, 768, 512]
        fpn_out = [256, 512, 1024]
        d_img = 768 
        d_txt = 512
        dim_ffn=vis_dim = 512

        stride = [1, 1, 1]
        
        if vit_type == 'swin':
            ladder_dim, nhead = 128, 8
            fpn_in =  [256, 512, 512]
            fpn_out = [256, 512, 1024]
            d_img = 768 
            d_txt = 512
            dim_ffn=vis_dim = 512

            from models.qa_grounding_swin import WrapperModel
            from bert.multimodal_bert import MultiModalBert
            from lib import multimodal_segmentation
            model_name = 'lavt'
            config['swin_type'] = 'base'
            config['fusion_drop'] = 0.0
            image_model = multimodal_segmentation.__dict__[model_name](pretrained=vit_path, args=config)
            language_model = MultiModalBert.from_pretrained(ck_bert, embed_dim=image_model.backbone.embed_dim)
            
            self.qa = WrapperModel(image_model.backbone, language_model)

            self.neck = HA_swin(in_channels=fpn_in, out_channels=fpn_out, stride=stride, d_model=fpn_out[1])

        else:
            from models.qa_grounding import QA
            backbone, _ = clip.load(vit_path, resolution=config['image_res'])
            self.backbone = backbone.cuda().float()

            print('Visual Backbone Is CLIP ...')
            if vit_type == 'vit_large':
                from models.qa_grounding_large import Bridger_ViT_1 as Bridger
                d_img = 1024
                d_txt = 768
                fpn_in = [256, 512, 512]
                fpn_out = [256, 512, 1024]
                dim_ffn = vis_dim = 768
                ladder_dim, nhead = 256, 8
            else:
                from models.qa_grounding import QA
            
            self.qa = QA(d_model=ladder_dim, nhead=nhead, num_reg=num_reg, \
                                    bridger_stages=bridger_stages, aggregate_layers=aggregate_layers,
                                    d_img=d_img, d_txt=d_txt)

            self.neck = HA(in_channels=fpn_in, out_channels=fpn_out, stride=stride, d_model=fpn_out[1])

        num_layers = 2
        num_head = 8
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

        empty_weight = torch.ones(2)
        empty_weight[-1] = 0.25
        self.register_buffer('empty_weight', empty_weight)

        
    def matcher(self, predict_xy, predict_cls, targets, sizes=None):
        bs, num_queries = predict_cls.shape[:2]
        out_prob = predict_cls.flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = predict_xy.flatten(0, 1)  # [batch_size * num_queries, 4]
        # Also concat the target labels and boxes
        # [3]  idx = 32, 1, 85  concat all labels
        cost_class = -out_prob[:, :1] # BxN,1
        # Compute the L1 cost between boxes
        
        targets = targets.type(torch.float32)
        cost_bbox = torch.cdist(out_bbox, targets, p=1) # BxN, B

        # Compute the giou cost betwen boxes
        _out_bbox = torchvision.ops.box_convert(out_bbox, in_fmt="cxcywh", out_fmt="xyxy")
        _targets = torchvision.ops.box_convert(targets, in_fmt="cxcywh", out_fmt="xyxy")
        
        cost_giou = -generalized_box_iou(_out_bbox, _targets) # BxN, B
        
        # Final cost matrix   [100, 3]  bs*100个预测框分别和3个gt框的损失矩阵
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).detach().cpu()  # [bs, N, bs]

        if sizes is None:
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
            predict_xy = torch.sigmoid(self.qa.aux_heads[i](rt)).transpose(0,1) # B,N,4  
            predict_cls = self.qa.aux_cls[i](rt).transpose(0,1) # B,N,2
            
            indices = self.matcher(predict_xy, predict_cls, targets)

            pre_index = torch.tensor([t[0].item() for t in indices]).cuda()
        
            cls_tgt = torch.ones(predict_cls.shape[:2], dtype=torch.long).cuda() #B,N
            cls_tgt[torch.arange(batch_size), pre_index] -= 1

            predict_cls = predict_cls.reshape(-1, 2)
            cls_tgt = cls_tgt.flatten()
            
            loss_ce = F.cross_entropy(predict_cls, cls_tgt, self.empty_weight)
            
            predict_xy = predict_xy[torch.arange(batch_size), pre_index]

            loss_aux_l1 = F.l1_loss(predict_xy, targets, reduction='none').sum() / predict_xy.size(0) * self.cost_bbox
            
            _targets = torchvision.ops.box_convert(targets, in_fmt="cxcywh", out_fmt="xyxy")
            predict_xy = torchvision.ops.box_convert(predict_xy, in_fmt="cxcywh", out_fmt="xyxy")
            loss_aux_giou = (1 - torch.diag(generalized_box_iou(predict_xy, _targets))).sum() / predict_xy.size(0) * self.cost_giou
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
        
        # 分类 第一个维度为 包含目标
        cls_tgt = torch.ones(pred_cls.shape[:2], dtype=torch.long).cuda() #B,N
        cls_tgt[torch.arange(batch_size), pre_index] -= 1
        
        pred_cls = pred_cls.reshape(-1, 2)
        cls_tgt = cls_tgt.flatten()
        loss_ce = F.cross_entropy(pred_cls, cls_tgt, self.empty_weight)
    
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
        losses['loss_aux'] = self.set_aux_criterion(output) * 0.1 #torch.tensor(0.).cuda().detach() #self.set_aux_criterion(output)
        
        return losses
    
    
    def set_aux_criterion_GREC(self, output, empty, empty_box, _tgt_bbox):
        loss_dict = {}
        reg_tokens, targets = output['reg_tokens'], output['targets']
        loss_aux = torch.zeros(1).to(reg_tokens[0].device)
        sizes = [len(t["boxes"]) for t in targets]
        _tgt_bbox = torch.cat([t["boxes"] for t in targets]).detach()
        for i, rt in enumerate(reg_tokens):
            batch_size = len(targets)
            predict_xy = torch.sigmoid(self.qa.aux_heads[i](rt)).transpose(0,1) # B,N,4  
            predict_cls = self.qa.aux_cls[i](rt).transpose(0,1) # B,N,2
            
            indices = self.matcher(predict_xy, predict_cls, _tgt_bbox, sizes=sizes)

            src_bbox = []
            cls_tgt = torch.ones(predict_cls.shape[:2], dtype=torch.long).cuda() #B,N
            for i, (src, _) in enumerate(indices):
                if empty[i] != 1:
                   cls_tgt[i, src] -= 1
                src_bbox.append(predict_xy[i, src].reshape(-1,4))

            src_bbox = torch.cat(src_bbox, dim=0)

            cls_tgt = cls_tgt.flatten()
            pred_cls = predict_cls.reshape(-1, 2)        

            loss_aux_ce = F.cross_entropy(pred_cls, cls_tgt)
            
            assert len(_tgt_bbox) == len(empty_box)
            assert len(src_bbox) == len(_tgt_bbox)

            loss_aux_l1 = F.l1_loss(src_bbox, _tgt_bbox, reduction='none')
            loss_aux_l1 = torch.mean(torch.sum(loss_aux_l1, dim=-1) * (1-empty_box))
            
            tgt_bbox = torchvision.ops.box_convert(_tgt_bbox, in_fmt="cxcywh", out_fmt="xyxy")
            src_bbox = torchvision.ops.box_convert(src_bbox, in_fmt="cxcywh", out_fmt="xyxy")
            loss_aux_giou = (1 - torch.diag(generalized_box_iou(src_bbox, tgt_bbox))).sum() / batch_size

            loss_aux += loss_aux_l1 + loss_aux_giou + loss_aux_ce
            
        return loss_aux 

    def set_criterion_GREC(self, output):
        pred_box, pred_cls, targets = output['pred_box'], output['pred_cls'], output['targets']   
        batch_size = pred_box.shape[0]
        num_boxes = pred_box.size(1)

        empty = torch.tensor([1 if t['empty'] is True else 0 for t in targets]).to(pred_box.device)
        empty_box = torch.tensor([1 if t['empty'] is True else 0 for t in targets for i in range(len(t['boxes']))]).to(pred_box.device)

        tgt_bbox = torch.cat([t["boxes"] for t in targets])
        sizes = [len(t["boxes"]) for t in targets]
        
        indices = self.matcher(pred_box, pred_cls, tgt_bbox, sizes=sizes)

        src_bbox = []
        cls_tgt = torch.ones(pred_cls.shape[:2], dtype=torch.long).cuda() #B,N
        for i, (src, _) in enumerate(indices):
            if empty[i] != 1:
                cls_tgt[i, src] -= 1
     
            src_bbox.append(pred_box[i, src].reshape(-1,4))
        
        src_bbox = torch.cat(src_bbox, dim=0) 

        cls_tgt = cls_tgt.flatten()
        pred_cls = pred_cls.reshape(-1, 2)        

        loss_ce = F.cross_entropy(pred_cls, cls_tgt)
        
        try:
            assert len(tgt_bbox) == len(empty_box)
            assert len(src_bbox) == len(tgt_bbox)
        except:
            print('tgt_bbox', tgt_bbox.shape)
            print('src_bbox', src_bbox.shape)
            print('empty_box', empty_box.shape)
            exit()

        loss_bbox = F.l1_loss(src_bbox, tgt_bbox, reduction='none')
        loss_bbox = torch.mean(torch.sum(loss_bbox, dim=-1) * (1-empty_box))
        
        tgt_bbox_new = torchvision.ops.box_convert(tgt_bbox, in_fmt="cxcywh", out_fmt="xyxy")
        src_bbox = torchvision.ops.box_convert(src_bbox, in_fmt="cxcywh", out_fmt="xyxy")
        loss_giou = 1 - torch.diag(generalized_box_iou(src_bbox, tgt_bbox_new))

        losses = {}
        losses['loss_bbox'] = loss_bbox * self.cost_bbox
        losses['loss_giou'] = loss_giou.sum() / batch_size * self.cost_giou
        losses['loss_ce'] = loss_ce
        # losses['loss_giou'] = losses['loss_bbox'] = torch.tensor(0.).cuda().detach()
       
        losses['loss_aux'] = self.set_aux_criterion_GREC(output, empty, empty_box, tgt_bbox)* 0.1 #torch.tensor(0.).cuda().detach() #self.set_aux_criterion(output)
        
        return losses

    def forward(self, image, text_ids, text_mask, targets=None, idx=None, text=None, epoch=None, training=False, output_all=False, vit_type='clip'):
        if vit_type == 'swin':
            return self.forward_swin(image, text_ids, text_mask, targets, idx, text, epoch, training, output_all)
        else:
            return self.forward_clip(image, text_ids, text_mask, targets, idx, text, epoch, training, output_all)


    def forward_swin(self, image, text_ids, text_mask, targets=None, idx=None, text=None, epoch=None, training=False, output_all=False):
        '''
            vis:list
                torch.Size([32, 768, 16, 16])
                torch.Size([32, 768, 16, 16])
                torch.Size([32, 512, 16, 16])
        
        '''     
        # padding mask used in decoder
        # pad_mask = torch.zeros_like(text_ids).masked_fill_(text_ids == 0, 1).bool()
        
        vis, word, state, reg_tokens, feat_ls, attn_ls = self.qa(image, text_ids, text_mask)
        
        # tgt = self.tgt_proj(reg_tokens[-1])
        bs = state.size(0)
        reg = self.reg_proj(reg_tokens[-1]) 
        tgt = torch.zeros(reg.shape).cuda()

        fq = self.neck(vis, state)
        
        # output  reg_token: torch.Size([3, bs, 512]) vis: torch.Size([400, bs, 512])
        reg_token, vis, attn = self.decoder(reg+tgt, fq, word, pad_mask=~(text_mask.bool()), query_pos=None, return_attn=True) 

        pred_box = self.bbox_embed(reg_token).sigmoid() # torch.Size([32, 4])
        pred_cls = self.cls_embed(reg_token)
        
        # reg_token = reg_tokens[-1]
        # pred_box = self.qa.aux_heads[-1](reg_token.squeeze(0)).sigmoid()  # B,4 
        pred_box = pred_box.transpose(0,1)
        pred_cls = pred_cls.transpose(0,1)
        output = {'pred_box': pred_box, 'pred_cls': pred_cls}
        
        if training:
            output.update(dict(reg_tokens=reg_tokens))
            output.update(dict(targets=targets))
            if output_all is True:
                losses = self.set_criterion_GREC(output)
            else:
                losses = self.set_criterion(output)
            return losses
        elif output_all is True:
         
            output['pred_cls'] = pred_cls.softmax(-1)[:,:,0]
            return output
        
        else:
            index = torch.argmax(pred_cls.softmax(-1)[:,:,0], dim=-1)
            pred_box = pred_box[torch.arange(len(index)), index]
            output['pred_box'] = pred_box
 
        return output



    def forward_clip(self, image, text_ids, text_mask, targets=None, idx=None, text=None, epoch=None, training=False, output_all=False):
        '''
            vis:list
                torch.Size([32, 768, 16, 16])
                torch.Size([32, 768, 16, 16])
                torch.Size([32, 512, 16, 16])
        
        '''     
        # padding mask used in decoder
        pad_mask = torch.zeros_like(text_ids).masked_fill_(text_ids == 0, 1).bool()
        
        vis, word, state, reg_tokens, feat_ls, attn_ls = self.qa(image, text_ids, self.backbone)
        
        # tgt = self.tgt_proj(reg_tokens[-1])
        bs = state.size(0)
        reg = self.reg_proj(reg_tokens[-1]) 
        tgt = torch.zeros(reg.shape).cuda()

        fq = self.neck(vis, state)
        
        # output  reg_token: torch.Size([3, bs, 512]) vis: torch.Size([400, bs, 512])
        reg_token, vis, attn = self.decoder(reg+tgt, fq, word, pad_mask=pad_mask, query_pos=None, return_attn=True) 

        pred_box = self.bbox_embed(reg_token).sigmoid() # torch.Size([32, 4])
        pred_cls = self.cls_embed(reg_token)
        
        pred_box = pred_box.transpose(0,1)
        pred_cls = pred_cls.transpose(0,1)
        output = {'pred_box': pred_box, 'pred_cls': pred_cls}
        
        if training:
            output.update(dict(reg_tokens=reg_tokens))
            output.update(dict(targets=targets))
            if output_all is True:
                losses = self.set_criterion_GREC(output)
            else:
                losses = self.set_criterion(output)
            return losses
        elif output_all is True:
         
            output['pred_cls'] = pred_cls.softmax(-1)[:,:,0]
            return output
        
        else:
            index = torch.argmax(pred_cls.softmax(-1)[:,:,0], dim=-1)
            pred_box = pred_box[torch.arange(len(index)), index]
            output['pred_box'] = pred_box
 
        return output


