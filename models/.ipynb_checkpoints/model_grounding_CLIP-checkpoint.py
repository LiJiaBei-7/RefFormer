import torch
from models import XVLMBase
import clip
import torch.nn as nn

from PIL import Image
from PIL import ImageFile

import torch.nn.functional as F

import os

from utils.misc import rescale_bboxes

import dataset.transforms as T

from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class Model(XVLMBase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=True, use_matching_loss=True, use_mlm_loss=False, use_bbox_loss=False)

        # self.num_attention_heads = self.text_encoder.config.num_attention_heads
        self.init_params = []
        self.dataset = 'refcocog'
        self.image_root = config['image_root']
        self.image_res = config['image_res']
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.topk = 30
        self.giou_threshold = 0.7
        self.mlp_hs = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 512))
        self.alpha = 0.6
        self.text_proj = nn.Linear(512, 512)
        


    def select_anchor(self, predicted_output, anchor_feats, manner='topk'):
        pred_logits = predicted_output['pred_logits']
        pred_boxes = predicted_output['pred_boxes']

        if manner == 'topk':
            pred_logits = torch.max(pred_logits, dim=-1)[0]
            _, indices = pred_logits.topk(dim=-1, k=self.topk)
            selected_boxes = torch.gather(pred_boxes, dim=1, index=indices.unsqueeze(-1).expand(-1,-1,pred_boxes.size(-1)))
            selected_anchor_feats = torch.gather(anchor_feats, dim=1, index=indices.unsqueeze(-1).expand(-1,-1,anchor_feats.size(-1)))

        return selected_boxes, selected_anchor_feats

    
    def get_crop_imgs(self, idx, bbox):
 
        crop_imgs = []
        for i, iid in enumerate(idx):
            if self.dataset in ['refcoco','refcoco+','refcocog']:
                img_path=os.path.join(self.image_root, 'COCO_train2014_%012d.jpg' % iid)
            elif self.dataset =='referit':
                img_path = os.path.join(self.image_root, '%d.jpg' % iid)
            else:
                assert NotImplementedError
        
            image = Image.open(img_path).convert('RGB')
            bboxes_scaled = rescale_bboxes(bbox[i], image.size)
            _crop_imgs = [self.normalize(image.crop(area).resize((self.image_res, self.image_res), Image.ANTIALIAS)) for area in bboxes_scaled.tolist()]
            _crop_imgs = torch.stack(_crop_imgs)
            crop_imgs.append(_crop_imgs)
        crop_imgs = torch.cat(crop_imgs)

        return crop_imgs
    

    def calculate_pseudo_targets(self, img_feats, txt_feats):
        img_feats = torch.stack(torch.split(img_feats, self.topk, dim=0))
        # bsz,topk,dim x bsz,dim, 1  --> bsz,topk,1
        targets = img_feats @ txt_feats.unsqueeze(-1)
        targets = targets.squeeze(-1)
        return targets

    
    def calculate_anchor_ref_similarity(self, anchor_feats, ref_featas):
        # anchor_feats = F.normalize(anchor_feats)
        # ref_featas = F.normalize(ref_featas)
        sim = anchor_feats @ ref_featas.unsqueeze(-1)
        return sim.squeeze(-1)
    
    def calculate_image_ref_similarity(self, image_feats, ref_feats):
        return ref_feats @ image_feats.t()

    def contrastive_loss(self, sim, pos_idx):
        labels = pos_idx / pos_idx.sum(1, keepdim=True)
        loss = -torch.sum(F.log_softmax(sim, dim=-1) * labels, dim=-1).mean()
        return loss
    
    @torch.no_grad()
    def test(self, image, text_ids, idx=None):
        predicted_output, hs, _ = self.detr(image)
        _text_feats = self.get_text_embeds(text_ids)
        text_feats = F.normalize(self.text_proj(_text_feats))
        selected_boxes, selected_anchor_feats = self.select_anchor(predicted_output, hs.squeeze(0))
        selected_anchor_feats = F.normalize(self.mlp_hs(selected_anchor_feats), dim=-1)
        sim = self.calculate_anchor_ref_similarity(selected_anchor_feats, text_feats)
        index = torch.argmax(sim, dim=-1)
        predict_box = selected_boxes[torch.arange(len(index)), index]
        return predict_box


    def forward(self, image, image_anchor, text_ids, idx=None, epoch=None):
        losses = {}

        image_anchor_feats = self.get_vision_embeds(image_anchor)
        text_feats = self.get_text_embeds(text_ids)
        image_feats = self.get_vision_embeds(image)
        image_anchor_feats = torch.stack(image_anchor_feats.split(self.topk, dim=0))

        targets = self.calculate_pseudo_targets(image_anchor_feats, _text_feats)  # (bsz, self.topk)

        # bsz,n,d
        image_anchor_feats_ = self.fusion_module(query=image_anchor_feats, key=image_feats, value=image_feats)
        sim = self.calculate_anchor_ref_similarity(image_anchor_feats_, text_feats)





        with torch.no_grad():
            predicted_output, hs, _ = self.detr(image)
            _text_feats = self.get_text_embeds(text_ids)
        text_feats = F.normalize(self.text_proj(_text_feats))
        del image, text_ids

        # get clip image features
        with torch.no_grad():
            selected_boxes, selected_anchor_feats = self.select_anchor(predicted_output, hs.squeeze(0))
            crop_imgs = self.get_crop_imgs(idx, selected_boxes.detach().cpu()).to(text_feats.device)
            crop_imgs = self.get_vision_embeds(crop_imgs)
            # clip similarity
            targets = self.calculate_pseudo_targets(crop_imgs, _text_feats)  # (bsz, self.topk)
        
        selected_anchor_feats = F.normalize(self.mlp_hs(selected_anchor_feats), dim=-1)
        sim = self.calculate_anchor_ref_similarity(selected_anchor_feats, text_feats)
        # update pseudo targets
        if epoch > 0:
            targets = self.alpha * targets + (1.-self.alpha) * sim

        p_ind= torch.argmax(targets, dim=-1)
        p_box = selected_boxes[torch.arange(len(p_ind)), p_ind]
        _p_box = p_box.unsqueeze(1)
        cost_giou = generalized_box_iou(box_cxcywh_to_xyxy(selected_boxes).reshape(-1,4), box_cxcywh_to_xyxy(_p_box).reshape(-1,4))
        cost_giou = cost_giou.reshape(len(text_feats), self.topk, -1).split(1, dim=-1)
        cost_giou = [c[i].squeeze(-1) for i, c in enumerate(cost_giou)]
        cost_giou = torch.stack(cost_giou, dim=0)   # (bsz, self.topk)
        # return column index
        # pos_ind_giou = torch.where(cost_giou > self.giou_threshold)[1]
        pos_idx = cost_giou > self.giou_threshold  # (bsz, self.topk)

        # anchor-ref constrastive loss
        losses['anchor_ref'] = self.contrastive_loss(sim, pos_idx)
        
        # postive anchor feats weighted sum
        w = torch.softmax(targets, dim=-1) * pos_idx
        image_feats = (w.unsqueeze(1) @ selected_anchor_feats).squeeze(1) # (bsz, 512)
        # image-ref contrastive loss
        losses['image_ref'] = self.get_contrastive_loss(image_feats, text_feats)

        torch.cuda.empty_cache()

        return losses