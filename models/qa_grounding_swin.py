import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional

from .layers_grounding_final import conv_layer, deconv_layer, InteractorT, Interactor, MLP, CrossAttention, SelfAttention, QFormer # NewBridgerFormer GateBridgerFormer

from bert.multimodal_bert import MultiModalBert

# from lib import multimodal_segmentation


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
            

class WrapperModel(nn.Module):
    def __init__(self, image_model, language_model) :
        super(WrapperModel, self).__init__()
        self.image_model = image_model
        self.language_model = language_model

        self.qformer = nn.ModuleList()
        self.linear1_t, self.linear2_t = nn.ModuleList(), nn.ModuleList()
        self.linear1_v, self.linear2_v = nn.ModuleList(), nn.ModuleList()
        self.aux_heads, self.aux_cls = nn.ModuleList(), nn.ModuleList()
        d_img = [128, 256, 512, 1024]
        d_txt = 768
        self.d_model, nhead = 128, 8

        self.n_reg = 3
        self.reg_token = nn.Parameter(torch.zeros(self.n_reg, 1, self.d_model))
        p = 40 
    
        for i in range(4):
            self.qformer.append(QFormer(self.d_model, nhead))
            self.linear1_t.append(MLP(d_txt, int(d_txt/2), self.d_model, 2))  
            self.linear2_t.append(MLP(self.d_model, int(d_txt/2), d_txt, 2))    

            # self.linear1_v.append(MLP(d_img[i], int(d_img[i]/2), self.d_model, 2))    
            dim = d_img[i]
            r = [4, 2, 1, 1]
            if i < 2:
                down = nn.Sequential(
                    nn.AdaptiveAvgPool2d(p),
                    nn.Conv2d(dim, dim*r[i], kernel_size=1, bias=False),
                    LayerNorm(dim*r[i]),
                    nn.Conv2d(dim*r[i], dim, kernel_size=1, bias=False),
                    nn.GELU(),
                    nn.Conv2d(dim, self.d_model, kernel_size=1, bias=False),
                )
            else:
                down = nn.Sequential(
                    nn.Conv2d(dim, dim*r[i], kernel_size=1, bias=False),
                    LayerNorm(dim*r[i]),
                    nn.Conv2d(dim*r[i], dim, kernel_size=1, bias=False),
                    nn.GELU(),
                    nn.Conv2d(dim, self.d_model, kernel_size=1, bias=False),
                )
            self.linear1_v.append(down)
            self.linear2_v.append(MLP(self.d_model, int(d_img[i]/2), d_img[i], 2))  
            
            self.aux_heads.append(MLP(self.d_model, d_txt, 4, 3))
            self.aux_cls.append(MLP(self.d_model, d_txt, 2, 3))
        
        self.proj_v = nn.Linear(d_img[-1], 512)
        self.proj_t = nn.Linear(d_txt, 512)
           

    def semantic_inference(self, mask_cls, mask_pred):       
        mask_cls = F.softmax(mask_cls, dim=1)[...,1:]
        mask_pred = mask_pred.sigmoid()      
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)        
        return semseg

    def forward(self, image, sentences, attentions): 
        reg_tokens = []
        attn_ls = []

        input_shape = image.shape[-2:]
        l_mask = attentions.unsqueeze(dim=-1)

        i0, Wh, Ww = self.image_model.forward_stem(image) # i0 torch.Size([4, 25600, 128])
        l0, extended_attention_mask = self.language_model.forward_stem(sentences, attentions) # l0 torch.Size([4, 13, 768])  extended_attention_mask torch.Size([4, 1, 1, 13])
        
        i1 = self.image_model.forward_stage1(i0, Wh, Ww)# torch.Size([4, 25600, 128])
        l1 = self.language_model.forward_stage1(l0, extended_attention_mask) # torch.Size([4, 13, 768])

        # ===================================stage 1===================================================
        x_reshape = i1.permute(0,2,1).view(i1.shape[0], i1.shape[2], Wh, Ww)
        x_size = x_reshape.size()
        x_reshape = self.linear1_v[0](x_reshape)
        x_reshape = x_reshape.flatten(-2).permute(-1,0,1)

        bs = i1.size(0)
        tgt = self.reg_token.expand(-1, bs, -1)
        reg = torch.zeros(self.n_reg, bs, self.d_model).cuda()
        l1_residual = self.linear1_t[0](l1)

        tgt, i1_residual, l1_residual, attn = self.qformer[0](tgt+reg, x_reshape, l1_residual.transpose(0,1), pos_query=None, return_attn=True)
        reg_tokens.append(tgt)
        attn_ls.append(attn)

        l1_residual = self.linear2_t[0](l1_residual)
        H = W = int(i1_residual.size(0) ** 0.5)
        i1_residual = self.linear2_v[0](i1_residual).permute(1,-1,0).view(i1_residual.shape[1], i1_residual.shape[-1], H, W)
        i1_residual = F.interpolate(i1_residual, x_size[2:], mode='bilinear', align_corners=True).flatten(-2).permute(0,-1,1)

        i1_residual = i1_residual + i1

        if self.image_model.layers[0].downsample is not None:
            x_down = self.image_model.layers[0].downsample(i1_residual, Wh, Ww) # torch.Size([4, 6400, 256])
            Wh, Ww = (Wh + 1) // 2, (Ww + 1) // 2 
        i1 = x_down
        l1 = l1_residual.transpose(0,1) + l1
        # ======================================================================================

        i2 = self.image_model.forward_stage2(i1, Wh, Ww) # torch.Size([4, 6400, 256])
        l2 = self.language_model.forward_stage2(l1, extended_attention_mask) # torch.Size([4, 13, 768])

        # ===================================stage 2===================================================
        x_reshape = i2.permute(0,2,1).view(i2.shape[0], i2.shape[2], Wh, Ww)
        x_size = x_reshape.size()
        x_reshape = self.linear1_v[1](x_reshape)
        x_reshape = x_reshape.flatten(-2).permute(-1,0,1)

        l2_residual = self.linear1_t[1](l2)

        tgt, i2_residual, l2_residual, attn = self.qformer[1](tgt+reg, x_reshape, l2_residual.transpose(0,1), pos_query=None, return_attn=True)
        reg_tokens.append(tgt)
        attn_ls.append(attn)

        l2_residual = self.linear2_t[1](l2_residual)
        H = W = int(i2_residual.size(0) ** 0.5)
        i2_residual = self.linear2_v[1](i2_residual)
        i2_residual = i2_residual.permute(1,-1,0).view(i2_residual.shape[1], i2_residual.shape[-1], H, W)
        i2_residual = F.interpolate(i2_residual, x_size[2:], mode='bilinear', align_corners=True).flatten(-2).permute(0,-1,1)

        i2_residual = i2_residual + i2

        if self.image_model.layers[1].downsample is not None:
            x_down = self.image_model.layers[1].downsample(i2_residual, Wh, Ww) # torch.Size([4, 6400, 256])
            Wh, Ww = (Wh + 1) // 2, (Ww + 1) // 2 
        i2 = x_down
        l2 = l2_residual.transpose(0,1) + l2
        # ======================================================================================

        i3 = self.image_model.forward_stage3(i2, Wh, Ww) # torch.Size([4, 1600, 512])
        l3 = self.language_model.forward_stage3(l2, extended_attention_mask) # torch.Size([4, 13, 768])
 
        # ===================================stage 3===================================================
        x_reshape = i3.permute(0,2,1).view(i3.shape[0], i3.shape[2], Wh, Ww)
        x_size = x_reshape.size()
        x_reshape = self.linear1_v[2](x_reshape)
        x_reshape = x_reshape.flatten(-2).permute(-1,0,1)

        l3_residual = self.linear1_t[2](l3)

        tgt, i3_residual, l3_residual, attn = self.qformer[2](tgt+reg, x_reshape, l3_residual.transpose(0,1), pos_query=None, return_attn=True)
        reg_tokens.append(tgt)
        attn_ls.append(attn)

        l3_residual = self.linear2_t[2](l3_residual)
        H = W = int(i3_residual.size(0) ** 0.5)
        i3_residual = self.linear2_v[2](i3_residual)
        i3_residual = i3_residual.permute(1,-1,0).view(i3_residual.shape[1], i3_residual.shape[-1], H, W)
        i3_residual = F.interpolate(i3_residual, x_size[2:], mode='bilinear', align_corners=True).flatten(-2).permute(0,-1,1)

        i3_residual = i3_residual + i3

        if self.image_model.layers[2].downsample is not None:
            x_down = self.image_model.layers[2].downsample(i3_residual, Wh, Ww) # torch.Size([4, 6400, 256])
            Wh, Ww = (Wh + 1) // 2, (Ww + 1) // 2 
        i3 = x_down
        l3 = l3_residual.transpose(0,1) + l3
        # ======================================================================================

        # i3_residual, H, W, i3_temp, Wh, Ww  = self.image_model.forward_pwam3(i3, Wh, Ww, l3, l_mask)
        # l3_residual, l3 = self.language_model.forward_pwam3(i3, l3, extended_attention_mask) 
        # i3 = i3_temp

        i4 = self.image_model.forward_stage4(i3, Wh, Ww) # torch.Size([4, 400, 1024])
        l4 = self.language_model.forward_stage4(l3, extended_attention_mask) # torch.Size([4, 13, 768])

         # ===================================stage 4===================================================
        x_reshape = i4.permute(0,2,1).view(i4.shape[0], i4.shape[2], Wh, Ww)
        x_size = x_reshape.size()
        x_reshape = self.linear1_v[3](x_reshape)
        x_reshape = x_reshape.flatten(-2).permute(-1,0,1)

        l4_residual = self.linear1_t[3](l4)

        tgt, i4_residual, l4_residual, attn = self.qformer[3](tgt+reg, x_reshape, l4_residual.transpose(0,1), pos_query=None, return_attn=True)
        reg_tokens.append(tgt)
        attn_ls.append(attn)

        l4_residual = self.linear2_t[3](l4_residual)
        H = W = int(i4_residual.size(0) ** 0.5)
        i4_residual = self.linear2_v[3](i4_residual)
        i4_residual = i4_residual.permute(1,-1,0).view(i4_residual.shape[1], i4_residual.shape[-1], H, W)
        i4_residual = F.interpolate(i4_residual, x_size[2:], mode='bilinear', align_corners=True).flatten(-2).permute(0,-1,1)

        i4_residual = i4_residual + i4

        if self.image_model.layers[3].downsample is not None:
            x_down = self.image_model.layers[3].downsample(i4_residual, Wh, Ww) # torch.Size([4, 6400, 256])
            Wh, Ww = (Wh + 1) // 2, (Ww + 1) // 2 
        i4 = x_down
        l4 = l4_residual.transpose(0,1) + l4

        l4 = self.proj_t(l4)
        i4 = self.proj_v(i4)
        # ======================================================================================

        # i4_residual, H, W, i4_temp, Wh, Ww  = self.image_model.forward_pwam4(i4, Wh, Ww, l4, l_mask)
        # l4_residual, l4 = self.language_model.forward_pwam4(i4, l4, extended_attention_mask) 
        # i4 = i4_temp

        # outputs = {}
        # outputs['s2'] = i2_residual
        # outputs['s3'] = i3_residual
        # outputs['s4'] = i4 #i4_residual
        outputs = [i2_residual, i3_residual, i4]
        

        output = outputs, l4, l4[:,0], reg_tokens, outputs, attn_ls

        return output






