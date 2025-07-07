import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional

from .layers_4 import conv_layer, deconv_layer, InteractorT, Interactor, MLP, CrossAttention, SelfAttention, BridgerFormer





class Bridger_ViT_1(nn.Module):
    def __init__(self,
                 d_img = [768, 768, 768],
                 d_txt = 512,
                 d_model = 64,
                 nhead = 8,
                 num_stages = 3,
                 strides = [1, 1, 1],
                 num_layers = 12,
                 shared_weights = False,
                 num_reg=1,
                 num_patches=500
                ):
        super().__init__()
        self.d_img = d_img
        self.d_txt = d_txt
        self.d_model = d_model
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.n_reg = num_reg
        self.reg_token = nn.Parameter(torch.zeros(self.n_reg, 1, d_model))
        
        # self.reg_pos = nn.Embedding(self.n_reg, d_model)

        self.linear1_t, self.linear2_t = nn.ModuleList(), nn.ModuleList()
        self.linear1_v, self.linear2_v = nn.ModuleList(), nn.ModuleList()
        self.bridgeformer = nn.ModuleList()
        
        self.aux_heads = nn.ModuleList()
        
        for i in range(num_stages):
            self.bridgeformer.append(BridgerFormer(d_model, nhead))
            self.linear1_t.append(MLP(d_txt, int(d_txt/2), d_model, 2))  
            self.linear2_t.append(MLP(d_model, int(d_txt/2), d_txt, 2))    

            self.linear1_v.append(MLP(d_img[i], int(d_img[i]/2), d_model, 2))    
            self.linear2_v.append(MLP(d_model, int(d_img[i]/2), d_img[i], 2))  
            
            self.aux_heads.append(MLP(d_model, d_txt, 4, 3))
           
        self.initialize_parameters()

    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')                
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')   


    def forward(self, img, text, backbone):
        # vision
        img = img.type(backbone.dtype)
        vis_enc = backbone.visual
        vis = vis_enc.conv1(img)  # shape = [*, width, grid, grid]
        vis = vis.reshape(vis.shape[0], vis.shape[1], -1)  # shape = [*, width, grid ** 2]
        vis = vis.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        vis = torch.cat([
            vis_enc.class_embedding.to(vis.dtype) + torch.zeros(vis.shape[0], 1, vis.shape[-1], 
                dtype=vis.dtype, device=vis.device), vis], dim=1)  # shape = [*, grid ** 2 + 1, width]
    
        vis = vis + vis_enc.positional_embedding.to(vis.dtype)
        vis = vis_enc.ln_pre(vis)
        vis = vis.permute(1, 0, 2)  # NLD -> LND

        # language
        txt = backbone.token_embedding(text).type(
            backbone.dtype)  # [batch_size, n_ctx, d_model]

        txt_enc = backbone.transformer
        txt = txt + backbone.positional_embedding.type(backbone.dtype)[:txt.size(1)]
        txt = txt.permute(1, 0, 2)  # NLD -> LND

        # fusion
        stage_i = 0
        vis_outs = []
        reg_tokens = []
        # reg_token = None
        bs = txt.size(1)  
        tgt = self.reg_token.expand(-1,bs,-1) #torch.zeros(self.n_reg, bs, self.d_model).cuda()
        # reg_token = self.reg_token.expand(-1, bs, -1)
        # tgt = torch.ones(self.n_reg, bs, self.d_model).cuda()
        attn_ls = []
        _t = _v = None
        for i in range(self.num_layers):
            if (i+1)%4 != 0:
                vis = vis_enc.transformer.resblocks[i](vis)
                txt = txt_enc.resblocks[i](txt)
            else:
                vis = vis_enc.transformer.resblocks[i](vis)
                txt = txt_enc.resblocks[i](txt)                 
                # residual operation
                v = vis.clone()
                t = txt.clone() # (N, B, D)    
                v = v[1:, :, :] # N, B, D
                
                if _v is not None:
                    v +=  _v
                    t += _t
                
                v = self.linear1_v[stage_i](v)
                t = self.linear1_t[stage_i](t)
                tgt, v, t, attn = self.bridgeformer[stage_i](tgt, v, t, pos_query=None, return_attn=True)
                v = self.linear2_v[stage_i](v)
                t = self.linear2_t[stage_i](t)
                
                _v = v.clone()
                _t = t.clone()
                
                reg_tokens.append(tgt)  # (1,B,D)
                attn_ls.append(attn)
        
                vis[1:, :, :] = v + vis[1:, :, :]
                txt = txt + t 
                stage_i += 1
                if stage_i < self.num_stages:
                    vis_out = vis[1:, :, :].permute(1, 2, 0) # B, D, N
                    B, C, N = vis_out.shape
                    H = int(N ** 0.5)
                    W = N // H
                    vis_out = vis_out.reshape(B, C, H, W) # B, D, H, W
                    vis_outs.append(vis_out)                      

        # After fusion
        # vision
        # 197, 64, 768 -> 64, 197, 768
        vis = vis.permute(1, 0, 2)  # LND -> NLD

        # x = vis_enc.ln_post(x[:, 0, :])
        # 64, 197, 768 -> 64, 196, 768
        vis = vis_enc.ln_post(vis[:, 1:, :])

        if vis_enc.proj is not None:
            vis = vis @ vis_enc.proj

        # 64, 196, 512 -> 64, 512, 196
        B, N, C = vis.shape
        H = int(N ** 0.5)
        W = N // H        
        vis = vis.permute(0, 2, 1).reshape(B, C, H, W) # B, N, D -> B, D, N -> B, D, H, W
        vis_outs.append(vis) 

        # language
        txt = txt.permute(1, 0, 2)  # LND -> NLD
        txt = backbone.ln_final(txt).type(backbone.dtype) 

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        state = txt[torch.arange(txt.shape[0]),
                  text.argmax(dim=-1)] @ backbone.text_projection

        # forward
        output = vis_outs, txt, state, reg_tokens, attn_ls
  
        return output






