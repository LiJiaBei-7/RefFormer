import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional
from .layers_3 import CrossAttention, _get_clones, _get_activation_fn
import math



class FFN(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0.1):
        super(FFN, self).__init__()
        self.dense = nn.Linear(dim_in, dim_out) #768 * 768 全连接
        self.LayerNorm = nn.LayerNorm(dim_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    

class Qformer_4(nn.Module):
    def __init__(self, hidden_size, nhead=8, selfattn_layer=1, crossattn_layer=1, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(hidden_size[1], hidden_size[0])
        self.crossattn = CrossAttention(d_model=hidden_size[0], nhead=nhead, num_layers=1)
        self.selfattn = CrossAttention(d_model=hidden_size[0], nhead=nhead, num_layers=1)
    def forward(self, queries, vis, txt, query_pos=None, return_attn=False):
        vis = self.fc(vis)
        n_q = queries.size(0)
        tgt = torch.cat([queries, vis], dim=0)
        pos = torch.zeros(tgt.shape)
        tgt = self.crossattn(tgt, txt)
        tgt, attn = self.selfattn(tgt, tgt[n_q:], query_pos=pos, return_attn=True)
        queries, vis = tgt[:n_q], tgt[n_q:]

        if return_attn:
            return queries, vis, attn
        else:
            return queries, vis
    




# 将视觉特征映射为512 和文本进行selfattn的时候，k&value不使用reg
class Qformer_3(nn.Module):
    def __init__(self, hidden_size, nhead=8, selfattn_layer=1, crossattn_layer=1, dropout=0.1):
        super().__init__()
        
        self.selfattn = nn.MultiheadAttention(hidden_size[0], nhead, dropout=dropout)  
        self.layers = _get_clones(self.selfattn, selfattn_layer)
        self.fc = nn.Linear(hidden_size[1], hidden_size[0])
        # self.self = nn.ModuleList(_self for i in range(selfattn_layer))
        self.cross = CrossAttention(d_model=hidden_size[0], nhead=nhead, num_layers=crossattn_layer)
        self.num_queries = 1

    def forward(self, queries, vis, txt, query_pos=None, return_attn=False):
        '''
            vis: N_v, B, D
            txt: N_t, B, D
            queries: N_q, B, D
        '''
        n_q = len(queries)

        tgt = torch.cat((queries, txt), dim=0)
        for layer in self.layers:
            out = layer(tgt, tgt[self.num_queries:], value=tgt[self.num_queries:])[0]
            tgt = out + tgt
        tgt2 = tgt[:n_q]
        
        vis = self.fc(vis)
        src, attn = self.cross(tgt2, vis, query_pos=query_pos, return_attn=return_attn)

        return src if not return_attn else src,attn



# 将视觉特征映射为512 引入condition
class Qformer_2(nn.Module):
    def __init__(self, hidden_size, nhead=8, selfattn_layer=1, crossattn_layer=1, dropout=0.1):
        super(Qformer_2, self).__init__()
        
        self.selfattn = nn.MultiheadAttention(hidden_size[0], nhead, dropout=dropout)  
        self.layers = _get_clones(self.selfattn, selfattn_layer)
        self.fc = nn.Linear(hidden_size[1], hidden_size[0])
        # self.self = nn.ModuleList(_self for i in range(selfattn_layer))
        self.cross = CrossAttention(d_model=hidden_size[0], nhead=nhead, num_layers=crossattn_layer)

    def forward(self, condition, queries, vis, txt, query_pos=None, return_attn=False):
        '''
            vis: N_v, B, D
            txt: N_t, B, D
            queries: N_q, B, D
        '''
        n_q = len(queries)

        for layer in self.layers:
            out = layer(condition, txt, value=txt)[0]
            condition = out + condition

        tgt = condition + queries
        vis = self.fc(vis)
        src, attn = self.cross(tgt, vis, query_pos=query_pos, return_attn=return_attn)

        return src if not return_attn else src,attn


# 将视觉特征映射为512
class Qformer_1(nn.Module):
    def __init__(self, hidden_size, nhead=8, selfattn_layer=1, crossattn_layer=1, dropout=0.1):
        super().__init__()
        
        self.selfattn = nn.MultiheadAttention(hidden_size[0], nhead, dropout=dropout)  
        self.layers = _get_clones(self.selfattn, selfattn_layer)
        self.fc = nn.Linear(hidden_size[1], hidden_size[0])
        # self.self = nn.ModuleList(_self for i in range(selfattn_layer))
        self.cross = CrossAttention(d_model=hidden_size[0], nhead=nhead, num_layers=crossattn_layer)

    def forward(self, queries, vis, txt, query_pos=None, return_attn=False):
        '''
            vis: N_v, B, D
            txt: N_t, B, D
            queries: N_q, B, D
        '''
        n_q = len(queries)

        tgt = torch.cat((queries, txt), dim=0)
        for layer in self.layers:
            out = layer(tgt, tgt, value=tgt)[0]
            tgt = out + tgt
        tgt2 = tgt[:n_q]
 
        vis = self.fc(vis)
        src, attn = self.cross(tgt2, vis, query_pos=query_pos, return_attn=return_attn)

        return src if not return_attn else src,attn


# 将视觉特征映射为512
class Qformer_1_2(nn.Module):
    def __init__(self, hidden_size, nhead=8, selfattn_layer=1, crossattn_layer=1, dropout=0.1):
        super().__init__()
        
        self.selfattn = nn.MultiheadAttention(hidden_size[1], nhead, dropout=dropout)  
        self.layers = _get_clones(self.selfattn, selfattn_layer)
        self.fc = nn.Linear(hidden_size[0], hidden_size[1])
        # self.self = nn.ModuleList(_self for i in range(selfattn_layer))
        self.cross = CrossAttention(d_model=hidden_size[1], nhead=nhead, num_layers=crossattn_layer)

    def forward(self, queries, vis, txt, query_pos=None, return_attn=False):
        '''
            vis: N_v, B, D
            txt: N_t, B, D
            queries: N_q, B, D
        '''
        n_q = len(queries)
        txt = self.fc(txt)
        tgt = torch.cat((queries, txt), dim=0)
        for layer in self.layers:
            out = layer(tgt, tgt, value=tgt)[0]
            tgt = out + tgt
        tgt2 = tgt[:n_q]
 
        src, attn = self.cross(tgt2, vis, query_pos=query_pos, return_attn=return_attn)

        return src if not return_attn else src,attn


# 将视觉特征映射为512 不是用txt
class Qformer_1_1(nn.Module):
    def __init__(self, hidden_size, nhead=8, selfattn_layer=1, crossattn_layer=1, dropout=0.1):
        super().__init__()
        
        self.selfattn = nn.MultiheadAttention(hidden_size[0], nhead, dropout=dropout)  
        self.layers = _get_clones(self.selfattn, selfattn_layer)
        self.fc = nn.Linear(hidden_size[1], hidden_size[0])
        # self.self = nn.ModuleList(_self for i in range(selfattn_layer))
        self.cross = CrossAttention(d_model=hidden_size[0], nhead=nhead, num_layers=crossattn_layer)

    def forward(self, queries, vis, txt, query_pos=None, return_attn=False):
        '''
            vis: N_v, B, D
            txt: N_t, B, D
            queries: N_q, B, D
        '''
        # n_q = len(queries)

        # tgt = torch.cat((queries, txt), dim=0)
        # for layer in self.layers:
        #     out = layer(tgt, tgt, value=tgt)[0]
        #     tgt = out + tgt
        # tgt2 = tgt[:n_q]
 
        vis = self.fc(vis)
        src, attn = self.cross(queries, vis, query_pos=query_pos, return_attn=return_attn)

        return src if not return_attn else src,attn

    



class Qformer(nn.Module):
    
    def __init__(self, hidden_size, nhead=8, selfattn_layer=1, crossattn_layer=1, dropout=0.1):
        super(Qformer, self).__init__()
        
        self.selfattn = nn.MultiheadAttention(hidden_size[0], nhead, dropout=dropout)  
        self.layers = _get_clones(self.selfattn, selfattn_layer)
        # self.self = nn.ModuleList(_self for i in range(selfattn_layer))
        self.cross = CrossAttention(d_model=hidden_size[1], nhead=nhead, num_layers=crossattn_layer)
        self.intermediate = BertIntermediate(hidden_size[0], hidden_size[1])
        self.output = BertIntermediate(hidden_size[1], hidden_size[0])

    def forward(self, queries, vis, txt, query_pos=None, return_attn=False):
        '''
            vis: N_v, B, D
            txt: N_t, B, D
            queries: N_q, B, D
        '''
        n_q = len(queries)

        tgt = torch.cat((queries, txt), dim=0)
        for layer in self.layers:
            out = layer(tgt, tgt, value=tgt)[0]
            tgt = out + tgt
        tgt2 = tgt[:n_q]
        tgt2 = self.intermediate(tgt2)
        src, attn = self.cross(tgt2, vis, query_pos=query_pos, return_attn=return_attn)
        src = self.output(src)

        return src if not return_attn else src,attn


