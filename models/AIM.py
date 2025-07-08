# some codes from CLIP github(https://github.com/openai/CLIP), from VideoMAE github(https://github.com/MCG-NJU/VideoMAE)
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from collections import OrderedDict
from einops import rearrange
import random
import math


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)
    
class Adapter(nn.Module):
    def __init__(self, dim, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        down_dim = int(dim * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(dim, down_dim)
        self.D_fc2 = nn.Linear(down_dim, dim)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x
    
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        if orig_type == torch.float16:
            ret = super().forward(x)
        elif orig_type == torch.float32:
            ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, 
                            kernel_size = (self.tubelet_size,  patch_size[0],patch_size[1]), 
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x

# 기존 weight load편의성을 위해 Attention이름을 유지한다.
class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        s2t_q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        s2t_q = s2t_q * self.scale
        attn = (s2t_q @ k.transpose(-2, -1))

        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
# spatial to temporal cross attention module.
class CrossAttentionS2T(nn.Module):
    def __init__(self, dim: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        # add for cross-attn
        self.num_head = n_head
        head_dim = dim // self.num_head
        self.scale = head_dim ** -0.5
        all_head_dim = head_dim * self.num_head
        scale = dim ** -0.5
        self.space_time_pos = nn.Parameter(scale * torch.randn((197 * 8, dim)))
        
        #여기에 cross attn t2s module이 들어가야 한다.
        self.s2t_q = nn.Linear(dim, all_head_dim, bias=False)
        self.s2t_q_bias = nn.Parameter(torch.zeros(all_head_dim))
        self.s2t_kv = nn.Linear(dim, all_head_dim * 2, bias=False) # 197 tokens(cls+patch) * num_frames
        self.s2t_kv_bias = nn.Parameter(torch.zeros(all_head_dim * 2))
        
        self.t2s_proj = nn.Linear(all_head_dim, dim)
        
        self.attn_mask = attn_mask
    
    def s2t_cross_attn(self, s_x, t_x): # s_x=[n (b t) d], t_x=[b n d]
        B, _, _ = t_x.shape
        s_x = rearrange(s_x, 'n (b t) d -> b (t n) d', b=B) # batch -> token
        s_x = s_x + self.space_time_pos ## sapce time position encoding
        s2t_q_bias = self.s2t_q_bias
        s2t_kv_bias = self.s2t_kv_bias
        
        s2t_q = F.linear(input=t_x, weight=self.s2t_q.weight, bias=s2t_q_bias)
        s2t_q = rearrange(s2t_q, 'b n (h d) -> b h n d', h=self.num_head)
        s2t_kv = F.linear(input=s_x, weight=self.s2t_kv.weight, bias=s2t_kv_bias)
        s2t_kv = rearrange(s2t_kv, 'b n (e h d) -> e b h n d',e=2, h=self.num_head)
        s2t_k, s2t_v = s2t_kv[0], s2t_kv[1]
        
        s2t_q = s2t_q * self.scale
        s2t_attn = (s2t_q @ s2t_k.transpose(-2, -1))
        
        s2t_attn = s2t_attn.softmax(dim=-1)
        
        t_x = (s2t_attn @ s2t_v)
        t_x = rearrange(t_x, 'b h n d -> b n (h d)')
        t_x = self.t2s_proj(t_x)
        return t_x

    def forward(self, s_x: torch.Tensor, t_x: torch.Tensor):
        return self.s2t_cross_attn(s_x, t_x)


# this codes from CLIP github(https://github.com/openai/CLIP)
class CrossAttentionT2S(nn.Module): # 이게 VMAE로 치면 blocks class다. 여기에 cross s2t_attn layer가 추가되어야 한다.
    def __init__(self, dim: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        # add for cross-attn
        self.num_head = n_head
        head_dim = dim // self.num_head
        self.scale = head_dim ** -0.5
        all_head_dim = head_dim * self.num_head
        
        #여기에 cross attn t2s module이 들어가야 한다.
        self.t2s_q = nn.Linear(dim, all_head_dim, bias=False) # 197 tokens(cls+patch) * num_frames
        self.t2s_q_bias = nn.Parameter(torch.zeros(all_head_dim))
        self.t2s_kv = nn.Linear(dim, all_head_dim * 2, bias=False)
        self.t2s_kv_bias = nn.Parameter(torch.zeros(all_head_dim * 2))
        
        self.t2s_proj = nn.Linear(all_head_dim, dim)
        
        self.attn_mask = attn_mask
    
    def t2s_cross_attn(self, s_x, t_x): # s_x=[n (b t) d], t_x=[b n d]
        B, _, _ = t_x.shape
        s_x_cls, s_x_pat = s_x[0, :, :], s_x[1:, :, :]
        s_x_pat = rearrange(s_x_pat, 'n (b t) d -> (b n) t d', b=B) # batch -> token
        t_x = rearrange(t_x, 'b (t n) d -> (b n) t d', t=8)
        t2s_q_bias = self.t2s_q_bias
        t2s_kv_bias = self.t2s_kv_bias
        
        t2s_q = F.linear(input=s_x_pat, weight=self.t2s_q.weight, bias=t2s_q_bias)
        t2s_q = rearrange(t2s_q, 'b t (h d) -> b h t d', h=self.num_head)
        t2s_kv = F.linear(input=t_x, weight=self.t2s_kv.weight, bias=t2s_kv_bias)
        t2s_kv = rearrange(t2s_kv, 'b t (e h d) -> e b h t d',e=2, h=self.num_head)
        t2s_k, t2s_v = t2s_kv[0], t2s_kv[1]
        
        t2s_q = t2s_q * self.scale
        t2s_attn = (t2s_q @ t2s_k.transpose(-2, -1))
        
        t2s_attn = t2s_attn.softmax(dim=-1)
        
        s_x_pat = (t2s_attn @ t2s_v)
        s_x_pat = rearrange(s_x_pat, 'b h n d -> b n (h d)')
        s_x_pat = self.t2s_proj(s_x_pat)
        s_x_pat = rearrange(s_x_pat,'(b n) t d -> n (b t) d', b=B)
        s_x = torch.cat([s_x_cls.unsqueeze(0), s_x_pat], dim=0)
        return s_x

    def forward(self, s_x: torch.Tensor, t_x: torch.Tensor):
        return self.t2s_cross_attn(s_x, t_x)

    
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, num_layer=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None,use_adapter=False):
        super().__init__()
        self.cross = None
        self.num_layer = num_layer
        self.num_heads = num_heads
        self.scale = 0.5
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.act = act_layer()
        self.use_adapter=use_adapter
        self.num_frames=16
        ############################ AIM MHSA ###########################
        self.clip_ln_1 = LayerNorm(dim)
        if self.use_adapter:
        #     self.time_attn = nn.MultiheadAttention(dim, num_heads)
            self.WT_Adapter = Adapter(dim, skip_connect=False)
            self.HT_Adapter = Adapter(dim, skip_connect=False)
            self.CLS_Adapter = Adapter(dim)
        self.clip_attn = nn.MultiheadAttention(dim, num_heads)
        if self.use_adapter:
            self.S_Adapter = Adapter(dim)


        
        ############################ AIM FFN ###############################
        self.clip_ln_2 = LayerNorm(dim)
        self.clip_mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(dim, dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(dim * 4, dim))
        ]))
        if self.use_adapter:
            self.S_MLP_Adapter = Adapter(dim, skip_connect=False)
        self.attn_mask = None

        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.clip_attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    def time_attention(self, x:torch.Tensor):
        return self.time_attn(x,x,x,need_weights=False,attn_mask=None)[0]
    
    def forward(self,s_x):
        n, bt, d = s_x.shape
        w=int(math.sqrt(n))
   
        if self.use_adapter: # shape[patchnum, batch, embeddim]
            # ############################ AIM TIME #############################
            # xt = rearrange(s_x, 'n (b t) d -> t (b n) d', t=self.num_frames)
            # xt = self.HT_Adapter(self.attention(self.clip_ln_1(xt)))
            # xt = rearrange(xt, 't (b n) d -> n (b t) d', n=n)
            # ##########################################################
            # s_x = s_x + self.drop_path(xt) # spatial attention connection + time attention result
            
            xt = s_x[1:,:,:]
            xt = rearrange(xt, '(w h) (b t) d -> (w t) (b h) d', t=self.num_frames, w=w)
            xwt = self.WT_Adapter(self.attention(self.clip_ln_1(xt)))
            xt = xt + self.drop_path(xwt) # skip connection original + width time attention result
            
            xt = rearrange(xt, '(w t) (b h) d -> (h t) (b w) d', t=self.num_frames, h=w)
            xht = self.HT_Adapter(self.attention(self.clip_ln_1(xt)))
            xt = xt + self.drop_path(xht) # skip connection original + width time attention result
            xt = rearrange(xt, '(h t) (b w) d -> (w h) (b t) d', h=w,w=w)
            xt = torch.cat((s_x[0,:,:].unsqueeze(0), xt),0)
            #########################################################
            s_x = xt
            
            
            # xt = s_x[1:,:,:]
            # xt = rearrange(xt, '(w h) (b t) d -> (w t) (b h) d', t=self.num_frames, w=w)
            # cls_token = rearrange(s_x[0,:,:], '(b t) d -> t b d', t=self.num_frames)
            # xwt = torch.cat((cls_token,xt),0)
            # xwt = self.WT_Adapter(self.attention(self.clip_ln_1(xwt)))
            # xt = xt + self.drop_path(xwt) # skip connection original + width time attention result
            
            # xt = rearrange(xt, '(w t) (b h) d -> (h t) (b w) d', t=self.num_frames, h=w)
            # xht = self.HT_Adapter(self.attention(self.clip_ln_1(xt)))
            # xt = xt + self.drop_path(xht) # skip connection original + width time attention result
            # xt = rearrange(xt, '(h t) (b w) d -> (w h) (b t) d', h=w,w=w)
            # xt = torch.cat((s_x[0,:,:].unsqueeze(0), xt),0)
            ##########################################################
            # s_x = xt
        
            # AIM Space MHSA
            s_x = s_x + self.S_Adapter(self.attention(self.clip_ln_1(s_x))) # original space multi head self attention
            # cls_token = rearrange(s_x[0,:,:], '(b t) d -> b t d', t=self.num_frames)
            # cls_token = cls_token.mean(1).repeat(1, self.num_frames, 1)
            # # cls_token = self.CLS_Adapter(cls_token)
            # cls_token = rearrange(cls_token, 'b t d -> (b t) d').unsqueeze(0)
            # s_x = torch.cat((cls_token,s_x[1:,:,:]),0)
            ############################ FFN Forward ##################################
            s_xn = self.clip_ln_2(s_x)
            s_x = s_x + self.clip_mlp(s_xn) + self.drop_path(self.scale * self.S_MLP_Adapter(s_xn))   
            ############################################################################
        else:
            s_x = s_x + self.attention(self.clip_ln_1(s_x))
            s_x = s_x + self.clip_mlp(self.clip_ln_2(s_x))
        return s_x
    
class STCrossTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 head_drop_rate=0.,
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 init_scale=0.,
                 all_frames=16,
                 tubelet_size=2,
                 use_mean_pooling=True,
                 composition=False,
                 pretrained_cfg = None,
                 use_adapter=False,
                 fusion_method=None,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size
        self.composition = composition
        self.use_adapter=use_adapter
        scale = embed_dim ** -0.5
        self.fusion_method=fusion_method
        self.clip_conv1 = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.clip_class_embedding = nn.Parameter(scale * torch.randn(embed_dim))
        self.clip_positional_embedding = nn.Parameter(scale * torch.randn((img_size // patch_size) ** 2 + 1, embed_dim))
        if self.use_adapter:
            self.clip_temporal_embedding = nn.Parameter(torch.zeros(1, all_frames, embed_dim))
        self.clip_ln_pre = LayerNorm(embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, num_layer=i,use_adapter=self.use_adapter)
            for i in range(depth)])
        
        self.clip_ln_post = LayerNorm(embed_dim)
        
        if self.composition:
            self.head_verb = nn.Linear(embed_dim, 97)
            self.head_verb_dropout = nn.Dropout(head_drop_rate)
            self.head_noun = nn.Linear(embed_dim, 300)
            self.head_noun_dropout = nn.Dropout(head_drop_rate)
        else:
            self.head = nn.Linear(embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)
        if self.use_adapter:
            self._init_adpater_weight()
            
        if self.composition:
            trunc_normal_(self.head_noun.weight, std=.02)
            trunc_normal_(self.head_verb.weight, std=.02)
            self.head_verb.weight.data.mul_(init_scale)
            self.head_verb.bias.data.mul_(init_scale)
            self.head_noun.weight.data.mul_(init_scale)
            self.head_noun.bias.data.mul_(init_scale)
        else:
            trunc_normal_(self.head.weight, std=.02)
            # self.head.weight.data.mul_(init_scale)
            # self.head.bias.data.mul_(init_scale)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def _init_adpater_weight(self):
        for n, m in self.blocks.named_modules():
            if 'Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)
            elif 'up' in n:
                for n2, m2 in m.named_modules():
                    if isinstance(m2, nn.Linear):
                        nn.init.constant_(m2.weight, 0)
                        nn.init.constant_(m2.bias, 0)
        

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'clip_temporal_embedding','pos_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    
    def reset_fcnorm(self):
        self.vmae_fc_norm = nn.LayerNorm(self.embed_dim)

    def forward_features(self, x):
        B = x.shape[0]
        ######################## AIM spatial path #########################
        s_x= x
        s_t = s_x.shape[2]
        s_x = rearrange(s_x, 'b c t h w -> (b t) c h w')
        s_x = self.clip_conv1(s_x) # shape = [*, embeddim, grid, grid]
        s_x = s_x.reshape(s_x.shape[0], s_x.shape[1], -1) # [*, embeddim, grid**2]
        s_x = s_x.permute(0, 2, 1) # shape[batch, patchnum, embeddim]
        s_x = torch.cat([self.clip_class_embedding.to(s_x.dtype) + torch.zeros(s_x.shape[0], 1, s_x.shape[-1], dtype=s_x.dtype, device=s_x.device), s_x], dim=1)
        s_x = s_x + self.clip_positional_embedding.to(s_x.dtype)
        n = s_x.shape[1]
        if self.use_adapter:
            s_x = rearrange(s_x, '(b t) n d -> (b n) t d', t=s_t)
            s_x = s_x + self.clip_temporal_embedding#(1,t,d)
            s_x = rearrange(s_x, '(b n) t d -> (b t) n d', n=n)
        s_x = self.clip_ln_pre(s_x)
        #####################################################################        #####################################################################        
       
       
        s_x = s_x.permute(1,0,2)
        for blk in self.blocks:
            s_x = blk(s_x)
        s_x = s_x.permute(1,0,2)
        
        s_x = rearrange(s_x, '(b t) n d -> b t n d', b=B)
        s_x = self.clip_ln_post(s_x[:,:,0,:].mean(1)) # all cls tokens avg pooling

        
        return s_x


    def forward(self, x):
        if self.composition:
            s_x = self.forward_features(x)
            noun = self.head_noun(s_x)
            verb = self.head_verb(s_x)
            return noun, verb
        else:
            x = self.forward_features(x)
            x = self.head(x)
            return x



@register_model
def original_clip(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=False,use_adapter=False,**kwargs)
    #model.default_cfg = _cfg()
    return model
@register_model
def original_clip_large(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=14, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=False,use_adapter=False,**kwargs)
    #model.default_cfg = _cfg()
    return model


@register_model
def aim_adapter_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True,use_adapter=True,**kwargs)
    #model.default_cfg = _cfg()
    return model

@register_model
def aim_adapter_vit_large_patch14_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=14, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True,use_adapter=True,**kwargs)
    #model.default_cfg = _cfg()
    return model
@register_model
def no_compo_aim_adapter_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=False,use_adapter=True,**kwargs)
    #model.default_cfg = _cfg()
    return model

@register_model
def no_compo_aim_adapter_vit_large_patch14_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=14, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=False,use_adapter=True,**kwargs)
    #model.default_cfg = _cfg()
    return model

from transformers import VideoMAEForVideoClassification, CLIPModel, ASTForAudioClassification, ASTConfig, ASTModel

class VideoCLIPWithHead(nn.Module):
    def __init__(self, num_classes=400, feature_agg='mean'):
        super().__init__()
        # load CLIP vision backbone
        model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16', use_safetensors=True)
        self.clip_vision = model.vision_model
        self.head = nn.Linear(self.clip_vision.config.hidden_size, num_classes)
        self.feature_agg = feature_agg  # 'mean' or 'sum'
    
    def get_num_layers(self):
        return len(self.clip_vision.encoder.layers)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'embeddings'}

    def forward(self, x, **kwargs):  # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        # (1) Frame 단위로 CLIP backbone 통과 (batch concat)
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # [B*T, C, H, W]
        outputs = self.clip_vision(pixel_values=x, return_dict=True)
        features = outputs.pooler_output  # [B*T, D]
        features = features.view(B, T, -1)  # [B, T, D]
        # (2) 모든 frame feature를 sum/mean pooling
        if self.feature_agg == 'mean':
            vid_feat = features.mean(dim=1)
        elif self.feature_agg == 'sum':
            vid_feat = features.sum(dim=1)
        else:
            raise NotImplementedError
        out = self.head(vid_feat)  # [B, num_classes]
        return out
    
class VideoMAEWithHead(nn.Module):
    def __init__(self, num_classes=400):
        super().__init__()
        self.model = VideoMAEForVideoClassification.from_pretrained(
            'MCG-NJU/videomae-base-finetuned-kinetics')
        self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, num_classes)
        
    def get_num_layers(self):
        return len(self.model.videomae.encoder.layer)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'embeddings'}

    def forward(self, x, **kwargs):  # x: [B, C, T, H, W]
        outputs = self.model(pixel_values=x.permute(0, 2, 1, 3, 4), return_dict=True) # [B, num_classes]
        out = outputs.logits  # [B, num_classes]
        return out

@register_model
def clip_model(pretrained=False, **kwargs):
    print('num_classes = %s' % kwargs.get('num_classes', None))
    model = VideoCLIPWithHead(num_classes=kwargs.get('num_classes', 400), feature_agg='mean')
    return model

@register_model
def videomae_v1_model(pretrained=False, **kwargs):
    model = VideoMAEWithHead(num_classes=kwargs.get('num_classes', 400))
    return model


if __name__== '__main__':
    import time
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    import numpy as np
    import torch


    seed = 4217
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    num_frames = 16
    img_size = 224
    
    # model = internvideo2_1B_patch14_224(num_classes=400).cuda().half()
    # model = aim_adapter_vit_base_patch16_224(num_classes=400).cuda().half()
    # model = clip_model(num_classes=400).cuda().half()
    model = videomae_v1_model(num_classes=400).cuda().half()
    # model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", attn_implementation="sdpa", torch_dtype=torch.float16).cuda()
    # model = ASTModel(ASTConfig(max_length=128, num_mel_bins=400)).cuda().half()
    # model = ASTModel(ASTConfig()).cuda().half()
    model.eval()
    print(model)

    dummy_input = torch.rand(1, 3, num_frames, img_size, img_size).cuda().half()
    # dummy_input = torch.rand(1, 128, 400).cuda().half()
    # dummy_input = torch.rand(1, 1024, 128).cuda().half()
    flops = FlopCountAnalysis(model, dummy_input)
    s = time.time()
    
    print(flop_count_table(flops, max_depth=1))
    print(time.time()-s)
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    
    with torch.no_grad():
    # 워밍업 (GPU 캐시 등 초기화)
        for _ in range(10):
            _ = model(dummy_input)

        torch.cuda.synchronize()  # 동기화

        # Latency 측정
        start_time = time.time()
        _ = model(dummy_input)
        torch.cuda.synchronize()  # 다시 동기화
        end_time = time.time()

    latency_ms = (end_time - start_time) * 1000  # ms 단위로 변환
    print(f"Latency: {latency_ms:.2f} ms")
    
    # 배치 크기 지정
    batch_size = 32  # 또는 원하는 숫자
    dummy_input = torch.randn(batch_size, 3, num_frames, img_size, img_size).cuda().half()

    # 워밍업
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

        torch.cuda.synchronize()

        # Throughput 측정
        repeats = 30
        start = time.time()
        for _ in range(repeats):
            _ = model(dummy_input)
        torch.cuda.synchronize()
        end = time.time()

    elapsed_time = end - start
    total_samples = batch_size * repeats
    throughput = total_samples / elapsed_time

    print(f"Throughput: {throughput:.2f} samples/sec")