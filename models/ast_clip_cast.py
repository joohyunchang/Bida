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
from timm.models.layers import Mlp as deit_Mlp
from timm.models.layers.drop import DropPath as deit_DropPath


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
    
class deit_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class deit_PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

# spatial to temporal cross attention module.
class CrossAttentionS2T(nn.Module):
    def __init__(self, dim: int, n_head: int, num_frames: int, spec_frames=1, attn_all_frame = True, 
                 audio_patch = 196, audio_only=False, attn_mask: torch.Tensor = None, time_encoding=False, spec_shape=None, video_patch=196, time_embedding_type=False, use_stpos=True):
        super().__init__()
        self.time_embedding_type = time_embedding_type if time_encoding else False
        self.use_stpos = use_stpos
        self.num_frames = num_frames//2
        self.spec_frames = spec_frames
        self.audio_patch = audio_patch
        self.spec_shape = spec_shape
        self.num_head = n_head
        self.time_encoding = time_encoding
        self.dim = dim
        if self.time_embedding_type:
            dim = dim * 2 if time_encoding else dim
        head_dim = dim // self.num_head
        self.scale = head_dim ** -0.5
        all_head_dim = head_dim * self.num_head
        self.attn_all_frame = attn_all_frame
        if not attn_all_frame:
            # self.clip_space_pos = nn.Parameter(self.scale * torch.randn((audio_patch, dim)))
            # self.vmae_space_pos = nn.Parameter(self.scale * torch.randn((196, dim)))
            self.clip_temporal_pos = nn.Parameter(self.scale * torch.randn((spec_frames, dim)))
            self.vmae_temporal_pos = nn.Parameter(self.scale * torch.randn((num_frames//2, dim))) if not audio_only else self.clip_temporal_pos
        else:
            if self.use_stpos:
                self.clip_space_pos = nn.Parameter(self.scale * torch.randn((audio_patch, dim)))
                self.vmae_space_pos = nn.Parameter(self.scale * torch.randn((video_patch, dim)))
                self.clip_temporal_pos = nn.Parameter(self.scale * torch.randn((spec_frames, dim)))
                self.vmae_temporal_pos = nn.Parameter(self.scale * torch.randn((num_frames//2, dim))) if not audio_only else self.clip_temporal_pos
            # self.clip_st_pos = nn.Parameter(self.scale * torch.randn((audio_patch * spec_frames, dim)))
            # self.vmae_st_pos = nn.Parameter(self.scale * torch.randn((video_patch * num_frames//2, dim)))
        if self.time_embedding_type:
            self.s2t_q = nn.Linear(dim, all_head_dim//2, bias=False)
            self.s2t_q_bias = nn.Parameter(torch.zeros(all_head_dim//2))
            self.s2t_kv = nn.Linear(dim, all_head_dim, bias=False) # 197 tokens(cls+patch) * num_frames
            self.s2t_kv_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.t2s_proj = nn.Linear(all_head_dim//2, dim//2) if time_encoding else nn.Linear(all_head_dim, dim)
        else:   
            self.s2t_q = nn.Linear(dim, all_head_dim, bias=False)
            self.s2t_q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.s2t_kv = nn.Linear(dim, all_head_dim*2, bias=False) # 197 tokens(cls+patch) * num_frames
            self.s2t_kv_bias = nn.Parameter(torch.zeros(all_head_dim*2))
            self.t2s_proj= nn.Linear(all_head_dim, dim)
        # self.s2t_k = nn.Linear(dim*2, all_head_dim, bias=False) # 197 tokens(cls+patch) * num_frames
        # self.s2t_k_bias = nn.Parameter(torch.zeros(all_head_dim))
        # self.s2t_v = nn.Linear(dim, all_head_dim, bias=False) # 197 tokens(cls+patch) * num_frames
        # self.s2t_v_bias = nn.Parameter(torch.zeros(all_head_dim))
        
        self.attn_mask = attn_mask
    
    def s2t_cross_attn(self, s_x, t_x, time_encodings=None, output_attentions=None): # s_x=[n (b t) d], t_x=[b (t n) d]
        B, _, _ = t_x.shape
        t = self.num_frames
        s_x_pat = s_x[2:, :, :]
        t_x_cls, t_x_pat = t_x[:1, :, :], t_x[1:, :, :]
        if not self.attn_all_frame:
            # s_x_pat = rearrange(s_x_pat, 'n b d -> b n d') # batch -> token
            # s_x_pat = s_x_pat + self.clip_space_pos
            s_x_pat = rearrange(s_x_pat, 'n (b t) d -> b n t d', t=self.spec_frames)
            s_x_pat = s_x_pat + self.clip_temporal_pos if self.use_stpos else s_x_pat
            s_x_pat = rearrange(s_x_pat, 'b n t d -> (b t) n d')
            s_x_pat = s_x_pat + rearrange(time_encodings[0],'b t n d -> (b t) n d') if self.time_encoding else s_x_pat
            if self.spec_frames != self.num_frames:
                exp = t // self.spec_frames
                s_x_pat = s_x_pat.unsqueeze(1).expand([-1 , exp, -1, -1])
                s_x_pat = rearrange(s_x_pat, 'b t n d -> (b t) n d')
            # t_x_pat = rearrange(t_x_pat, 'n (b t) d -> (b t) n d', t=t)
            # t_x_pat = t_x_pat + self.vmae_space_pos
            t_x_pat = rearrange(t_x_pat, 'n (b t) d -> b n t d', t=t)
            t_x_pat = t_x_pat + self.vmae_temporal_pos if self.use_stpos else t_x_pat
            t_x_pat = rearrange(t_x_pat, 'b n t d -> (b t) n d')
            t_x_pat = t_x_pat + rearrange(time_encodings[1],'b t n d -> (b t) n d') if self.time_encoding else t_x_pat
        else:
            # s_x_pat = rearrange(s_x_pat, 'n (b t) d -> b (n t) d', t=self.spec_frames) # batch -> token
            # s_x_pat = s_x_pat + self.clip_st_pos
            # t_x_pat = rearrange(t_x_pat, 'n (b t) d -> b (n t) d', t=t)
            # t_x_pat = t_x_pat + self.vmae_st_pos
            s_x_pat = rearrange(s_x_pat, 'n (b t) d -> b t n d', t=self.spec_frames)
            if self.time_embedding_type:
                s_x_pat = torch.cat([s_x_pat,time_encodings[0]],dim=-1) if self.time_encoding else s_x_pat
            else:
                s_x_pat = s_x_pat + time_encodings[0] if self.time_encoding else s_x_pat
            s_x_pat = s_x_pat + self.clip_space_pos if self.use_stpos else s_x_pat
            s_x_pat = rearrange(s_x_pat, 'b t n d -> b n t d')
            s_x_pat = s_x_pat + self.clip_temporal_pos if self.use_stpos else s_x_pat
            s_x_pat = rearrange(s_x_pat, 'b n t d -> b (n t) d')
            # s_x_pat = s_x_pat + self.clip_st_pos
            t_x_pat = rearrange(t_x_pat, 'n (b t) d -> b t n d', t=t)
            if self.time_embedding_type:
                t_x_pat = torch.cat([t_x_pat,time_encodings[1]],dim=-1) if self.time_encoding else t_x_pat
            else:
                t_x_pat = t_x_pat + time_encodings[1] if self.time_encoding else t_x_pat
            t_x_pat = t_x_pat + self.vmae_space_pos if self.use_stpos else t_x_pat
            t_x_pat = rearrange(t_x_pat, 'b t n d -> b n t d')
            t_x_pat = t_x_pat + self.vmae_temporal_pos if self.use_stpos else t_x_pat
            t_x_pat = rearrange(t_x_pat, 'b n t d -> b (n t) d')
            # t_x_pat = t_x_pat + self.vmae_st_pos
        
        s2t_q = F.linear(input=t_x_pat, weight=self.s2t_q.weight, bias=self.s2t_q_bias)
        s2t_q = rearrange(s2t_q, 'b n (h d) -> b h n d', h=self.num_head)
        s2t_kv = F.linear(input=s_x_pat, weight=self.s2t_kv.weight, bias=self.s2t_kv_bias)
        s2t_kv = rearrange(s2t_kv, 'b n (e h d) -> e b h n d',e=2, h=self.num_head)
        s2t_k, s2t_v = s2t_kv[0], s2t_kv[1]
        # s2t_k = F.linear(input=s_x_pat, weight=self.s2t_k.weight, bias=self.s2t_k_bias)
        # s2t_k = rearrange(s2t_k, 'b n (h d) -> b h n d', h=self.num_head)
        # s2t_v = F.linear(input=s_x_pat[:,:,:self.dim], weight=self.s2t_v.weight, bias=self.s2t_v_bias)
        # s2t_v = rearrange(s2t_v, 'b n (h d) -> b h n d', h=self.num_head)
        
        s2t_q = s2t_q * self.scale
        s2t_attn = (s2t_q @ s2t_k.transpose(-2, -1))
        
        s2t_attn = s2t_attn.softmax(dim=-1)
        
        t_x_pat = (s2t_attn @ s2t_v)
        t_x_pat = rearrange(t_x_pat, 'b h t d -> b t (h d)')
        t_x_pat = self.t2s_proj(t_x_pat)
        if not self.attn_all_frame:
            t_x_pat = rearrange(t_x_pat, '(b t) n d -> n (b t) d', t=t)
        else:
            t_x_pat = rearrange(t_x_pat, 'b (n t) d -> n (b t) d', t=t)
        t_x = torch.cat([t_x_cls, t_x_pat], dim=0)
        if output_attentions is not None:
            return (t_x, s2t_attn)
        return (t_x, )

    def forward(self, s_x: torch.Tensor, t_x: torch.Tensor, time_encodings=None, output_attentions=None):
        return self.s2t_cross_attn(s_x, t_x, time_encodings, output_attentions=output_attentions)


# this codes from CLIP github(https://github.com/openai/CLIP)
class CrossAttentionT2S(nn.Module):
    def __init__(self, dim: int, n_head: int, num_frames: int, spec_frames=1, attn_all_frame = True, 
                 audio_patch = 196, audio_only=False, attn_mask: torch.Tensor = None, time_encoding=False, spec_shape=None, video_patch=196, time_embedding_type=False, use_stpos=True):
        super().__init__()
        self.time_embedding_type = time_embedding_type if time_encoding else False
        self.use_stpos = use_stpos
        self.num_frames = num_frames//2
        self.spec_frames = spec_frames
        self.audio_patch = audio_patch
        self.spec_shape = spec_shape
        self.num_head = n_head
        self.time_encoding = time_encoding
        self.dim = dim
        if self.time_embedding_type:
            dim = dim * 2 if time_encoding else dim
        head_dim = dim // self.num_head
        self.scale = head_dim ** -0.5
        all_head_dim = head_dim * self.num_head
        self.attn_all_frame = attn_all_frame
        if not attn_all_frame:
            # self.clip_space_pos = nn.Parameter(self.scale * torch.randn((audio_patch, dim)))
            # self.vmae_space_pos = nn.Parameter(self.scale * torch.randn((196, dim)))
            self.clip_temporal_pos = nn.Parameter(self.scale * torch.randn((spec_frames, dim)))
            self.vmae_temporal_pos = nn.Parameter(self.scale * torch.randn((num_frames//2, dim))) if not audio_only else self.clip_temporal_pos
        else:
            # self.clip_st_pos = nn.Parameter(self.scale * torch.randn((audio_patch * spec_frames, dim)))
            # self.vmae_st_pos = nn.Parameter(self.scale * torch.randn((video_patch * num_frames//2, dim)))
            if self.use_stpos:
                self.clip_space_pos = nn.Parameter(self.scale * torch.randn((audio_patch, dim)))
                self.vmae_space_pos = nn.Parameter(self.scale * torch.randn((video_patch, dim)))
                self.clip_temporal_pos = nn.Parameter(self.scale * torch.randn((spec_frames, dim)))
                self.vmae_temporal_pos = nn.Parameter(self.scale * torch.randn((num_frames//2, dim))) if not audio_only else self.clip_temporal_pos
            pass
        
        if self.time_embedding_type:
            self.t2s_q = nn.Linear(dim, all_head_dim//2, bias=False) # 197 tokens(cls+patch) * num_frames
            self.t2s_q_bias = nn.Parameter(torch.zeros(all_head_dim//2))
            self.t2s_kv = nn.Linear(dim, all_head_dim, bias=False)
            self.t2s_kv_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.t2s_proj = nn.Linear(all_head_dim//2, dim//2) if time_encoding else nn.Linear(all_head_dim, dim)
        else:
            self.t2s_q = nn.Linear(dim, all_head_dim, bias=False) # 197 tokens(cls+patch) * num_frames
            self.t2s_q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.t2s_kv = nn.Linear(dim, all_head_dim*2, bias=False)
            self.t2s_kv_bias = nn.Parameter(torch.zeros(all_head_dim*2))
            self.t2s_proj = nn.Linear(all_head_dim, dim)
        # self.t2s_k = nn.Linear(dim*2, all_head_dim, bias=False)
        # self.t2s_k_bias = nn.Parameter(torch.zeros(all_head_dim))
        # self.t2s_v = nn.Linear(dim, all_head_dim, bias=False)
        # self.t2s_v_bias = nn.Parameter(torch.zeros(all_head_dim))
        
        
        self.attn_mask = attn_mask
    
    def t2s_cross_attn(self, s_x, t_x, time_encodings=None, output_attentions=None): # s_x=[n (b t) d], t_x=[b (t n) d]
        B, _, _ = t_x.shape
        t = self.num_frames
        s_x_cls, s_x_pat = s_x[:2, :, :], s_x[2:, :, :]
        t_x_pat = t_x[1:, :, :]
        if not self.attn_all_frame:
            # s_x_pat = rearrange(s_x_pat, 'n b d -> b n d') # batch -> token
            # s_x_pat = s_x_pat + self.clip_space_pos
            s_x_pat = rearrange(s_x_pat, 'n (b t) d -> b n t d', t=self.spec_frames)
            s_x_pat = s_x_pat + self.clip_temporal_pos if self.use_stpos else s_x_pat
            s_x_pat = rearrange(s_x_pat, 'b n t d -> (b t) n d')
            # print('1',s_x_pat.shape, time_encodings[0].shape, time_encodings[1].shape)
            s_x_pat = s_x_pat + rearrange(time_encodings[0],'b t n d -> (b t) n d') if self.time_encoding else s_x_pat
            # print('2',s_x_pat.shape)
            if self.spec_frames != self.num_frames:
                exp = t // self.spec_frames
                s_x_pat = s_x_pat.unsqueeze(1).expand([-1 , exp, -1, -1])
                s_x_pat = rearrange(s_x_pat, 'b t n d -> (b t) n d')
            # t_x_pat = rearrange(t_x_pat, 'n (b t) d -> (b t) n d', t=t)
            # t_x_pat = t_x_pat + self.vmae_space_pos
            t_x_pat = rearrange(t_x_pat, 'n (b t) d -> b n t d', t=t)
            t_x_pat = t_x_pat + self.vmae_temporal_pos if self.use_stpos else t_x_pat
            t_x_pat = rearrange(t_x_pat, 'b n t d -> (b t) n d')
            # print('3',t_x_pat.shape)
            t_x_pat = t_x_pat + rearrange(time_encodings[1],'b t n d -> (b t) n d') if self.time_encoding else t_x_pat
            # print('4',t_x_pat.shape)
        else:
            # s_x_pat = rearrange(s_x_pat, 'n (b t) d -> b (n t) d', t=self.spec_frames) # batch -> token
            # s_x_pat = s_x_pat + self.clip_st_pos
            # t_x_pat = rearrange(t_x_pat, 'n (b t) d -> b (n t) d', t=t)
            # t_x_pat = t_x_pat + self.vmae_st_pos
            s_x_pat = rearrange(s_x_pat, 'n (b t) d -> b t n d', t=self.spec_frames)
            if self.time_embedding_type:
                s_x_pat = torch.cat([s_x_pat,time_encodings[0]],dim=-1) if self.time_encoding else s_x_pat
            else:
                s_x_pat = s_x_pat + time_encodings[0] if self.time_encoding else s_x_pat
            s_x_pat = s_x_pat + self.clip_space_pos if self.use_stpos else s_x_pat
            s_x_pat = rearrange(s_x_pat, 'b t n d -> b n t d')
            s_x_pat = s_x_pat + self.clip_temporal_pos if self.use_stpos else s_x_pat
            s_x_pat = rearrange(s_x_pat, 'b n t d -> b (n t) d')
            # s_x_pat = s_x_pat + self.clip_st_pos
            t_x_pat = rearrange(t_x_pat, 'n (b t) d -> b t n d', t=t)
            if self.time_embedding_type:
                t_x_pat = torch.cat([t_x_pat,time_encodings[1]],dim=-1) if self.time_encoding else t_x_pat
            else:
                t_x_pat = t_x_pat + time_encodings[1] if self.time_encoding else t_x_pat
            t_x_pat = t_x_pat + self.vmae_space_pos if self.use_stpos else t_x_pat
            t_x_pat = rearrange(t_x_pat, 'b t n d -> b n t d')
            t_x_pat = t_x_pat + self.vmae_temporal_pos if self.use_stpos else t_x_pat
            t_x_pat = rearrange(t_x_pat, 'b n t d -> b (n t) d')
            # t_x_pat = t_x_pat + self.vmae_st_pos
        
        t2s_q = F.linear(input=s_x_pat, weight=self.t2s_q.weight, bias=self.t2s_q_bias)
        t2s_q = rearrange(t2s_q, 'b t (h d) -> b h t d', h=self.num_head)
        t2s_kv = F.linear(input=t_x_pat, weight=self.t2s_kv.weight, bias=self.t2s_kv_bias)
        t2s_kv = rearrange(t2s_kv, 'b t (e h d) -> e b h t d',e=2, h=self.num_head)
        t2s_k, t2s_v = t2s_kv[0], t2s_kv[1]
        # t2s_k = F.linear(input=t_x_pat, weight=self.t2s_k.weight, bias=self.t2s_k_bias)
        # t2s_k = rearrange(t2s_k, 'b t (h d) -> b h t d', h=self.num_head)
        # t2s_v = F.linear(input=t_x_pat[:,:,:self.dim], weight=self.t2s_v.weight, bias=self.t2s_v_bias)
        # t2s_v = rearrange(t2s_v, 'b t (h d) -> b h t d', h=self.num_head)

        
        t2s_q = t2s_q * self.scale
        t2s_attn = (t2s_q @ t2s_k.transpose(-2, -1))
        
        t2s_attn = t2s_attn.softmax(dim=-1)
        
        s_x_pat = (t2s_attn @ t2s_v)
        s_x_pat = rearrange(s_x_pat, 'b h n d -> b n (h d)')
        s_x_pat = self.t2s_proj(s_x_pat)
        if not self.attn_all_frame:
            if self.spec_frames != self.num_frames:
                s_x_pat = rearrange(s_x_pat, '(b t) n d -> n t b d', t=t)
                s_x_pat = s_x_pat.mean(dim=1)
            else:
                s_x_pat = rearrange(s_x_pat, 'b n d -> n b d')
        else:
            # s_x_pat = s_x_pat[:,:,:s_x_pat.shape[-1]//2] # 추가
            s_x_pat = rearrange(s_x_pat, 'b (n t) d -> n (b t) d', t=self.spec_frames)
        s_x = torch.cat([s_x_cls, s_x_pat], dim=0)
        if output_attentions is not None:
            return (s_x, t2s_attn)
        return (s_x, )

    def forward(self, s_x: torch.Tensor, t_x: torch.Tensor, time_encodings=None, output_attentions=None):
        return self.t2s_cross_attn(s_x, t_x, time_encodings, output_attentions=output_attentions)

    
class Block(nn.Module):
    def __init__(self, dim, num_heads, num_frames=16, mlp_ratio=4., down_ratio=2, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, num_layer=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None, 
                 spec_frames=1, attn_all_frame=True, CA=0, use_Adapter=True, audio_patch=196, use_AIM=False, audio_only=False,
                 late_fusion=0, use_SA=True, use_MLP=True, time_encoding=False, spec_shape=None, video_patch=196, clip_only=False, time_embedding_type=False, use_stpos=True):
        super().__init__()
        self.num_layer = num_layer
        self.num_heads = num_heads
        self.down_ratio = down_ratio
        self.scale = 0.5
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.act = act_layer()
        self.num_frames = num_frames//2
        self.CA = CA
        self.use_Adapter = use_Adapter
        self.late_fusion = late_fusion
        self.use_AIM = use_AIM
        self.use_SA, self.use_MLP = True, True
        self.time_encoding = time_encoding
        if num_layer >= late_fusion:
            self.use_Adapter = True
            self.use_SA, self.use_MLP = use_SA, use_MLP
        ###################################### MHSA code #####################################
        ############################ CLIP MHSA ###########################
        if self.use_SA:
            self.ast_norm1 = norm_layer(dim)
            self.ast_attn = deit_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            self.ast_drop_path = deit_DropPath(drop_path) if drop_path > 0. else nn.Identity()
            
            self.clip_ln_1 = LayerNorm(dim)
            self.clip_attn = nn.MultiheadAttention(dim, num_heads)
            if self.use_Adapter:
                self.S_Adapter = Adapter(dim)
        ##################################################################
        
        ############################ AIM MHSA ###########################
            if self.use_Adapter:
                if self.use_AIM:
                    self.AT_Adapter = Adapter(dim, skip_connect=False)
                    self.AS_Adapter = Adapter(dim)
                else:
                    self.T_Adapter = Adapter(dim)
        ##################################################################
        #########################################################################################
        self.after_Adapter = False
        
        # if num_layer >= 6:
        #     time_encoding = False
        if num_layer >= self.CA and num_layer >= late_fusion:
            ###################################### Cross attention ####################################
            if not self.after_Adapter:
                self.cross_s_down = nn.Linear(dim, dim//self.down_ratio)
                self.cross_t_down = nn.Linear(dim, dim//self.down_ratio)
                # self.t2s_lateral = nn.Linear(196*8,470)
                # self.s2t_lateral = nn.Linear(470,197)
                # self.cross_s_down = nn.Identity()
                # self.cross_t_down = nn.Identity()
                self.ln_s_cross = norm_layer(dim//self.down_ratio)
                self.ln_t_cross = norm_layer(dim//self.down_ratio)
                self.t2s_cross = CrossAttentionT2S(dim//self.down_ratio, num_heads, num_frames, spec_frames=spec_frames, attn_all_frame=attn_all_frame, audio_patch=audio_patch, audio_only=audio_only, time_encoding=time_encoding, spec_shape=spec_shape, video_patch=video_patch, time_embedding_type=time_embedding_type, use_stpos=use_stpos)
                self.s2t_cross = CrossAttentionS2T(dim//self.down_ratio, num_heads, num_frames, spec_frames=spec_frames, attn_all_frame=attn_all_frame, audio_patch=audio_patch, audio_only=audio_only, time_encoding=time_encoding, spec_shape=spec_shape, video_patch=video_patch, time_embedding_type=time_embedding_type, use_stpos=use_stpos)
                self.cross_s_up = nn.Linear(dim//self.down_ratio, dim)
                self.cross_t_up = nn.Linear(dim//self.down_ratio, dim)
                # self.cross_s_up = nn.Identity()
                # self.cross_t_up = nn.Identity()
            else:
                self.t2s_cross = CrossAttentionT2S(dim, num_heads, num_frames, spec_frames=spec_frames, attn_all_frame=attn_all_frame, audio_patch=audio_patch, audio_only=audio_only, time_encoding=time_encoding, spec_shape=spec_shape, video_patch=video_patch, time_embedding_type=time_embedding_type, use_stpos=use_stpos)
                self.s2t_cross = CrossAttentionS2T(dim, num_heads, num_frames, spec_frames=spec_frames, attn_all_frame=attn_all_frame, audio_patch=audio_patch, audio_only=audio_only, time_encoding=time_encoding, spec_shape=spec_shape, video_patch=video_patch, time_embedding_type=time_embedding_type, use_stpos=use_stpos)
                self.cross_s_down = nn.Linear(dim, dim//self.down_ratio)
                self.cross_t_down = nn.Linear(dim, dim//self.down_ratio)
                self.ln_s_cross = norm_layer(dim//self.down_ratio)
                self.ln_t_cross = norm_layer(dim//self.down_ratio)
                self.cross_s_up = nn.Linear(dim//self.down_ratio, dim)
                self.cross_t_up = nn.Linear(dim//self.down_ratio, dim)
                
            ###########################################################################################
        
        ###################################### FFN code #########################################
        ############################ CLIP FFN ###############################
        if self.use_MLP:
            self.ast_norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.ast_mlp = deit_Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
            
            self.clip_ln_2 = LayerNorm(dim)
            self.clip_mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(dim, dim * 4)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(dim * 4, dim))
            ]))
            if self.use_Adapter:
                self.S_MLP_Adapter = Adapter(dim, skip_connect=False)
            self.attn_mask = None
        #####################################################################
        
        ############################ AIM FFN ###############################
            if self.use_Adapter:
                self.T_MLP_Adapter = Adapter(dim, skip_connect=False)
        #######################################################################
        #########################################################################################
        

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.clip_attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self,s_x, t_x, time_encodings=None, output_attentions=None):
        B = t_x.shape[0]
        n, bt, _ = t_x.shape
        num_frames = bt//B
        
        ############################ MHSA Forward #############################
        if self.use_SA:
            if self.use_Adapter:
                # CLIP Space MHSA
                s_x = s_x + self.ast_drop_path(self.S_Adapter(self.ast_attn(self.ast_norm1(s_x))))
                
                # AIM Time MHSA
                if self.use_AIM:
                    ############################ AIM TIME #############################
                    xt = rearrange(t_x, 'n (b t) d -> t (b n) d', t=self.num_frames)
                    xt = self.AT_Adapter(self.attention(self.clip_ln_1(xt)))
                    xt = rearrange(xt, 't (b n) d -> n (b t) d', n=n)
                    t_x = t_x + self.drop_path(xt)
                    ##########################################################
                    t_x = t_x + self.AS_Adapter(self.attention(self.clip_ln_1(t_x)))
                else:
                    t_x = t_x + self.T_Adapter(self.attention(self.clip_ln_1(t_x)))
            else:
                # AIM Space MHSA
                s_x = s_x + self.ast_drop_path(self.ast_attn(self.ast_norm1(s_x)))
                # VMAE Time MHSA
                t_x = t_x + self.attention(self.clip_ln_1(t_x))
        ########################################################################
        
        if self.num_layer >= self.CA and self.num_layer >= self.late_fusion:
            ############################ Cross Forward #############################
            s_x = rearrange(s_x,'b n d -> n b d')
            if not self.after_Adapter:
                n_s_x = self.ln_s_cross(self.cross_s_down(s_x))
                n_t_x = self.ln_t_cross(self.cross_t_down(t_x))
                a_s_x = self.t2s_cross(n_s_x, n_t_x, time_encodings, output_attentions=output_attentions)
                a_t_x = self.s2t_cross(n_s_x, n_t_x, time_encodings, output_attentions=output_attentions)
                c_s_x = self.cross_s_up(self.act(a_s_x[0]))
                c_t_x = self.cross_t_up(self.act(a_t_x[0]))
                s_x = s_x + self.drop_path(c_s_x)
                t_x = t_x + self.drop_path(c_t_x)
                
                # n_s_x = rearrange(s_x, 'n b d -> b d n')
                # n_s_x = self.s2t_lateral(n_s_x)
                # n_s_x = n_s_x.unsqueeze(2).repeat(1,1,8,1)
                # n_s_x = rearrange(n_s_x, 'b d t n-> n (b t) d')
                # n_t_x = rearrange(t_x[1:,:,:], 'n (b t) d -> b d (n t)', t=8)
                # n_t_x = self.t2s_lateral(n_t_x)
                # n_t_x = rearrange(n_t_x, 'b d nt-> nt b d')
                # n_s_x = self.cross_s_down(n_s_x)
                # n_t_x = self.cross_t_down(n_t_x)
                # # print(s_x.shape, t_x.shape, n_s_x.shape, n_t_x.shape)
                # s_x = s_x + n_t_x
                # t_x = t_x + n_s_x
            else:
                a_s_x = self.t2s_cross(n_s_x, n_t_x, time_encodings, output_attentions=output_attentions)
                a_t_x = self.s2t_cross(n_s_x, n_t_x, time_encodings, output_attentions=output_attentions)
                n_s_x = self.ln_s_cross(self.cross_s_down(self.act(a_s_x[0])))
                n_t_x = self.ln_t_cross(self.cross_t_down(self.act(a_t_x[0])))
                c_s_x = self.cross_s_up(n_s_x)
                c_t_x = self.cross_t_up(n_t_x)
                s_x = s_x + self.drop_path(c_s_x)
                t_x = t_x + self.drop_path(c_t_x)
            s_x = rearrange(s_x,'n b d -> b n d')
            #########################################################################
        
        ############################ FFN Forward ##################################
        if self.use_MLP:
            s_xn = self.ast_norm2(s_x)
            t_xn = self.clip_ln_2(t_x)
            if self.use_Adapter:
                s_x = s_x + self.ast_mlp(s_xn) + self.ast_drop_path(self.scale * self.S_MLP_Adapter(s_xn))
                t_x = t_x + self.clip_mlp(t_xn) + self.drop_path(self.scale * self.T_MLP_Adapter(t_xn))
            else:
                s_x = s_x + self.ast_drop_path(self.ast_mlp(s_xn))
                t_x = t_x + self.clip_mlp(t_xn)
        ############################################################################
        if output_attentions is not None:
            return (s_x, t_x, a_s_x[1], a_t_x[1])
        return (s_x, t_x)
    
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
                 down_ratio=2,
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
                 fusion_method=None,
                 audio_enabled=False,
                 spec_frames=1,
                 attn_all_frame=True,
                 CA=0,
                 use_Adapter=True,
                 audio_patch=196,
                 use_AIM=False,
                 audio_only=False,
                 late_fusion=0,
                 use_SA=True, 
                 use_MLP=True,
                 time_encoding=False,
                 spec_shape=None,
                 audio_only_finetune=False,
                 label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024,
                 time_embedding_type=False, 
                 use_stpos=True,
                 pre_time_encoding=False,
                 split_time_mlp=False,
                 pretrained_cfg = None,
                 pretrained_cfg_overlay = None):
        super().__init__()
        self.num_classes = num_classes
        self.num_frames = all_frames
        self.spec_frames = spec_frames
        self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size
        self.down_ratio = down_ratio
        self.composition = composition
        self.audio_enabled = audio_enabled
        self.audio_patch = audio_patch
        self.spec_shape = spec_shape
        self.video_patch = (img_size // patch_size) ** 2
        
        scale = embed_dim ** -0.5
        
        ################################################################
        self.ast_patch_embed = deit_PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        num_patches = self.ast_patch_embed.num_patches
        self.ast_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.ast_dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.ast_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))
        self.ast_pos_drop = nn.Dropout(p=drop_rate)
        self.ast_norm = norm_layer(embed_dim)

        self.original_embedding_dim = self.ast_pos_embed.shape[2]
        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.ast_patch_embed.num_patches = num_patches
        new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        self.ast_patch_embed.proj = new_proj
        new_pos_embed = nn.Parameter(torch.zeros(1, self.ast_patch_embed.num_patches + 2, self.original_embedding_dim))
        self.ast_pos_embed = new_pos_embed
        trunc_normal_(self.ast_pos_embed, std=.02)
        
        # self.original_embedding_dim = self.ast_pos_embed.shape[2]
        # f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        # num_patches = f_dim * t_dim
        # self.ast_patch_embed.num_patches = num_patches
        
        # new_pos_embed = self.ast_pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
        # # if the input sequence length is larger than the original audioset (10s), then cut the positional embedding
        # if t_dim < 101:
        #     new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim/2): 50 - int(t_dim/2) + t_dim]
        # # otherwise interpolate
        # else:
        #     new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode='bilinear')
        # if f_dim < 12:
        #     new_pos_embed = new_pos_embed[:, :, 6 - int(f_dim/2): 6 - int(f_dim/2) + f_dim, :]
        # # otherwise interpolate
        # elif f_dim > 12:
        #     new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
        # new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
        # self.ast_pos_embed = nn.Parameter(torch.cat([self.ast_pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
        self.spec_shape=[f_dim, t_dim]
        self.audio_patch = self.ast_patch_embed.num_patches
        print(f'model ast patch {self.audio_patch}, spec {self.spec_shape}, astposembed_shape {self.ast_pos_embed.shape}, {self.original_embedding_dim}')
        print(f'patch test {torch.randn(1, 1, input_fdim, input_tdim).shape}, {new_proj}, {new_proj(torch.randn(1, 1, input_fdim, input_tdim)).shape}')
        ################################################################
        
        self.clip_conv1 = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.clip_class_embedding = nn.Parameter(scale * torch.randn(embed_dim))
        self.clip_positional_embedding = nn.Parameter(scale * torch.randn((img_size // patch_size) ** 2 + 1, embed_dim))
        self.clip_ln_pre = LayerNorm(embed_dim)
        # self.clip_ln_pre = nn.Identity()

        ###################################
        self.audio_only = audio_only
        self.use_AIM = use_AIM
        if audio_only:
            self.video_patch = (input_fdim * input_tdim)//(patch_size*patch_size)
            self.clip_positional_embedding = nn.Parameter(torch.zeros(1, self.video_patch+1, embed_dim))
            self.num_frames = 2
            all_frames = 2
        if self.use_AIM:
            self.clip_temporal_embedding = nn.Parameter(torch.zeros(1, all_frames//2, embed_dim))
        spec_frames = (spec_frames+1) //2
        attn_all_frame=attn_all_frame
        CA=CA
        # self.num_frames = all_frames * 2
        # CA=12
        self.pre_time_encoding = pre_time_encoding
        self.split_time_mlp = split_time_mlp
        self.use_spec_time = True
        time_mlp_layers = 3
        self.time_encoding = time_encoding if not self.pre_time_encoding else False
        if self.time_encoding or self.pre_time_encoding:
            down_dim = self.embed_dim // self.down_ratio if not self.pre_time_encoding else self.embed_dim
            self.time_mlp = nn.Sequential(
                nn.Linear(2, down_dim),
                nn.ReLU(),
                *[nn.Sequential(nn.Linear(down_dim, down_dim), nn.ReLU()) for i in range(time_mlp_layers - 1)],
                nn.LayerNorm(down_dim)
                )
            if self.split_time_mlp:
                self.audio_time_mlp = nn.Sequential(
                    nn.Linear(2, down_dim),
                    nn.ReLU(),
                    *[nn.Sequential(nn.Linear(down_dim, down_dim), nn.ReLU()) for i in range(time_mlp_layers - 1)],
                    nn.LayerNorm(down_dim)
                    )
        ###################################

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, num_frames=self.num_frames, mlp_ratio=mlp_ratio,down_ratio=self.down_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, num_layer=i, spec_frames=spec_frames, attn_all_frame=attn_all_frame, CA=CA, use_Adapter=use_Adapter,late_fusion=late_fusion, use_SA=use_SA, use_MLP=use_MLP,
                audio_patch=self.audio_patch, video_patch=self.video_patch, use_AIM=use_AIM, audio_only=audio_only, time_encoding=self.time_encoding, spec_shape=spec_shape, time_embedding_type=time_embedding_type, use_stpos=use_stpos)
            for i in range(depth)])
        
        self.audio_ln_post = LayerNorm(embed_dim)
        self.clip_ln_post = norm_layer(embed_dim)
        
        if self.composition:
            self.head_verb = nn.Linear(embed_dim, 97)
            self.head_verb_dropout = nn.Dropout(head_drop_rate)
            self.head_noun = nn.Linear(embed_dim, 300)
            self.head_noun_dropout = nn.Dropout(head_drop_rate)
            if audio_enabled is not None:
                self.noun_last_Adapter = Adapter(embed_dim, skip_connect=True)
                self.verb_last_Adapter = Adapter(embed_dim, skip_connect=True)
        else:
            self.noun_last_Adapter = Adapter(embed_dim, skip_connect=True)
            self.verb_last_Adapter = Adapter(embed_dim, skip_connect=True)
            self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
            self.head_dropout = nn.Dropout(head_drop_rate)

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)
        self._init_adpater_weight()
        
        if self.composition:
            self.head_verb.weight.data.mul_(init_scale)
            self.head_verb.bias.data.mul_(init_scale)
            self.head_noun.weight.data.mul_(init_scale)
            self.head_noun.bias.data.mul_(init_scale)
            if audio_enabled is not None:
                nn.init.constant_(self.noun_last_Adapter.D_fc2.weight, 0)
                nn.init.constant_(self.verb_last_Adapter.D_fc2.weight, 0)
        else:
            nn.init.constant_(self.noun_last_Adapter.D_fc2.weight, 0)
            nn.init.constant_(self.verb_last_Adapter.D_fc2.weight, 0)
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

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
    
    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024, fshape=16, tshape=16):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'clip_time_pos','clip_space_pos','vmae_space_pos','vmae_time_pos','pos_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    
    def forward_features(self, x, spec=None, time_encodings=None, output_attentions=None):
        B = x.shape[0]
        s_x = spec[:, :, 1::2, :, :] if spec.dim() == 5 else spec.unsqueeze(2)  # batch 3 1 time_frame bin
        # s_x = torch.randn_like(x[:, :, 1::2, :, :]) # pick even frames
        ######################## Audio spatial path #########################
        s_x = s_x[:,0,:,:,:]
        s_x = s_x.transpose(2, 3)
        
        s_B = s_x.shape[0]
        s_x = self.ast_patch_embed(s_x)
        cls_tokens = self.ast_cls_token.expand(s_B, -1, -1)
        dist_token = self.ast_dist_token.expand(s_B, -1, -1)
        s_x = torch.cat((cls_tokens, dist_token, s_x), dim=1)
        s_x = s_x + self.ast_pos_embed
        s_x = self.ast_pos_drop(s_x)
            
        #####################################################################
        
        ######################## CLIP spatial path #########################
        if not self.audio_only:
            t_x = x[:, :, 1::2, :, :]
            # t_x = x
        else:
            t_x = spec[:, :, 1::2, :, :] if spec.dim() == 5 else spec.unsqueeze(2)
        t_t = t_x.shape[2]
        t_x = rearrange(t_x, 'b c t h w -> (b t) c h w')
        t_x = self.clip_conv1(t_x) # shape = [*, embeddim, grid, grid]
        t_x = t_x.reshape(t_x.shape[0], t_x.shape[1], -1) # [*, embeddim, grid**2]
        t_x = t_x.permute(0, 2, 1) # shape[batch, patchnum, embeddim]
        t_x = torch.cat([self.clip_class_embedding.to(t_x.dtype) + torch.zeros(t_x.shape[0], 1, t_x.shape[-1], dtype=t_x.dtype, device=t_x.device), t_x], dim=1)
        t_x = t_x + self.clip_positional_embedding.to(t_x.dtype)
        t_n = t_x.shape[1]
        if self.use_AIM:
            t_x = rearrange(t_x, '(b t) n d -> (b n) t d', t=t_t)
            t_x = t_x + self.clip_temporal_embedding.to(t_x.dtype)#(1,t,d)
            t_x = rearrange(t_x, '(b n) t d -> (b t) n d', n=t_n)
        t_x = self.clip_ln_pre(t_x)
        #####################################################################
        
        t_x = t_x.permute(1,0,2)
        if self.pre_time_encoding:
            s_x[:,2:,:] = s_x[:,2:,:] + rearrange(time_encodings[0], 'b t n d -> (b t) n d')
            t_x[1:,:,:] = t_x[1:,:,:] + rearrange(time_encodings[1], 'b t n d -> n (b t) d')
        all_self_attentions = () if output_attentions is not None else None
        for blk in self.blocks:
            layer_outputs = blk(s_x, t_x, time_encodings, output_attentions=output_attentions)
            s_x, t_x = layer_outputs[0], layer_outputs[1]
            if output_attentions is not None:
                all_self_attentions = all_self_attentions + (layer_outputs[2], layer_outputs[3])
        s_x = self.ast_norm(s_x)
        s_x = (s_x[:, 0] + s_x[:, 1]) / 2

        t_x = t_x.permute(1,0,2)
        t_x = rearrange(t_x, '(b t) n d -> b t n d', b=B)
        t_x = self.clip_ln_post(t_x[:,:,0,:].mean(1)) # all patch avg pooling
        
        return s_x, t_x, all_self_attentions

    def forward(self, x, caption=None, spec=None, idx=None, output_attentions=None):
        time_encodings=None
        if self.time_encoding or self.pre_time_encoding:
            if not self.use_spec_time:
                if self.split_time_mlp:
                    audio_time_encodings = self.audio_time_mlp(idx.reshape(idx.shape[0],-1,2).to(dtype=x.dtype, device=x.device)[:,:self.spec_frames,:])
                    video_time_encodings = self.time_mlp(idx.reshape(idx.shape[0],-1,2).to(dtype=x.dtype, device=x.device)[:,self.spec_frames:,:])
                    s = audio_time_encodings.unsqueeze(2).repeat(1,1,self.audio_patch,1)
                    t = video_time_encodings.unsqueeze(2).repeat(1,1,self.video_patch,1)
                else:
                    time_encodings = self.time_mlp(idx.reshape(idx.shape[0],-1,2).to(dtype=x.dtype, device=x.device))
                    s = time_encodings[:,:self.spec_frames,:].unsqueeze(2).repeat(1,1,self.audio_patch,1)
                    t = time_encodings[:,self.spec_frames:,:].unsqueeze(2).repeat(1,1,self.video_patch,1)
            else:
                idx = idx.reshape(idx.shape[0],-1,2).to(dtype=x.dtype, device=x.device)
                # start_end = idx[:,0,:]
                # linspace = torch.linspace(0, 1, steps=self.spec_shape[1]+1).to(dtype=x.dtype, device=x.device)
                # segments = start_end[:, 0:1] + (start_end[:, 1:2] - start_end[:, 0:1]) * linspace[:-1]
                # next_segments = start_end[:, 0:1] + (start_end[:, 1:2] - start_end[:, 0:1]) * linspace[1:]
                # segments = torch.stack([segments, next_segments], dim=-1).view(idx.shape[0], self.spec_shape[1], 2)
                start_end = idx[:, 0, :]
                num_segments = self.num_frames // 2
                linspace = torch.linspace(0, 1, steps=num_segments+1).to(dtype=x.dtype, device=x.device)
                segments = start_end[:, 0:1] + (start_end[:, 1:2] - start_end[:, 0:1]) * linspace[:-1]
                next_segments = start_end[:, 0:1] + (start_end[:, 1:2] - start_end[:, 0:1]) * linspace[1:]
                segments = torch.stack([segments, next_segments], dim=-1)
                repeats = self.spec_shape[1] // num_segments + 1
                expanded_indices = torch.arange(num_segments).repeat_interleave(repeats)[:self.spec_shape[1]]
                segments = segments[:, expanded_indices, :]
                
                if self.split_time_mlp:
                    audio_time_encodings = self.audio_time_mlp(segments.to(dtype=x.dtype, device=x.device))
                    video_time_encodings = self.time_mlp(idx[:,1:,:].to(dtype=x.dtype, device=x.device))
                    s = audio_time_encodings.unsqueeze(1).unsqueeze(3).repeat(1,1,1,self.spec_shape[0],1)
                    s = rearrange(s, 'n t h w d -> n t (h w) d')
                    t = video_time_encodings.unsqueeze(2).repeat(1,1,self.video_patch,1)
                else:
                    idx = torch.concat([segments,idx[:,1:,:]],dim=1)
                    time_encodings = self.time_mlp(idx.to(dtype=x.dtype, device=x.device))
                    s = time_encodings[:,:self.spec_shape[1],:].unsqueeze(1).unsqueeze(3).repeat(1,1,1,self.spec_shape[0],1)
                    s = rearrange(s, 'n t h w d -> n t (h w) d')
                    t = time_encodings[:,self.spec_shape[1]:,:].unsqueeze(2).repeat(1,1,self.video_patch,1)
            time_encodings = [s, t]
        if self.composition:
            if self.audio_only:
                s_x, t_x, all_self_attentions = self.forward_features(spec, spec, output_attentions=output_attentions)
                x = self.noun_last_Adapter(s_x) + self.verb_last_Adapter(t_x)
                # x = self.verb_last_Adapter(t_x)
                s_x = self.head_noun_dropout(x)
                s_x = self.head_noun(s_x)
                t_x = self.head_verb_dropout(x)
                t_x = self.head_verb(t_x)
                if output_attentions is not None:
                    return s_x, t_x, all_self_attentions
                return s_x, t_x
            elif self.audio_enabled:
                s_x, t_x, all_self_attentions = self.forward_features(x, spec, time_encodings, output_attentions=output_attentions)
                x = self.noun_last_Adapter(s_x) + self.verb_last_Adapter(t_x)
                # x = self.verb_last_Adapter(t_x)
                s_x = self.head_noun_dropout(x)
                s_x = self.head_noun(s_x)
                t_x = self.head_verb_dropout(x)
                t_x = self.head_verb(t_x)
                if output_attentions is not None:
                    return s_x, t_x, all_self_attentions
                return s_x, t_x
            else:
                s_x, t_x, all_self_attentions = self.forward_features(x, spec, output_attentions=output_attentions)
                s_x = self.head_noun_dropout(s_x)
                s_x = self.head_noun(s_x)
                t_x = self.head_verb_dropout(t_x)
                t_x = self.head_verb(t_x)
                if output_attentions is not None:
                    return s_x, t_x, all_self_attentions
                return s_x, t_x
        else:
            if self.audio_only:
                s_x, t_x, all_self_attentions = self.forward_features(spec, spec, output_attentions=output_attentions)
                # x = self.noun_last_Adapter(s_x) + self.verb_last_Adapter(t_x)
                x = self.head_dropout(s_x)
                x = self.head(x)
                if output_attentions is not None:
                    return x, all_self_attentions
                return x
            elif self.audio_enabled:
                s_x, t_x, all_self_attentions = self.forward_features(x, spec, time_encodings, output_attentions=output_attentions)
                x = self.noun_last_Adapter(s_x) + self.verb_last_Adapter(t_x)
                x = self.head_dropout(x)
                x = self.head(x)
                if output_attentions is not None:
                    return x, all_self_attentions
                return x
            else:
                s_x, t_x, all_self_attentions = self.forward_features(x, output_attentions=output_attentions)
                x = (self.noun_last_Adapter(s_x) + self.verb_last_Adapter(t_x))/2
                x = self.head_dropout(x)
                x = self.head(x)
                if output_attentions is not None:
                    return x, all_self_attentions
                return x

@register_model
def single_ast_clip_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=False, audio_enabled=True, CA=0, spec_frames=1, attn_all_frame=True, **kwargs)
    return model

@register_model
def single_ast_clip_vit_base_CA12_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=False, audio_enabled=True, CA=12, spec_frames=1, attn_all_frame=True, **kwargs)
    return model

@register_model
def single_ast_clip_cat_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=False, audio_enabled=True, CA=0, spec_frames=1, attn_all_frame=False, **kwargs)
    return model

@register_model
def single_ast_clip_down4_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=False, audio_enabled=True, CA=0, spec_frames=1, attn_all_frame=True, down_ratio=4, **kwargs)
    return model

@register_model
def single_ast_clip_down1_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=False, audio_enabled=True, CA=0, spec_frames=1, attn_all_frame=True, down_ratio=1, **kwargs)
    return model

@register_model
def single_ast_only_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=False, audio_enabled=True, 
        CA=0, spec_frames=1, attn_all_frame=True, audio_only=True, **kwargs)
    return model

@register_model
def single_ast_only2_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=False, audio_enabled=True, 
        CA=12, spec_frames=1, attn_all_frame=True, audio_only=True, **kwargs)
    return model

@register_model
def single_ast_AIM_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=False, audio_enabled=True, 
        CA=0, spec_frames=1, attn_all_frame=True, use_AIM=True, **kwargs)
    return model

@register_model
def single_ast_clip_vit_base_test(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=False, audio_enabled=True, CA=0, spec_frames=1, attn_all_frame=True, down_ratio=1, **kwargs)
    return model

@register_model
def compo_single_ast_noDown_CA9_clip_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True, audio_enabled=True, 
        CA=9, spec_frames=1, attn_all_frame=True, down_ratio=2, **kwargs)
    return model

@register_model
def compo_single_ast_clip_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True, audio_enabled=True, CA=0, spec_frames=1, attn_all_frame=True, **kwargs)
    return model

@register_model
def compo_single_ast_AIM_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True, audio_enabled=True, 
        CA=0, spec_frames=1, attn_all_frame=True, use_AIM=True, **kwargs)
    return model

@register_model
def compo_single_ast_only_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True, audio_enabled=True, 
        CA=0, spec_frames=1, attn_all_frame=True, audio_only=True, **kwargs)
    return model

@register_model
def compo_single_ast_clip_down4_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True, audio_enabled=True, 
        CA=0, spec_frames=1, attn_all_frame=True, down_ratio=4, **kwargs)
    return model

@register_model
def compo_single_ast_clip_noAdap_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True, audio_enabled=True, 
        CA=0, spec_frames=1, attn_all_frame=True, down_ratio=2, use_Adapter=False, **kwargs)
    return model

# @register_model
# def compo_single_fbf_ast_clip_vit_base_patch16_224(pretrained=False, **kwargs):
#     model = STCrossTransformer(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True, audio_enabled=True, CA=0, spec_frames=1, attn_all_frame=False, **kwargs)
#     return model

@register_model
def compo_stacks_ast_clip_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True, audio_enabled=True, CA=0, spec_frames=16, attn_all_frame=False, **kwargs)
    return model

@register_model
def compo_stacks_full_ast_clip_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True, audio_enabled=True, CA=0, spec_frames=16, attn_all_frame=True, **kwargs)
    return model


@register_model
def compo_single_ast_clip_vit_late_fusion_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=15, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True, audio_enabled=True, spec_frames=1, attn_all_frame=True,
        late_fusion=12, CA=12, use_Adapter=False, **kwargs)
    return model

@register_model
def compo_single_ast_clip_vit_late_fusion2_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=15, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True, audio_enabled=True, spec_frames=1, attn_all_frame=True,
        late_fusion=12, CA=12, use_Adapter=False, use_SA=False, use_MLP=True, **kwargs)
    return model

@register_model
def compo_single_ast_clip_vit_late_fusion3_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=15, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True, audio_enabled=True, spec_frames=1, attn_all_frame=True,
        late_fusion=12, CA=12, use_Adapter=False, use_SA=False, use_MLP=False, **kwargs)
    return model