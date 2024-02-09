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
from models import clip
from models.clip.clip import tokenize


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

# spatial to text attention module.
class CrossAttentionS2Text(nn.Module):
    def __init__(self, dim: int, text_dim: int, n_head: int, num_frames: int, attn_mask: torch.Tensor = None):
        super().__init__()
        
        # add for cross-attn
        self.num_frames = num_frames//2
        self.num_head = n_head
        head_dim = text_dim // self.num_head
        self.scale = head_dim ** -0.5
        all_head_dim = head_dim * self.num_head
        attn_all_frame = True
        if not attn_all_frame:
            self.clip_space_pos = nn.Parameter(self.scale * torch.randn((196, dim)))
        else:
            self.clip_st_pos = nn.Parameter(self.scale * torch.randn((196 * num_frames//2, dim)))
        
        self.q = nn.Linear(text_dim, all_head_dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
        self.kv = nn.Linear(dim, all_head_dim * 2, bias=False) # 197 tokens(cls+patch) * num_frames
        self.kv_bias = nn.Parameter(torch.zeros(all_head_dim * 2))
        
        self.proj = nn.Linear(all_head_dim, text_dim)
    
    def s2text_cross_attn(self, s_x, text): # s_x=[n (b t) d], t_x=[b (t n) d], text=[m=77 b d]
        t = self.num_frames
        s_x_pat = s_x[1:, :, :]
        attn_all_frame = True
        if not attn_all_frame:
            s_x_pat = rearrange(s_x_pat, 'n b d -> b n d') # batch -> token
            s_x_pat = s_x_pat + self.clip_space_pos
            text = rearrange(text, 'm b d -> b m d')
            text = text.unsqueeze(1).expand([-1 , t, -1, -1])
            text = rearrange(text, 'b t m d -> (b t) m d')
        else:
            s_x_pat = rearrange(s_x_pat, 'n (b t) d -> b (n t) d', t=t) # batch -> token
            s_x_pat = s_x_pat + self.clip_st_pos
            text = rearrange(text, 'm b d -> b m d')
        
        q = F.linear(input=text, weight=self.q.weight, bias=self.q_bias)
        q = rearrange(q, 'b m (h d) -> b h m d', h=self.num_head)
        kv = F.linear(input=s_x_pat, weight=self.kv.weight, bias=self.kv_bias)
        kv = rearrange(kv, 'b n (e h d) -> e b h n d',e=2, h=self.num_head)
        k, v = kv[0], kv[1]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        attn = attn.softmax(dim=-1)
        
        text = (attn @ v)
        text = rearrange(text, 'b h m d -> b m (h d)')
        text = self.proj(text)
        if not attn_all_frame:
            text = rearrange(text, '(b t) m d -> b t m d', t=t)
            text = text.mean(dim=1)
        text = rearrange(text, 'b m d -> m b d')
        return text

    def forward(self, s_x: torch.Tensor, text: torch.Tensor):
        return self.s2text_cross_attn(s_x, text)

# temporal to text attention module.
class CrossAttentionT2Text(nn.Module):
    def __init__(self, dim: int, text_dim: int, n_head: int, num_frames: int, attn_mask: torch.Tensor = None):
        super().__init__()
        
        # add for cross-attn
        self.num_frames = num_frames//2
        self.num_head = n_head
        head_dim = text_dim // self.num_head
        self.scale = head_dim ** -0.5
        all_head_dim = head_dim * self.num_head
        attn_all_frame = True
        if not attn_all_frame:
            self.vmae_time_pos = nn.Parameter(self.scale * torch.randn((num_frames//2, dim)))
        else:
            self.vmae_st_pos = nn.Parameter(self.scale * torch.randn((196 * num_frames//2, dim)))
        
        self.q = nn.Linear(text_dim, all_head_dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
        self.kv = nn.Linear(dim, all_head_dim * 2, bias=False) # 197 tokens(cls+patch) * num_frames
        self.kv_bias = nn.Parameter(torch.zeros(all_head_dim * 2))
        
        self.proj = nn.Linear(all_head_dim, text_dim)
    
    def t2text_cross_attn(self, t_x, text): # s_x=[n (b t) d], t_x=[b (t n) d], text=[m=77 b d]
        t = self.num_frames
        n = t_x.shape[1] // t
        attn_all_frame = True
        if not attn_all_frame:
            t_x = rearrange(t_x, 'b (t n) d -> (b n) t d', t=t)
            t_x = t_x + self.vmae_time_pos
            text = rearrange(text, 'm b d -> b m d')
            text = text.unsqueeze(1).expand([-1 , n, -1, -1])
            text = rearrange(text, 'b n m d -> (b n) m d')
        else:
            t_x = t_x + self.vmae_st_pos
            text = rearrange(text, 'm b d -> b m d')
        
        q = F.linear(input=text, weight=self.q.weight, bias=self.q_bias)
        q = rearrange(q, 'b m (h d) -> b h m d', h=self.num_head)
        kv = F.linear(input=t_x, weight=self.kv.weight, bias=self.kv_bias)
        kv = rearrange(kv, 'b n (e h d) -> e b h n d',e=2, h=self.num_head)
        k, v = kv[0], kv[1]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        attn = attn.softmax(dim=-1)
        
        text = (attn @ v)
        text = rearrange(text, 'b h m d -> b m (h d)')
        text = self.proj(text)
        if not attn_all_frame:
            text = rearrange(text, '(b n) m d -> b n m d', n=n)
            text = text.mean(dim=1)
        text = rearrange(text, 'b m d -> m b d')
        return text

    def forward(self, t_x: torch.Tensor, text: torch.Tensor,):
        return self.t2text_cross_attn(t_x, text)
    
# text to spatial attention module.
class CrossAttentionText2S(nn.Module):
    def __init__(self, dim: int, text_dim: int, n_head: int, num_frames: int, attn_mask: torch.Tensor = None):
        super().__init__()
        
        # add for cross-attn
        self.num_frames = num_frames//2
        self.num_head = n_head
        head_dim = dim // self.num_head
        self.scale = head_dim ** -0.5
        all_head_dim = head_dim * self.num_head
        self.clip_st_pos = nn.Parameter(self.scale * torch.randn((196 * num_frames//2, dim)))
        
        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
        self.kv = nn.Linear(text_dim, all_head_dim * 2, bias=False) # 197 tokens(cls+patch) * num_frames
        self.kv_bias = nn.Parameter(torch.zeros(all_head_dim * 2))
        
        self.proj = nn.Linear(all_head_dim, dim)
    
    def text2s_cross_attn(self, s_x, text): # s_x=[n (b t) d], t_x=[b (t n) d], text=[m=77 b d]
        t = self.num_frames
        s_x_cls, s_x_pat = s_x[:1,:,:], s_x[1:, :, :]
        s_x_pat = rearrange(s_x_pat, 'n (b t) d -> b (n t) d', t=t) # batch -> token
        s_x_pat = s_x_pat + self.clip_st_pos
        text = rearrange(text, 'm b d -> b m d')
        
        q = F.linear(input=s_x_pat, weight=self.q.weight, bias=self.q_bias)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_head)
        kv = F.linear(input=text, weight=self.kv.weight, bias=self.kv_bias)
        kv = rearrange(kv, 'b m (e h d) -> e b h m d',e=2, h=self.num_head)
        k, v = kv[0], kv[1]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        attn = attn.softmax(dim=-1)
        
        x_pat = (attn @ v)
        x_pat = rearrange(x_pat, 'b h n d -> b n (h d)')
        x_pat = self.proj(x_pat)
        x_pat = rearrange(x_pat, 'b (n t) d -> n (b t) d', t=t)
        s_x = torch.cat([s_x_cls, x_pat], dim=0)
        return s_x

    def forward(self, s_x: torch.Tensor, text: torch.Tensor):
        return self.text2s_cross_attn(s_x, text)

# temporal to text attention module.
class CrossAttentionText2T(nn.Module):
    def __init__(self, dim: int, text_dim: int, n_head: int, num_frames: int, attn_mask: torch.Tensor = None):
        super().__init__()
        
        # add for cross-attn
        self.num_frames = num_frames//2
        self.num_head = n_head
        head_dim = dim // self.num_head
        self.scale = head_dim ** -0.5
        all_head_dim = head_dim * self.num_head
        self.vmae_st_pos = nn.Parameter(self.scale * torch.randn((196 * num_frames//2, dim)))
        
        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
        self.kv = nn.Linear(text_dim, all_head_dim * 2, bias=False) # 197 tokens(cls+patch) * num_frames
        self.kv_bias = nn.Parameter(torch.zeros(all_head_dim * 2))
        
        self.proj = nn.Linear(all_head_dim, dim)
    
    def text2t_cross_attn(self, t_x, text): # s_x=[n (b t) d], t_x=[b (t n) d], text=[m=77 b d]
        t = self.num_frames
        n = t_x.shape[1] // t
        t_x = t_x + self.vmae_st_pos
        text = rearrange(text, 'm b d -> b m d')
        
        q = F.linear(input=t_x, weight=self.q.weight, bias=self.q_bias)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_head)
        kv = F.linear(input=text, weight=self.kv.weight, bias=self.kv_bias)
        kv = rearrange(kv, 'b m (e h d) -> e b h m d',e=2, h=self.num_head)
        k, v = kv[0], kv[1]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        attn = attn.softmax(dim=-1)
        
        t_x = (attn @ v)
        t_x = rearrange(t_x, 'b h n d -> b n (h d)')
        t_x = self.proj(t_x)
        return t_x

    def forward(self, t_x: torch.Tensor, text: torch.Tensor,):
        return self.text2t_cross_attn(t_x, text)

# spatial to temporal cross attention module.
class CrossAttentionS2T(nn.Module):
    def __init__(self, dim: int, n_head: int, num_frames: int, attn_mask: torch.Tensor = None):
        super().__init__()

        # add for cross-attn
        self.num_frames = num_frames
        self.num_head = n_head
        head_dim = dim // self.num_head
        self.scale = head_dim ** -0.5
        all_head_dim = head_dim * self.num_head
        self.clip_space_pos = nn.Parameter(self.scale * torch.randn((196, dim)))
        self.vmae_space_pos = nn.Parameter(self.scale * torch.randn((196, dim)))
        

        self.s2t_q = nn.Linear(dim, all_head_dim, bias=False)
        self.s2t_q_bias = nn.Parameter(torch.zeros(all_head_dim))
        self.s2t_kv = nn.Linear(dim, all_head_dim * 2, bias=False) # 197 tokens(cls+patch) * num_frames
        self.s2t_kv_bias = nn.Parameter(torch.zeros(all_head_dim * 2))
        
        self.t2s_proj = nn.Linear(all_head_dim, dim)
        
        self.attn_mask = attn_mask
    
    def s2t_cross_attn(self, s_x, t_x): # s_x=[n (b t) d], t_x=[b (t n) d]
        B, _, _ = t_x.shape
        t = s_x.shape[1] // t_x.shape[0]
        s_x_pat = s_x[1:, :, :]
        s_x_pat = rearrange(s_x_pat, 'n b d -> b n d') # batch -> token
        s_x_pat = s_x_pat + self.clip_space_pos
        t_x = rearrange(t_x, 'b (t n) d -> (b t) n d', t=t)
        t_x = t_x + self.vmae_space_pos
        s2t_q_bias = self.s2t_q_bias
        s2t_kv_bias = self.s2t_kv_bias
        
        s2t_q = F.linear(input=t_x, weight=self.s2t_q.weight, bias=s2t_q_bias)
        s2t_q = rearrange(s2t_q, 'b n (h d) -> b h n d', h=self.num_head)
        s2t_kv = F.linear(input=s_x_pat, weight=self.s2t_kv.weight, bias=s2t_kv_bias)
        s2t_kv = rearrange(s2t_kv, 'b n (e h d) -> e b h n d',e=2, h=self.num_head)
        s2t_k, s2t_v = s2t_kv[0], s2t_kv[1]
        
        s2t_q = s2t_q * self.scale
        s2t_attn = (s2t_q @ s2t_k.transpose(-2, -1))
        
        s2t_attn = s2t_attn.softmax(dim=-1)
        
        t_x = (s2t_attn @ s2t_v)
        t_x = rearrange(t_x, 'b h t d -> b t (h d)')
        t_x = self.t2s_proj(t_x)
        t_x = rearrange(t_x, '(b t) n d -> b (t n) d', b=B)
        return t_x

    def forward(self, s_x: torch.Tensor, t_x: torch.Tensor):
        return self.s2t_cross_attn(s_x, t_x)


# this codes from CLIP github(https://github.com/openai/CLIP)
class CrossAttentionT2S(nn.Module):
    def __init__(self, dim: int, n_head: int, num_frames: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.num_frames = num_frames
        self.num_head = n_head
        head_dim = dim // self.num_head
        self.scale = head_dim ** -0.5
        all_head_dim = head_dim * self.num_head
        self.clip_time_pos = nn.Parameter(self.scale * torch.randn((num_frames//2, dim)))
        self.vmae_time_pos = nn.Parameter(self.scale * torch.randn((num_frames//2, dim)))
        
        self.t2s_q = nn.Linear(dim, all_head_dim, bias=False) # 197 tokens(cls+patch) * num_frames
        self.t2s_q_bias = nn.Parameter(torch.zeros(all_head_dim))
        self.t2s_kv = nn.Linear(dim, all_head_dim * 2, bias=False)
        self.t2s_kv_bias = nn.Parameter(torch.zeros(all_head_dim * 2))
        
        self.t2s_proj = nn.Linear(all_head_dim, dim)
        
        self.attn_mask = attn_mask
    
    def t2s_cross_attn(self, s_x, t_x): # s_x=[n (b t) d], t_x=[b n d]
        B, _, _ = t_x.shape
        t = s_x.shape[1] // t_x.shape[0]
        s_x_cls, s_x_pat = s_x[0, :, :], s_x[1:, :, :]
        s_x_pat = rearrange(s_x_pat, 'n (b t) d -> (b n) t d', b=B) # batch -> token
        s_x_pat = s_x_pat + self.clip_time_pos
        t_x = rearrange(t_x, 'b (t n) d -> (b n) t d', t=t)
        t_x = t_x + self.vmae_time_pos
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
    def __init__(self, dim, num_heads, num_frames=16, mlp_ratio=4., down_ratio=2, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, num_layer=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None,
                 text_dim=512, text_num_heads=8, use_Adapter=False):
        super().__init__()
        self.num_layer = num_layer
        self.num_heads = num_heads
        self.down_ratio = down_ratio
        self.scale = 0.5
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.act = act_layer()
        self.use_Adapter = use_Adapter
        
        ###################################### MHSA code #####################################
        ############################ AIM MHSA ###########################
        self.clip_ln_1 = LayerNorm(dim)
        self.clip_attn = nn.MultiheadAttention(dim, num_heads)
        self.S_Adapter = Adapter(dim)
        ##################################################################
        
        ############################ VMAE MHSA ###########################
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.T_Adapter = Adapter(dim)
        ##################################################################
        
        ############################ CLIP TEXT MHSA ######################
        self.clip_text_ln_1 = LayerNorm(text_dim)
        self.clip_text_attn = nn.MultiheadAttention(text_dim, text_num_heads)
        if self.use_Adapter:
            self.Text_Adapter = Adapter(text_dim)
        ##################################################################
        #########################################################################################
        
        ###################################### Cross attention ####################################
        self.cross_s_down = nn.Linear(dim, dim//self.down_ratio)
        self.cross_t_down = nn.Linear(dim, dim//self.down_ratio)
        self.cross_text_down = nn.Linear(text_dim, text_dim//self.down_ratio)
        self.ln_s_cross = norm_layer(dim//self.down_ratio)
        self.ln_t_cross = norm_layer(dim//self.down_ratio)
        self.ln_text_cross = norm_layer(text_dim//self.down_ratio)
        self.t2s_cross = CrossAttentionT2S(dim//self.down_ratio, num_heads, num_frames)
        self.s2t_cross = CrossAttentionS2T(dim//self.down_ratio, num_heads, num_frames)
        self.s2text_cross = CrossAttentionS2Text(dim//self.down_ratio, text_dim//self.down_ratio, text_num_heads, num_frames)
        self.t2text_cross = CrossAttentionT2Text(dim//self.down_ratio, text_dim//self.down_ratio, text_num_heads, num_frames)
        self.text2s_cross = CrossAttentionText2S(dim//self.down_ratio, text_dim//self.down_ratio, text_num_heads, num_frames)
        self.text2t_cross = CrossAttentionText2T(dim//self.down_ratio, text_dim//self.down_ratio, text_num_heads, num_frames)
        self.cross_s_up = nn.Linear(dim//self.down_ratio, dim)
        self.cross_t_up = nn.Linear(dim//self.down_ratio, dim)
        self.cross_text_up = nn.Linear(text_dim//self.down_ratio, text_dim)
        ###########################################################################################
        
        ###################################### FFN code #########################################
        ############################ AIM FFN ###############################
        self.clip_ln_2 = LayerNorm(dim)
        self.clip_mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(dim, dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(dim * 4, dim))
        ]))
        self.S_MLP_Adapter = Adapter(dim, skip_connect=False)
        self.attn_mask = None
        #####################################################################
        
        ############################ VMAE FFN ###############################
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.T_MLP_Adapter = Adapter(dim, skip_connect=False)
        #####################################################################
        
        ############################ CLIP TEXT FFN ##########################
        self.clip_text_ln_2 = LayerNorm(text_dim)
        self.clip_text_mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(text_dim, text_dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(text_dim * 4, text_dim))
        ]))
        if self.use_Adapter:
            self.Text_MLP_Adapter = Adapter(text_dim, skip_connect=False)
        self.attn_mask = None
        ####################################################################
        #########################################################################################
        

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.clip_attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    
    def text_attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.clip_text_attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self,s_x, t_x, text):
        B = t_x.shape[0]
        n, bt, _ = s_x.shape
        num_frames = bt//B
        
        ############################ MHSA Forward #############################
        # AIM Space MHSA
        s_x = s_x + self.S_Adapter(self.attention(self.clip_ln_1(s_x)))
        # VMAE Time MHSA
        t_x = t_x + self.T_Adapter(self.attn(self.norm1(t_x)))
        # CLIP Text MHSA
        if self.use_Adapter:
            text = text + self.Text_Adapter(self.text_attention(self.clip_text_ln_1(text)))
        else:
            text = text + self.text_attention(self.clip_text_ln_1(text))
        ########################################################################
        
        ############################ Cross Forward #############################
        n_s_x = self.ln_s_cross(self.cross_s_down(s_x))
        n_t_x = self.ln_t_cross(self.cross_t_down(t_x))
        n_text = self.ln_text_cross(self.cross_text_down(text))
        c_s_x = self.cross_s_up(self.act(self.t2s_cross(n_s_x, n_t_x) + self.text2s_cross(n_s_x, n_text)))
        c_t_x = self.cross_t_up(self.act(self.s2t_cross(n_s_x, n_t_x) + self.text2t_cross(n_t_x, n_text)))
        n_text = self.cross_text_up(self.act(self.s2text_cross(n_s_x, n_text) + self.t2text_cross(n_t_x, n_text)))
        s_x = s_x + self.drop_path(c_s_x)
        t_x = t_x + self.drop_path(c_t_x)
        text = text + self.drop_path(n_text)
        #########################################################################
        
        ############################ FFN Forward ##################################
        s_xn = self.clip_ln_2(s_x)
        s_x = s_x + self.clip_mlp(s_xn) + self.drop_path(self.scale * self.S_MLP_Adapter(s_xn))
        
        t_xn = self.norm2(t_x)
        t_x = t_x + self.mlp(t_xn) + self.drop_path(self.scale * self.T_MLP_Adapter(t_xn))
        
        text_xn = self.clip_text_ln_2(text)
        if self.use_Adapter:
            text = text + self.clip_text_mlp(text_xn) + self.drop_path(self.scale * self.Text_MLP_Adapter(text_xn))
        else:
            text = text + self.clip_text_mlp(text_xn)
        ############################################################################
        
        return s_x, t_x, text
    
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.clip_text_attn = nn.MultiheadAttention(d_model, n_head)
        self.clip_text_ln_1 = LayerNorm(d_model)
        self.clip_text_mlp = nn.Sequential(OrderedDict([
            ("D_fc1", nn.Linear(d_model, d_model )),
            ("act", nn.GELU()),
            ("D_fc2", nn.Linear(d_model * 4, d_model))
        ]))
        self.clip_text_ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.clip_text_attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.clip_text_ln_1(x))
        x = x + self.clip_text_mlp(self.clip_text_ln_2(x))
        return x

class STCrossTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dim=768,
                 text_dim=512, 
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
                 pretrained_cfg = None,
                 pretrained_cfg_overlay = None,
                 text_num_heads = 8,
                 vocab_size = 49408,
                 context_length = 77,
                 prefix = None,
                 postfix = None,
                 use_textF = True):
        super().__init__()
        self.num_classes = num_classes
        self.num_frames = all_frames
        self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size
        self.down_ratio = down_ratio
        self.composition = composition
        # ==============================================================================================================
        self.text_dim = text_dim
        self.prefix = prefix
        self.postfix = postfix
        
        self.context_length = context_length
        self.prompt_embedding = torch.nn.Embedding(context_length, self.text_dim)
        nn.init.normal_(self.prompt_embedding.weight, std=0.01)
        self.vocab_size = vocab_size
        self.clip_text_token_embedding = nn.Embedding(vocab_size, text_dim)
        self.clip_text_positional_embedding = nn.Parameter(torch.empty(self.context_length, text_dim))
        self.clip_text_ln_final = LayerNorm(text_dim)
        self.clip_text_text_projection = nn.Parameter(torch.empty(text_dim, text_dim))
        
        # self.text_blocks = nn.ModuleList([ResidualAttentionBlock(text_dim, text_num_heads) for _ in range(depth)])
        
        nn.init.normal_(self.clip_text_token_embedding.weight, std=0.02)
        nn.init.normal_(self.clip_text_positional_embedding, std=0.01)
        nn.init.normal_(self.clip_text_text_projection, std=self.text_dim ** -0.5)
        
        # abliation study 용
        use_Adapter = True
        self.split_projection = True
        self.use_videoF = False
        self.use_textF = True
        # ==============================================================================================================
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames, tubelet_size=self.tubelet_size)
        num_patches = self.patch_embed.num_patches
        
        scale = embed_dim ** -0.5
        self.clip_conv1 = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.clip_class_embedding = nn.Parameter(scale * torch.randn(embed_dim))
        self.clip_positional_embedding = nn.Parameter(scale * torch.randn((img_size // patch_size) ** 2 + 1, embed_dim))
        self.clip_ln_pre = LayerNorm(embed_dim)

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, num_frames=self.num_frames, mlp_ratio=mlp_ratio,down_ratio=self.down_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, num_layer=i, text_num_heads=text_num_heads, use_Adapter=use_Adapter)
            for i in range(depth)])
        
        self.clip_ln_post = LayerNorm(embed_dim)
        self.vmae_fc_norm = norm_layer(embed_dim)
        
        # 768 to 512
        if self.composition:
            if self.use_videoF:
                self.noun_last_Adapter = Adapter(embed_dim, skip_connect=False)
                self.verb_last_Adapter = Adapter(embed_dim, skip_connect=False)
            if self.use_textF:
                self.text_noun_last_Adapter = nn.Sequential(OrderedDict([
                    ("c_fc", nn.Linear(text_dim, text_dim // 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(text_dim // 4, embed_dim))
                ]))
                self.text_verb_last_Adapter = nn.Sequential(OrderedDict([
                    ("c_fc", nn.Linear(text_dim, text_dim // 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(text_dim // 4, embed_dim))
                ]))
            self.head_verb = nn.Linear(embed_dim, 97)
            self.head_verb_dropout = nn.Dropout(head_drop_rate)
            self.head_noun = nn.Linear(embed_dim, 300)
            self.head_noun_dropout = nn.Dropout(head_drop_rate)
            pass
        else:
            self.noun_last_Adapter = Adapter(embed_dim, skip_connect=False)
            self.verb_last_Adapter = Adapter(embed_dim, skip_connect=False)
            self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
            self.head_dropout = nn.Dropout(head_drop_rate)

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)
        self._init_adpater_weight()
        
        if self.composition:
            if self.use_videoF:
                nn.init.constant_(self.noun_last_Adapter.D_fc2.weight, 0)
                nn.init.constant_(self.verb_last_Adapter.D_fc2.weight, 0)
            self.head_verb.weight.data.mul_(init_scale)
            self.head_verb.bias.data.mul_(init_scale)
            self.head_noun.weight.data.mul_(init_scale)
            self.head_noun.bias.data.mul_(init_scale)
            pass
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

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'clip_time_pos','clip_space_pos','vmae_space_pos','vmae_time_pos','pos_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    
    def reset_fcnorm(self):
        self.vmae_fc_norm = nn.LayerNorm(self.embed_dim)

    def forward_features(self, x, caption=None, split_projection=False):
        B = x.shape[0]
        s_x = x[:, :, 1::2, :, :] # pick even frames
        ######################## AIM spatial path #########################
        s_t = s_x.shape[2]
        s_x = rearrange(s_x, 'b c t h w -> (b t) c h w')
        s_x = self.clip_conv1(s_x) # shape = [*, embeddim, grid, grid]
        s_x = s_x.reshape(s_x.shape[0], s_x.shape[1], -1) # [*, embeddim, grid**2]
        s_x = s_x.permute(0, 2, 1) # shape[batch, patchnum, embeddim]
        s_x = torch.cat([self.clip_class_embedding.to(s_x.dtype) + torch.zeros(s_x.shape[0], 1, s_x.shape[-1], dtype=s_x.dtype, device=s_x.device), s_x], dim=1)
        s_x = s_x + self.clip_positional_embedding.to(s_x.dtype)
        s_x = self.clip_ln_pre(s_x)
        #####################################################################
        
        ######################## VMAE spatial path #########################
        t_x = self.patch_embed(x)

        if self.pos_embed is not None:
            t_x = t_x + self.pos_embed.expand(B, -1, -1).type_as(t_x).to(t_x.device).clone().detach()
        t_x = self.pos_drop(t_x)
        #####################################################################
        
        ######################## CLIP TEXT path #############################
        if caption is not None:
            tokenized = torch.cat([tokenize(cap) for cap in caption]).to(x.device)
            xlight = self.clip_text_token_embedding(tokenized).type_as(x)
        else:
            text = " ".join(["X"] * (self.prefix + self.postfix))
            tokenized = tokenize(text).to(x.device)
            xlight = self.clip_text_token_embedding(tokenized).type_as(x).repeat([B, 1, 1])
            text_embedding = self.prompt_embedding(torch.arange(77).to(self.prompt_embedding.weight.device))[None, :].repeat([B, 1, 1])
            xlight[:,1:self.prefix+self.postfix+1,:] = text_embedding[:,1:self.prefix+self.postfix+1,:]
        text_x = xlight + self.clip_text_positional_embedding
        text_x = text_x.permute(1, 0, 2)   # NLD -> LND
        #####################################################################
        
        s_x = s_x.permute(1,0,2)
        for blk in self.blocks:
            # s_x, t_x = blk(s_x, t_x, text)
            s_x, t_x, text_x = blk(s_x, t_x, text_x)
        s_x = s_x.permute(1,0,2)
        text_x = text_x.permute(1,0,2)  # LND -> NLD
        
        s_x = rearrange(s_x, '(b t) n d -> b t n d', b=B)
        s_x = self.clip_ln_post(s_x[:,:,0,:].mean(1)) # all cls tokens avg pooling
        t_x = self.vmae_fc_norm(t_x.mean(1)) # all patch avg pooling
        text_x = self.clip_text_ln_final(text_x)
        eot = text_x[torch.arange(text_x.shape[0]), tokenized.argmax(dim=-1)] @ self.clip_text_text_projection
        if split_projection:
            sos = text_x[torch.arange(text_x.shape[0]), torch.zeros_like(tokenized.sum(-1))] @ self.clip_text_text_projection
            return s_x, t_x, eot, sos
        
        return s_x, t_x, eot
    
    def encode_text(self, xlight, text):
        x = xlight + self.clip_text_positional_embedding
        x = x.permute(1,0,2) # NLD -> LND
        for blk in self.text_blocks:
            x = blk(x)
        x = x.permute(1,0,2) # LND -> NLD
        x = self.clip_text_ln_final(x)
        
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip_text_text_projection
        
        return x

    def replace_text_embedding(self, actionlist, actiondict, actiontoken, videofeature = None, embedding = 'noun'):
        if videofeature is not None:
            B = videofeature.shape[0] # Batch size
            if embedding == 'noun':
                text_embedding = (self.noun_embedding(torch.arange(77).to(self.noun_embedding.weight.device))[None, :].expand([B,-1,-1]) + videofeature.unsqueeze(1).expand(-1, 77, -1)).unsqueeze(1).repeat([1, len(actionlist), 1, 1])
            else:
                text_embedding = (self.verb_embedding(torch.arange(77).to(self.verb_embedding.weight.device))[None, :].expand([B,-1,-1]) + videofeature.unsqueeze(1).expand(-1, 77, -1)).unsqueeze(1).repeat([1, len(actionlist), 1, 1])
            prompt_texttoken = torch.zeros(len(actionlist), 77).unsqueeze(0).repeat(B,1,1)
            
            for i, a in enumerate(actionlist):
                embedding = torch.from_numpy(actiondict[a][0]).float().to(self.verb_embedding.weight.device)
                token = torch.from_numpy(actiontoken[a][0])
                text_embedding[:,i,0] = embedding[0]
                ind = np.argmax(token, -1)

                text_embedding[:, i , self.prefix + 1: self.prefix + ind] = embedding[1:ind]
                text_embedding[:, i, self.prefix + ind + self.postfix] = embedding[ind]

                prompt_texttoken[:, i, 0] = token[0]
                prompt_texttoken[:, i, self.prefix + 1: self.prefix + ind] = token[1:ind]
                prompt_texttoken[:, i, self.prefix + ind + self.postfix] = token[ind]
            return text_embedding, prompt_texttoken
        else:
            if embedding == 'noun':
                text_embedding = self.noun_embedding(torch.arange(77).to(self.noun_embedding.weight.device))[None, :].repeat([len(actionlist), 1, 1])
            else:
                text_embedding = self.verb_embedding(torch.arange(77).to(self.verb_embedding.weight.device))[None, :].repeat([len(actionlist), 1, 1])
            prompt_texttoken = torch.zeros(len(actionlist), 77)
            
            for i, a in enumerate(actionlist):
                embedding = torch.from_numpy(actiondict[a][0]).float().to(self.noun_embedding.weight.device)
                token = torch.from_numpy(actiontoken[a][0])
                text_embedding[i][0] = embedding[0]
                ind = np.argmax(token, -1)

                text_embedding[i][self.prefix + 1: self.prefix + ind] = embedding[1:ind]
                text_embedding[i][self.prefix + ind + self.postfix] = embedding[ind]

                prompt_texttoken[i][0] = token[0]
                prompt_texttoken[i][self.prefix + 1: self.prefix + ind] = token[1:ind]
                prompt_texttoken[i][self.prefix + ind + self.postfix] = token[ind]
            return text_embedding, prompt_texttoken

    
    def forward(self, x, caption=None):
        if self.composition:
            if not self.split_projection:
                s_x, t_x, text_x = self.forward_features(x, caption=caption)
                if self.use_videoF and self.use_textF:
                    s_x = self.noun_last_Adapter(s_x) + self.text_noun_last_Adapter(text_x)
                    t_x = self.verb_last_Adapter(t_x) + self.text_verb_last_Adapter(text_x)
                elif self.use_videoF:
                    s_x = self.noun_last_Adapter(s_x)
                    t_x = self.verb_last_Adapter(t_x)
                else:
                    s_x = self.text_noun_last_Adapter(text_x)
                    t_x = self.text_verb_last_Adapter(text_x)
                s_x = self.head_noun_dropout(s_x)
                s_x = self.head_noun(s_x)
                t_x = self.head_verb_dropout(t_x)
                t_x = self.head_verb(t_x)
                return s_x, t_x
            else:
                s_x, t_x, eos, sos  = self.forward_features(x, caption=caption, split_projection=self.split_projection)
                if self.use_videoF and self.use_textF:
                    s_x = self.noun_last_Adapter(s_x) + self.text_noun_last_Adapter(sos)
                    t_x = self.verb_last_Adapter(t_x) + self.text_verb_last_Adapter(eos)
                elif self.use_videoF:
                    s_x = self.noun_last_Adapter(s_x)
                    t_x = self.verb_last_Adapter(t_x)
                else:
                    s_x = self.text_noun_last_Adapter(sos)
                    t_x = self.text_verb_last_Adapter(eos)
                s_x = self.head_noun_dropout(s_x)
                s_x = self.head_noun(s_x)
                t_x = self.head_verb_dropout(t_x)
                t_x = self.head_verb(t_x)
                return s_x, t_x
        else:
            s_x, t_x, text_x = self.forward_features(x)
            x = self.noun_last_Adapter(s_x) + self.verb_last_Adapter(t_x)
            x = self.head_dropout(x)
            x = self.head(x)
            return x


@register_model
def cast_bisquare_base_patch16_224(pretrained=False, args=None, class_list=None, **kwargs):
    textlist, textdict, texttoken = class_list
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, text_dim=512, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=False, 
        nounlist = textlist, noundict=textdict, nountoken=texttoken,
        prefix = 16, postfix = 16, split_prompt = args.split_prompt, **kwargs)
    return model

@register_model
def compo_cast_bisquare_base_patch16_224(pretrained=False, args=None, class_list=None, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, text_dim=512, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True, 
        prefix = 16, postfix = 16, **kwargs)
    return model

@register_model
def compo_cast_bisquare_8_base_patch16_224(pretrained=False, args=None, class_list=None, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, text_dim=512, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True, 
        prefix = 4, postfix = 4, use_textF=False, **kwargs)
    return model