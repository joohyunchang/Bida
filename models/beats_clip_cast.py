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
from models.beats.modules import SamePad, get_activation_fn
from models.beats.backbone import MultiheadAttention


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
    
# spatial to temporal cross attention module.
class CrossAttentionS2T(nn.Module):
    def __init__(self, dim: int, n_head: int, num_frames: int, spec_frames=1, attn_all_frame = True, audio_patch = 196, attn_mask: torch.Tensor = None):
        super().__init__()

        # add for cross-attn
        self.num_frames = num_frames//2
        self.spec_frames = spec_frames
        self.num_head = n_head
        head_dim = dim // self.num_head
        self.scale = head_dim ** -0.5
        all_head_dim = head_dim * self.num_head
        self.attn_all_frame = attn_all_frame
        if not attn_all_frame:
            # self.clip_space_pos = nn.Parameter(self.scale * torch.randn((audio_patch, dim)))
            # self.vmae_space_pos = nn.Parameter(self.scale * torch.randn((196, dim)))
            self.clip_temporal_pos = nn.Parameter(self.scale * torch.randn((spec_frames, dim)))
            self.vmae_temporal_pos = nn.Parameter(self.scale * torch.randn((num_frames//2, dim)))
        else:
            # self.clip_st_pos = nn.Parameter(self.scale * torch.randn((audio_patch * spec_frames, dim)))
            # self.vmae_st_pos = nn.Parameter(self.scale * torch.randn((196 * num_frames//2, dim)))
            self.clip_space_pos = nn.Parameter(self.scale * torch.randn((audio_patch, dim)))
            self.vmae_space_pos = nn.Parameter(self.scale * torch.randn((196, dim)))
            self.clip_temporal_pos = nn.Parameter(self.scale * torch.randn((spec_frames, dim)))
            self.vmae_temporal_pos = nn.Parameter(self.scale * torch.randn((num_frames//2, dim)))

        self.s2t_q = nn.Linear(dim, all_head_dim, bias=False)
        self.s2t_q_bias = nn.Parameter(torch.zeros(all_head_dim))
        self.s2t_kv = nn.Linear(dim, all_head_dim * 2, bias=False) # 197 tokens(cls+patch) * num_frames
        self.s2t_kv_bias = nn.Parameter(torch.zeros(all_head_dim * 2))
        
        self.t2s_proj = nn.Linear(all_head_dim, dim)
        
        self.attn_mask = attn_mask
    
    def s2t_cross_attn(self, s_x, t_x): # s_x=[n (b t) d], t_x=[b (t n) d]
        B, _, _ = t_x.shape
        t = self.num_frames
        s_x_pat = s_x
        t_x_cls, t_x_pat = t_x[:1, :, :], t_x[1:, :, :]
        if not self.attn_all_frame:
            # s_x_pat = rearrange(s_x_pat, 'n b d -> b n d') # batch -> token
            # s_x_pat = s_x_pat + self.clip_space_pos
            s_x_pat = rearrange(s_x_pat, 'n (b t) d -> b n t d', t=self.spec_frames)
            s_x_pat = s_x_pat + self.clip_temporal_pos
            s_x_pat = rearrange(s_x_pat, 'b n t d -> (b t) n d')
            if self.spec_frames != self.num_frames:
                exp = t // self.spec_frames
                s_x_pat = s_x_pat.unsqueeze(1).expand([-1 , exp, -1, -1])
                s_x_pat = rearrange(s_x_pat, 'b t n d -> (b t) n d')
            # t_x_pat = rearrange(t_x_pat, 'n (b t) d -> (b t) n d', t=t)
            # t_x_pat = t_x_pat + self.vmae_space_pos
            t_x_pat = rearrange(t_x_pat, 'n (b t) d -> b n t d', t=t)
            t_x_pat = t_x_pat + self.vmae_temporal_pos
            t_x_pat = rearrange(t_x_pat, 'b n t d -> (b t) n d')
        else:
            # s_x_pat = rearrange(s_x_pat, 'n (b t) d -> b (n t) d', t=self.spec_frames) # batch -> token
            # s_x_pat = s_x_pat + self.clip_st_pos
            # t_x_pat = rearrange(t_x_pat, 'n (b t) d -> b (n t) d', t=t)
            # t_x_pat = t_x_pat + self.vmae_st_pos
            s_x_pat = rearrange(s_x_pat, 'n (b t) d -> b t n d', t=self.spec_frames)
            s_x_pat = s_x_pat + self.clip_space_pos
            s_x_pat = rearrange(s_x_pat, 'b t n d -> b n t d')
            s_x_pat = s_x_pat + self.clip_temporal_pos
            s_x_pat = rearrange(s_x_pat, 'b n t d -> b (n t) d')
            t_x_pat = rearrange(t_x_pat, 'n (b t) d -> b t n d', t=t)
            t_x_pat = t_x_pat + self.vmae_space_pos
            t_x_pat = rearrange(t_x_pat, 'b t n d -> b n t d')
            t_x_pat = t_x_pat + self.vmae_temporal_pos
            t_x_pat = rearrange(t_x_pat, 'b n t d -> b (n t) d')
        s2t_q_bias = self.s2t_q_bias
        s2t_kv_bias = self.s2t_kv_bias
        
        s2t_q = F.linear(input=t_x_pat, weight=self.s2t_q.weight, bias=s2t_q_bias)
        s2t_q = rearrange(s2t_q, 'b n (h d) -> b h n d', h=self.num_head)
        s2t_kv = F.linear(input=s_x_pat, weight=self.s2t_kv.weight, bias=s2t_kv_bias)
        s2t_kv = rearrange(s2t_kv, 'b n (e h d) -> e b h n d',e=2, h=self.num_head)
        s2t_k, s2t_v = s2t_kv[0], s2t_kv[1]
        
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
        return t_x

    def forward(self, s_x: torch.Tensor, t_x: torch.Tensor):
        return self.s2t_cross_attn(s_x, t_x)


# this codes from CLIP github(https://github.com/openai/CLIP)
class CrossAttentionT2S(nn.Module):
    def __init__(self, dim: int, n_head: int, num_frames: int, spec_frames=1, attn_all_frame = True, audio_patch = 196, attn_mask: torch.Tensor = None):
        super().__init__()

        self.num_frames = num_frames//2
        self.spec_frames = spec_frames
        self.num_head = n_head
        head_dim = dim // self.num_head
        self.scale = head_dim ** -0.5
        all_head_dim = head_dim * self.num_head
        self.attn_all_frame = attn_all_frame
        if not attn_all_frame:
            # self.clip_space_pos = nn.Parameter(self.scale * torch.randn((audio_patch, dim)))
            # self.vmae_space_pos = nn.Parameter(self.scale * torch.randn((196, dim)))
            self.clip_temporal_pos = nn.Parameter(self.scale * torch.randn((spec_frames, dim)))
            self.vmae_temporal_pos = nn.Parameter(self.scale * torch.randn((num_frames//2, dim)))
        else:
            # self.clip_st_pos = nn.Parameter(self.scale * torch.randn((audio_patch * spec_frames, dim)))
            # self.vmae_st_pos = nn.Parameter(self.scale * torch.randn((196 * num_frames//2, dim)))
            self.clip_space_pos = nn.Parameter(self.scale * torch.randn((audio_patch, dim)))
            self.vmae_space_pos = nn.Parameter(self.scale * torch.randn((196, dim)))
            self.clip_temporal_pos = nn.Parameter(self.scale * torch.randn((spec_frames, dim)))
            self.vmae_temporal_pos = nn.Parameter(self.scale * torch.randn((num_frames//2, dim)))
        
        self.t2s_q = nn.Linear(dim, all_head_dim, bias=False) # 197 tokens(cls+patch) * num_frames
        self.t2s_q_bias = nn.Parameter(torch.zeros(all_head_dim))
        self.t2s_kv = nn.Linear(dim, all_head_dim * 2, bias=False)
        self.t2s_kv_bias = nn.Parameter(torch.zeros(all_head_dim * 2))
        
        self.t2s_proj = nn.Linear(all_head_dim, dim)
        
        self.attn_mask = attn_mask
    
    def t2s_cross_attn(self, s_x, t_x): # s_x=[n (b t) d], t_x=[b (t n) d]
        B, _, _ = t_x.shape
        t = self.num_frames
        s_x_pat = s_x
        t_x_pat = t_x[1:, :, :]
        if not self.attn_all_frame:
            # s_x_pat = rearrange(s_x_pat, 'n b d -> b n d') # batch -> token
            # s_x_pat = s_x_pat + self.clip_space_pos
            s_x_pat = rearrange(s_x_pat, 'n (b t) d -> b n t d', t=self.spec_frames)
            s_x_pat = s_x_pat + self.clip_temporal_pos
            s_x_pat = rearrange(s_x_pat, 'b n t d -> (b t) n d')
            if self.spec_frames != self.num_frames:
                exp = t // self.spec_frames
                s_x_pat = s_x_pat.unsqueeze(1).expand([-1 , t, -1, -1])
                s_x_pat = rearrange(s_x_pat, 'b t n d -> (b t) n d')
            # t_x_pat = rearrange(t_x_pat, 'n (b t) d -> (b t) n d', t=t)
            # t_x_pat = t_x_pat + self.vmae_space_pos
            t_x_pat = rearrange(t_x_pat, 'n (b t) d -> b n t d', t=t)
            t_x_pat = t_x_pat + self.vmae_temporal_pos
            t_x_pat = rearrange(t_x_pat, 'b n t d -> (b t) n d')
        else:
            # s_x_pat = rearrange(s_x_pat, 'n (b t) d -> b (n t) d', t=self.spec_frames) # batch -> token
            # s_x_pat = s_x_pat + self.clip_st_pos
            # t_x_pat = rearrange(t_x_pat, 'n (b t) d -> b (n t) d', t=t)
            # t_x_pat = t_x_pat + self.vmae_st_pos
            s_x_pat = rearrange(s_x_pat, 'n (b t) d -> b t n d', t=self.spec_frames)
            s_x_pat = s_x_pat + self.clip_space_pos
            s_x_pat = rearrange(s_x_pat, 'b t n d -> b n t d')
            s_x_pat = s_x_pat + self.clip_temporal_pos
            s_x_pat = rearrange(s_x_pat, 'b n t d -> b (n t) d')
            t_x_pat = rearrange(t_x_pat, 'n (b t) d -> b t n d', t=t)
            t_x_pat = t_x_pat + self.vmae_space_pos
            t_x_pat = rearrange(t_x_pat, 'b t n d -> b n t d')
            t_x_pat = t_x_pat + self.vmae_temporal_pos
            t_x_pat = rearrange(t_x_pat, 'b n t d -> b (n t) d')
        t2s_q_bias = self.t2s_q_bias
        t2s_kv_bias = self.t2s_kv_bias
        
        t2s_q = F.linear(input=s_x_pat, weight=self.t2s_q.weight, bias=t2s_q_bias)
        t2s_q = rearrange(t2s_q, 'b t (h d) -> b h t d', h=self.num_head)
        t2s_kv = F.linear(input=t_x_pat, weight=self.t2s_kv.weight, bias=t2s_kv_bias)
        t2s_kv = rearrange(t2s_kv, 'b t (e h d) -> e b h t d',e=2, h=self.num_head)
        t2s_k, t2s_v = t2s_kv[0], t2s_kv[1]
        
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
            s_x_pat = rearrange(s_x_pat, 'b (n t) d -> n (b t) d', t=self.spec_frames)
        s_x = s_x_pat
        return s_x

    def forward(self, s_x: torch.Tensor, t_x: torch.Tensor):
        return self.t2s_cross_attn(s_x, t_x)

    
class Block(nn.Module):
    def __init__(self, dim, num_heads, num_frames=16, mlp_ratio=4., down_ratio=2, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, num_layer=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None, 
                 spec_frames=1, attn_all_frame=True, CA=0, use_Adapter=True, audio_patch=196, 
                 relative_position_embedding=True, num_buckets=320, max_distance=800, gru_rel_pos=True):
        super().__init__()
        self.num_layer = num_layer
        self.num_heads = num_heads
        self.down_ratio = down_ratio
        self.scale = 0.5
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.act = act_layer()
        self.CA = CA
        self.use_Adapter = use_Adapter
        self.audio_patch = audio_patch
        
        ###################################### MHSA code #####################################
        ############################ BEATs MHSA ###########################
        self.activation_fn = get_activation_fn('gelu')
        self.self_attn = MultiheadAttention(
            dim,
            num_heads,
            dropout=attn_drop,
            self_attention=True,
            has_relative_attention_bias=relative_position_embedding,
            num_buckets=num_buckets,
            max_distance=max_distance,
            rescale_init=False,
            gru_rel_pos=gru_rel_pos,
        )
        self.dropout1 = nn.Dropout(drop)
        activation_dropout = 0.0
        self.dropout2 = nn.Dropout(activation_dropout)
        self.dropout3 = nn.Dropout(drop)
        self.self_attn_layer_norm = LayerNorm(dim)
        if self.use_Adapter:
            self.S_Adapter = Adapter(dim)
        ##################################################################
        
        ############################ CLIP MHSA ###########################
        self.clip_ln_1 = LayerNorm(dim)
        self.clip_attn = nn.MultiheadAttention(dim, num_heads)
        if self.use_Adapter:
            self.T_Adapter = Adapter(dim)
        ##################################################################
        #########################################################################################
        
        if num_layer >= self.CA:
            ###################################### Cross attention ####################################
            self.cross_s_down = nn.Linear(dim, dim//self.down_ratio)
            self.cross_t_down = nn.Linear(dim, dim//self.down_ratio)
            self.ln_s_cross = norm_layer(dim//self.down_ratio)
            self.ln_t_cross = norm_layer(dim//self.down_ratio)
            self.t2s_cross = CrossAttentionT2S(dim//self.down_ratio, num_heads, num_frames, spec_frames=spec_frames, attn_all_frame=attn_all_frame, audio_patch=audio_patch)
            self.s2t_cross = CrossAttentionS2T(dim//self.down_ratio, num_heads, num_frames, spec_frames=spec_frames, attn_all_frame=attn_all_frame, audio_patch=audio_patch)
            self.cross_s_up = nn.Linear(dim//self.down_ratio, dim)
            self.cross_t_up = nn.Linear(dim//self.down_ratio, dim)
            ###########################################################################################
        
        ###################################### FFN code #########################################
        ############################ BEATs FFN ###############################
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.final_layer_norm = LayerNorm(dim)
        self.deep_norm_alpha = math.pow(2 * num_layer, 1 / 4)
        if self.use_Adapter:
            self.S_MLP_Adapter = Adapter(dim, skip_connect=False)
        self.attn_mask = None
        #####################################################################
        
        ############################ CLIP FFN ###############################
        self.clip_ln_2 = LayerNorm(dim)
        self.clip_mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(dim, dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(dim * 4, dim))
        ]))
        if self.use_Adapter:
            self.T_MLP_Adapter = Adapter(dim, skip_connect=False)
        #######################################################################
        #########################################################################################
        

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.clip_attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, s_x, t_x, pos_bias=None):
        B = t_x.shape[0]
        n, bt, _ = s_x.shape
        num_frames = bt//B
        
        ############################ MHSA Forward #############################
        residual = s_x
        s_x, attn, pos_bias = self.self_attn(
            query=s_x,
            key=s_x,
            value=s_x,
            key_padding_mask=torch.zeros(16,self.audio_patch, device=t_x.device).bool(),
            need_weights=False,
            attn_mask=None,
            position_bias=pos_bias
        )
        s_x = self.dropout1(s_x)
        if self.use_Adapter:
            # BEATs Space MHSA
            s_x = residual * self.deep_norm_alpha + self.S_Adapter(s_x)
            s_x = self.self_attn_layer_norm(s_x)
            # CLIP Time MHSA
            t_x = t_x + self.T_Adapter(self.attention(self.clip_ln_1(t_x)))
        else:
            # BEATs Space MHSA
            s_x = residual * self.deep_norm_alpha + s_x
            s_x = self.self_attn_layer_norm(s_x)
            # CLIP Time MHSA
            t_x = t_x + self.attention(self.clip_ln_1(t_x))
        ########################################################################
        
        if self.num_layer >= self.CA:
            ############################ Cross Forward #############################
            n_s_x = self.ln_s_cross(self.cross_s_down(s_x))
            n_t_x = self.ln_t_cross(self.cross_t_down(t_x))
            c_s_x = self.cross_s_up(self.act(self.t2s_cross(n_s_x, n_t_x)))
            c_t_x = self.cross_t_up(self.act(self.s2t_cross(n_s_x, n_t_x)))
            s_x = s_x + self.drop_path(c_s_x)
            t_x = t_x + self.drop_path(c_t_x)
            #########################################################################
        
        ############################ FFN Forward ##################################
        residual = s_x
        s_x = self.activation_fn(self.fc1(s_x))
        s_x = self.dropout2(s_x)
        s_x = self.fc2(s_x)
        s_x = self.dropout3(s_x)
        t_xn = self.clip_ln_2(t_x)
        if self.use_Adapter:
            s_x = residual * self.deep_norm_alpha + self.S_MLP_Adapter(s_x)
            s_x = self.final_layer_norm(s_x)
            t_x = t_x + self.clip_mlp(t_xn) + self.drop_path(self.scale * self.T_MLP_Adapter(t_xn))
        else:
            s_x = residual * self.deep_norm_alpha + s_x
            s_x = self.final_layer_norm(s_x)
            t_x = t_x + self.clip_mlp(t_xn)
        ############################################################################
        
        return s_x, t_x, pos_bias
    
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
                 pretrained_cfg = None,
                 pretrained_cfg_overlay = None):
        super().__init__()
        self.num_classes = num_classes
        self.num_frames = all_frames
        self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size
        self.down_ratio = down_ratio
        self.composition = composition
        self.audio_enabled = audio_enabled
        self.audio_patch = audio_patch
        
        scale = embed_dim ** -0.5
        self.clip_conv1 = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.clip_class_embedding = nn.Parameter(scale * torch.randn(embed_dim))
        self.clip_positional_embedding = nn.Parameter(scale * torch.randn((img_size // patch_size) ** 2 + 1, embed_dim))
        self.clip_ln_pre = LayerNorm(embed_dim)

        ###################################
        self.embed = 512
        input_patch_size = 16
        self.patch_embedding = nn.Conv2d(1, self.embed, kernel_size=input_patch_size, stride=input_patch_size, bias=False)
        self.layer_norm = LayerNorm(self.embed)
        
        self.post_extract_proj = (
            nn.Linear(self.embed, self.embed_dim)
            if self.embed != self.embed_dim
            else None
        )
        self.dropout_input = nn.Dropout(0.0)
        
        # self.layer_wise_gradient_decay_ratio = 0.6
        conv_pos = 128
        self.pos_conv = nn.Conv1d(
            self.embed_dim,
            self.embed_dim,
            kernel_size=conv_pos, # args.conv_pos
            padding=conv_pos // 2, # args.conv_pos // 2
            groups=16, # args.conv_pos_groups
        )
        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(conv_pos), nn.GELU())
        self.layer_norm_first = LayerNorm(self.embed_dim)
        self.layerdrop = 0.05
        
        self.relative_position_embedding = True # args.relative_position_embedding
        self.num_buckets = 320 # args.num_buckets
        self.max_distance = 800 # args.max_distance
        gru_rel_pos = True
        
        
        spec_frames = (spec_frames+1) //2
        attn_all_frame=attn_all_frame
        CA=CA
        ###################################

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, num_frames=self.num_frames, mlp_ratio=mlp_ratio,down_ratio=self.down_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, num_layer=i, spec_frames=spec_frames, attn_all_frame=attn_all_frame, CA=CA, use_Adapter=use_Adapter, audio_patch=audio_patch,
                relative_position_embedding=self.relative_position_embedding, num_buckets=self.num_buckets, max_distance=self.max_distance, gru_rel_pos=gru_rel_pos)
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

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'clip_time_pos','clip_space_pos','vmae_space_pos','vmae_time_pos','pos_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    
    def forward_features(self, x, spec=None):
        B = x.shape[0]
        s_x = spec[:, :, 1::2, :, :] if spec.dim() == 5 else spec.unsqueeze(2)
        s_x = s_x[:,:1,:,:,:]
        # s_x = torch.randn_like(x[:, :, 1::2, :, :]) # pick even frames
        ######################## Audio spatial path #########################
        s_x = rearrange(s_x, 'b c t h w -> (b t) c h w')
        s_x = self.patch_embedding(s_x)
        s_x = s_x.reshape(s_x.shape[0], s_x.shape[1], -1) # [*, embeddim, grid**2] # B C T
        s_x = s_x.permute(0, 2, 1) # shape[batch, patchnum, embeddim] # B T(196) C(768)
        s_x = self.layer_norm(s_x)
        
        if self.post_extract_proj is not None:
            s_x = self.post_extract_proj(s_x)

        s_x = self.dropout_input(s_x)
        #####################################################################
        
        ######################## CLIP spatial path #########################
        t_x = x[:, :, 1::2, :, :]
        t_t = t_x.shape[2]
        t_x = rearrange(t_x, 'b c t h w -> (b t) c h w')
        t_x = self.clip_conv1(t_x) # shape = [*, embeddim, grid, grid] 
        t_x = t_x.reshape(t_x.shape[0], t_x.shape[1], -1) # [*, embeddim, grid**2] (bt) d hw
        t_x = t_x.permute(0, 2, 1) # shape[batch, patchnum, embeddim] bt hw d
        t_x = torch.cat([self.clip_class_embedding.to(t_x.dtype) + torch.zeros(t_x.shape[0], 1, t_x.shape[-1], dtype=t_x.dtype, device=t_x.device), t_x], dim=1)
        t_x = t_x + self.clip_positional_embedding.to(t_x.dtype) # bt hw+1 d
        t_x = self.clip_ln_pre(t_x)
        #####################################################################
        
        s_x_conv = self.pos_conv(s_x.transpose(1, 2))
        s_x_conv = s_x_conv.transpose(1, 2)
        s_x = s_x + s_x_conv
        s_x = self.layer_norm_first(s_x)
        s_x = s_x.transpose(0,1) # T x B x C
        
        t_x = t_x.permute(1,0,2)
        pos_bias = None
        for blk in self.blocks:
            s_x, t_x, pos_bias = blk(s_x, t_x, pos_bias)
        t_x = t_x.permute(1,0,2)
        
        # T x B x C -> B x T x C
        s_x = s_x.transpose(0, 1)
        
        s_x = s_x.mean(1)
        t_x = rearrange(t_x, '(b t) n d -> b t n d', b=B)
        t_x = self.clip_ln_post(t_x[:,:,0,:].mean(1)) # all patch avg pooling
        
        return s_x, t_x

    def forward(self, x, caption=None, spec=None):
        # if spec is not None:
        #     spec = torch.stack(spec, dim=0)
        if self.audio_enabled:
            s_x, t_x = self.forward_features(x, spec)
            x = self.noun_last_Adapter(s_x) + self.verb_last_Adapter(t_x)
            # x = self.verb_last_Adapter(t_x)
            s_x = self.head_noun_dropout(x)
            s_x = self.head_noun(s_x)
            t_x = self.head_verb_dropout(x)
            t_x = self.head_verb(t_x)
            return s_x, t_x
        elif self.composition:
            s_x, t_x = self.forward_features(x, spec)
            s_x = self.head_noun_dropout(s_x)
            s_x = self.head_noun(s_x)
            t_x = self.head_verb_dropout(t_x)
            t_x = self.head_verb(t_x)
            return s_x, t_x
        else:
            s_x, t_x = self.forward_features(x)
            x = (self.noun_last_Adapter(s_x) + self.verb_last_Adapter(t_x))/2
            x = self.head_dropout(x)
            x = self.head(x)
            return x

# @register_model
# def bidir_vit_base_patch16_224(pretrained=False, **kwargs):
#     model = STCrossTransformer(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=False, **kwargs)
#     return model

# @register_model
# def compo_bidir_vit_base_patch16_224(pretrained=False, **kwargs):
#     model = STCrossTransformer(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True, **kwargs)
#     return model

@register_model
def compo_single_beats_clip_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True, audio_enabled=True, CA=0, spec_frames=1, attn_all_frame=True, **kwargs)
    return model

@register_model
def compo_single_beats_clip_down4_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True, audio_enabled=True, 
        CA=0, spec_frames=1, attn_all_frame=True, down_ratio=4, **kwargs)
    return model

@register_model
def compo_single_beats_clip_Patch512_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True, audio_enabled=True, 
        CA=0, spec_frames=1, attn_all_frame=True, down_ratio=4, audio_patch=512, **kwargs)
    return model

@register_model
def compo_single_beats_clip_down4_Patch512_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True, audio_enabled=True, 
        CA=0, spec_frames=1, attn_all_frame=True, down_ratio=4, audio_patch=512, **kwargs)
    return model

@register_model
def compo_single_beats_clip_CA9_down4_Patch512_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True, audio_enabled=True, 
        CA=9, spec_frames=1, attn_all_frame=True, down_ratio=4, audio_patch=512, **kwargs)
    return model

@register_model
def compo_single_beats_clip_CA0_down4_noAdap_Patch512_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True, audio_enabled=True, 
        CA=0, spec_frames=1, attn_all_frame=True, down_ratio=4, audio_patch=512, use_Adapter=False, **kwargs)
    return model

@register_model
def compo_single_beats_clip_down4_noAdap_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True, audio_enabled=True, 
        CA=0, spec_frames=1, attn_all_frame=True, down_ratio=4, use_Adapter=False, **kwargs)
    return model

@register_model
def compo_single_beats_clip_down8_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True, audio_enabled=True, 
        CA=0, spec_frames=1, attn_all_frame=True, down_ratio=8, **kwargs)
    return model

@register_model
def compo_single_beats_clip_CA9_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True, audio_enabled=True, CA=9, spec_frames=1, attn_all_frame=True, **kwargs)
    return model

# @register_model
# def compo_stacks_beats_clip_vit_base_patch16_224(pretrained=False, **kwargs):
#     model = STCrossTransformer(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True, audio_enabled=True, CA=0, spec_frames=16, attn_all_frame=True, **kwargs)
#     return model


@register_model
def compo_single_fbf_beats_clip_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True, audio_enabled=True, CA=0, spec_frames=1, attn_all_frame=False, **kwargs)
    return model

@register_model
def compo_stacks_beats_clip_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True, audio_enabled=True, CA=0, spec_frames=16, attn_all_frame=False, **kwargs)
    return model

@register_model
def compo_stacks_full_beats_clip_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True, audio_enabled=True, CA=0, spec_frames=16, attn_all_frame=True, **kwargs)
    return model


@register_model
def compo_stacks_beats_clip_CA9_vit_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True, audio_enabled=True, CA=9, spec_frames=16, attn_all_frame=False, **kwargs)
    return model