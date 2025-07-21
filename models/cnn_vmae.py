# some codes from CLIP github(https://github.com/openai/CLIP), from VideoMAE github(https://github.com/MCG-NJU/VideoMAE)
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from torchvision.models import resnet50, ResNet50_Weights
from collections import OrderedDict
from einops import rearrange
import random
from math import sqrt


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
    def __init__(self, dim: int, n_head: int, num_frames: int, attn_mask: torch.Tensor = None, cnn_shape=None):
        super().__init__()

        # add for cross-attn
        self.num_frames = num_frames
        self.num_head = n_head
        head_dim = dim // self.num_head
        self.scale = head_dim ** -0.5
        all_head_dim = head_dim * self.num_head
        self.cnn_space_pos = nn.Parameter(self.scale * torch.randn((cnn_shape[-1] * cnn_shape[-2], dim)))
        self.vmae_space_pos = nn.Parameter(self.scale * torch.randn((196, dim)))
        

        self.s2t_q = nn.Linear(dim, all_head_dim, bias=False) # vmae
        self.s2t_q_bias = nn.Parameter(torch.zeros(all_head_dim))
        self.s2t_kv = nn.Linear(dim, all_head_dim * 2, bias=False) # 197 tokens(cls+patch) * num_frames
        self.s2t_kv_bias = nn.Parameter(torch.zeros(all_head_dim * 2))
        
        self.s2t_proj = nn.Linear(all_head_dim, dim)
        
        self.attn_mask = attn_mask
    
    def s2t_cross_attn(self, s_x, t_x): # s_x=[(b t) h w c], t_x=[b n d]
        B, _, _ = t_x.shape
        _, H, W, C = s_x.shape
        t = s_x.shape[0] // t_x.shape[0]
        s_x_pat = rearrange(s_x, '(b t) h w c-> (b t) (h w) c', b=B, t=t) # [B, T, C, H, W]
        s_x_pat = s_x_pat + self.cnn_space_pos
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
        t_x = self.s2t_proj(t_x)
        t_x = rearrange(t_x, '(b t) n d -> b (t n) d', b=B)
        return t_x

    def forward(self, s_x: torch.Tensor, t_x: torch.Tensor):
        return self.s2t_cross_attn(s_x, t_x)


# this codes from CLIP github(https://github.com/openai/CLIP)
class CrossAttentionT2S(nn.Module):
    def __init__(self, dim: int, n_head: int, num_frames: int, attn_mask: torch.Tensor = None, cnn_shape=None):
        super().__init__()

        self.num_frames = num_frames
        self.num_head = n_head
        head_dim = dim // self.num_head
        self.scale = head_dim ** -0.5
        all_head_dim = head_dim * self.num_head
        self.cnn_time_pos = nn.Parameter(self.scale * torch.randn((num_frames//2, dim)))
        self.vmae_time_pos = nn.Parameter(self.scale * torch.randn((num_frames//2, dim)))
        self.cnn_shape = cnn_shape
        
        self.t2s_q = nn.Linear(dim, all_head_dim, bias=False) # cnn filters
        self.t2s_q_bias = nn.Parameter(torch.zeros(all_head_dim))
        self.t2s_kv = nn.Linear(dim, all_head_dim * 2, bias=False) # vmae
        self.t2s_kv_bias = nn.Parameter(torch.zeros(all_head_dim * 2))
        
        self.t2s_proj = nn.Linear(all_head_dim, dim)
        
        self.attn_mask = attn_mask
    
    def t2s_cross_attn(self, s_x, t_x): # s_x=[(b t) h w c], t_x=[b n d]
        B, TN, D = t_x.shape
        _, H, W, C = s_x.shape
        t = s_x.shape[0] // t_x.shape[0]
        N = TN // t
        t_h, t_w = int(sqrt(N)), int(sqrt(N))
        s_x_pat = rearrange(s_x, '(b t) h w c -> b (h w) t c', b=B) # batch -> token
        s_x_pat = s_x_pat+ self.cnn_time_pos
        t_x = rearrange(t_x, 'b (t n) d -> b n t d', t=t,)
        t_x = t_x + self.vmae_time_pos
        if s_x_pat.shape[1] > t_x.shape[1]:
            scale_h, scale_w = int(H // t_h), int(W // t_w)
            t_x = rearrange(t_x, 'b (h w) t d -> b h w t d', h=t_h, w=t_w)
            t_x_interleave = t_x.repeat_interleave(scale_h, dim=1).repeat_interleave(scale_w, dim=2)
            assert t_x_interleave.shape[1] == H and t_x_interleave.shape[2] == W, f"t_x_interleave shape: {t_x_interleave.shape}, H: {H}, W: {W}"
            t_x = rearrange(t_x_interleave, 'b h w t d -> b (h w) t d') # [B, H*W, T, D]
        elif s_x_pat.shape[1] < t_x.shape[1]:
            assert t_h % H == 0 and t_w % W == 0, f"t_h: {t_h}, t_w: {t_w}, H: {H}, W: {W}"
            patch_per_cnn = N // (H * W)
            t_x_grouped = t_x.view(B,H*W,patch_per_cnn, t, D) # [B, H*W, patch_per_cnn, T, D]
            t_x = t_x_grouped.reshape(B, H*W,patch_per_cnn*t, D) # [B, H*W,patch_per_cnn*T, D]
        
        s_x_pat = rearrange(s_x_pat, 'b hw t c -> (b hw) t c')
        t_x = rearrange(t_x, 'b n t d -> (b n) t d')
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
        s_x_pat = rearrange(s_x_pat, 'b h n c -> b n (h c)')
        s_x_pat = self.t2s_proj(s_x_pat)
        s_x = rearrange(s_x_pat,'(b h w) t c -> (b t) h w c', b=B, h=H, w=W)
        return s_x

    def forward(self, s_x: torch.Tensor, t_x: torch.Tensor):
        return self.t2s_cross_attn(s_x, t_x)

    
class Block(nn.Module):
    def __init__(self, dim, num_heads, num_frames=16, mlp_ratio=4., down_ratio=2, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, num_layer=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None, 
                 use_Adapter=True, late_fusion=0, CA=None, use_SA=True, use_MLP=True, cnn_model=None, cnn_hidden_shape=None):
        super().__init__()
        self.num_layer = num_layer
        self.num_heads = num_heads
        self.down_ratio = down_ratio
        self.scale = 0.5
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.act = act_layer()
        self.use_Adapter = use_Adapter 
        self.late_fusion = late_fusion
        self.CA = CA
        self.use_SA, self.use_MLP = True, True
        if num_layer >= late_fusion:
            self.use_Adapter = True
        self.enable_B_CA = False
        ###################################### MHSA code #####################################
        ############################ CNN MHSA ###########################
        if num_layer in CA:
            idx = CA.index(num_layer)
            self.cnn_shape = cnn_hidden_shape['after_layer' + str(idx + 1)]
            cnn_backbone = cnn_model['layer' + str(idx + 1)]
            self.cnn_layer = cnn_backbone
            self.enable_B_CA = True
        ##################################################################
        
        ############################ VMAE MHSA ###########################
        if self.use_SA:
            self.norm1 = norm_layer(dim)
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
            if self.use_Adapter:
                self.T_Adapter = Adapter(dim)
        ##################################################################
        #########################################################################################
        
        if self.enable_B_CA:
            ###################################### Cross attention ####################################
            self.cross_s_down = nn.Linear(self.cnn_shape[1], dim//self.down_ratio)
            self.cross_t_down = nn.Linear(dim, dim//self.down_ratio)
            self.ln_s_cross = norm_layer(dim//self.down_ratio)
            self.ln_t_cross = norm_layer(dim//self.down_ratio)
            self.t2s_cross = CrossAttentionT2S(dim//self.down_ratio, num_heads, num_frames, cnn_shape=self.cnn_shape)
            self.s2t_cross = CrossAttentionS2T(dim//self.down_ratio, num_heads, num_frames, cnn_shape=self.cnn_shape)
            self.cross_s_up = nn.Linear(dim//self.down_ratio, self.cnn_shape[1])
            self.cross_t_up = nn.Linear(dim//self.down_ratio, dim)
            ###########################################################################################
        
        ###################################### FFN code #########################################
        ############################ AIM FFN ###############################
        #####################################################################
        
        ############################ VMAE FFN ###############################
        if self.use_MLP:
            self.norm2 = norm_layer(dim)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
            if self.use_Adapter:
                self.T_MLP_Adapter = Adapter(dim, skip_connect=False)
        #######################################################################
        #########################################################################################
        

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.clip_attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self,s_x, t_x):
        B = t_x.shape[0]
        bt, _, _, _ = s_x.shape
        num_frames = bt//B
        
        ############################ MHSA Forward #############################
        if self.use_SA:
            if self.use_Adapter:
                # VMAE Time MHSA
                t_x = t_x + self.T_Adapter(self.attn(self.norm1(t_x)))
            else:
                # VMAE Time MHSA
                t_x = t_x + self.attn(self.norm1(t_x))
        ########################################################################
        
        if self.enable_B_CA:
            # CNN
            s_x = self.cnn_layer(s_x)
            ############################ Cross Forward #############################
            s_x = rearrange(s_x, 'bt c h w -> bt h w c')
            n_s_x = self.ln_s_cross(self.cross_s_down(s_x))
            n_t_x = self.ln_t_cross(self.cross_t_down(t_x))
            c_s_x = self.cross_s_up(self.act(self.t2s_cross(n_s_x, n_t_x)))
            c_t_x = self.cross_t_up(self.act(self.s2t_cross(n_s_x, n_t_x)))
            s_x = s_x + self.drop_path(c_s_x)
            s_x = rearrange(s_x, 'bt h w c -> bt c h w')
            t_x = t_x + self.drop_path(c_t_x)
            #########################################################################
        
        ############################ FFN Forward ##################################
        if self.use_MLP:
            t_xn = self.norm2(t_x)
            if self.use_Adapter:
                t_x = t_x + self.mlp(t_xn) + self.drop_path(self.scale * self.T_MLP_Adapter(t_xn))
            else:
                t_x = t_x + self.mlp(t_xn)
        ############################################################################
        
        return s_x, t_x
    
class STCrossTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 vmae_patch_size=None,
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
                 audio_patch=False,
                 late_fusion=0,
                 CA = 0,
                 use_Adapter=True,
                 use_SA=True, 
                 use_MLP=True,
                 clip_only=False,
                 vmae_only=False,
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
        self.clip_only, self.vmae_only = clip_only, vmae_only
        # =================================================================
        self.vmae_patch_size = vmae_patch_size if vmae_patch_size is not None else patch_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=self.vmae_patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames, tubelet_size=self.tubelet_size)
        num_patches = self.patch_embed.num_patches
        
        scale = embed_dim ** -0.5
        # self.clip_conv1 = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        # self.clip_class_embedding = nn.Parameter(scale * torch.randn(embed_dim))
        # self.clip_positional_embedding = nn.Parameter(scale * torch.randn((img_size // patch_size) ** 2 + 1, embed_dim))
        # self.clip_ln_pre = LayerNorm(embed_dim)

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)
        
        ######################
        # CA = [3,7,11]
        CA = [2,5,8,11]
        # For ResNet
        self.cnn_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        self.cnn_stem = nn.Sequential(
            self.cnn_model.conv1,
            self.cnn_model.bn1,
            self.cnn_model.relu,
            self.cnn_model.maxpool
        )
        self.pooling = self.cnn_model.avgpool
        dummy_input = torch.randn(1, 3, img_size, img_size)
        dummy_stem = self.cnn_stem(dummy_input)
        dummy_layer1 = self.cnn_model.layer1(dummy_stem)
        dummy_layer2 = self.cnn_model.layer2(dummy_layer1)
        dummy_layer3 = self.cnn_model.layer3(dummy_layer2)
        dummy_layer4 = self.cnn_model.layer4(dummy_layer3)
        dummy_pooling = self.pooling(dummy_layer4)
        self.cnn_hidden_shape = {'after_stem': dummy_stem.shape,
                                'after_layer1': dummy_layer1.shape,
                                'after_layer2': dummy_layer2.shape,
                                'after_layer3': dummy_layer3.shape,
                                'after_layer4': dummy_layer4.shape,
                                'after_pooling': dummy_pooling.shape}
        self.cnn_layers = {'layer1': self.cnn_model.layer1,
                          'layer2': self.cnn_model.layer2,
                          'layer3': self.cnn_model.layer3,
                          'layer4': self.cnn_model.layer4}
        ######################


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, num_frames=self.num_frames, mlp_ratio=mlp_ratio,down_ratio=self.down_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, late_fusion=late_fusion, CA=CA, use_Adapter=use_Adapter, use_SA=use_SA, use_MLP=use_MLP,
                init_values=init_values, num_layer=i, cnn_model=self.cnn_layers, cnn_hidden_shape=self.cnn_hidden_shape,)
            for i in range(depth)])
        
        # self.clip_ln_post = LayerNorm(embed_dim)
        self.vmae_fc_norm = norm_layer(embed_dim)
        
        if self.composition:
            self.head_verb = nn.Linear(embed_dim, 97)
            self.head_verb_dropout = nn.Dropout(head_drop_rate)
            self.head_noun = nn.Linear(self.cnn_hidden_shape['after_pooling'][1], 300)
            self.head_noun_dropout = nn.Dropout(head_drop_rate)
        else:
            self.noun_projection = nn.Linear(self.cnn_hidden_shape['after_pooling'][1], embed_dim)
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
        else:
            self.noun_projection.weight.data.mul_(init_scale)
            self.noun_projection.bias.data.mul_(init_scale)
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
        return {'cnn_time_pos','cnn_space_pos','vmae_space_pos','vmae_time_pos','pos_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    
    def reset_fcnorm(self):
        self.vmae_fc_norm = nn.LayerNorm(self.embed_dim)
    
    def forward_features(self, x, spec=None):
        B = x.shape[0]
        if spec is not None:
            s_x = spec[:, :, 1::2, :, :] if spec.dim() == 5 else spec.unsqueeze(2)
        else:
            s_x = x[:, :, 1::2, :, :] # pick even frames
        # s_x = torch.randn_like(x[:, :, 1::2, :, :]) # pick even frames
        ######################## AIM spatial path #########################
        s_t = s_x.shape[2]
        s_x = rearrange(s_x, 'b c t h w -> (b t) c h w')
        s_x = self.cnn_stem(s_x) # [BT, C, H, W] [BT, 64, 56, 56]
        #####################################################################
        
        ######################## VMAE spatial path #########################
        t_x = self.patch_embed(x) # [B, 1568, 768]

        if self.pos_embed is not None:
            t_x = t_x + self.pos_embed.expand(B, -1, -1).type_as(t_x).to(t_x.device).clone().detach()
        t_x = self.pos_drop(t_x)
        #####################################################################
        
        for blk in self.blocks:
            s_x, t_x = blk(s_x, t_x)
        
        s_x = self.pooling(s_x) # [BT, 2048, 1, 1]
        s_x = rearrange(s_x.squeeze(-1).squeeze(-1), '(b t) d -> b t d', b=B)
        s_x = s_x.mean(1) # all frame avg pooling
        t_x = self.vmae_fc_norm(t_x.mean(1)) # all patch avg pooling
        
        return s_x, t_x

    def forward(self, x, caption=None, spec=None):
        if self.clip_only or self.vmae_only:
            s_x, t_x = self.forward_features(x)
            x = s_x if self.clip_only else t_x
            if self.composition:
                return self.head_noun(self.head_noun_dropout(x)), self.head_verb(self.head_verb_dropout(x))
            else:
                return self.head(self.head_dropout(x))
        elif self.audio_enabled:
            s_x, t_x = self.forward_features(x, spec)
            s_x = self.head_noun_dropout(s_x)
            s_x = self.head_noun(s_x)
            t_x = self.head_verb_dropout(t_x)
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
            x = (self.noun_projection(s_x) + self.verb_last_Adapter(t_x))/2
            x = self.head_dropout(x)
            x = self.head(x)
            return x

@register_model
def resnet50_vmae_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=False, **kwargs)
    return model

@register_model
def compo_resnet50_vmae_base_patch16_224(pretrained=False, **kwargs):
    model = STCrossTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), composition=True, **kwargs)
    return model

class VideoResNetWithHead(nn.Module):
    def __init__(self, num_classes=400, feature_agg='mean', img_size=224, composition=False, init_scale=0., head_drop_rate=0.,):
        super().__init__()
        # load CLIP vision backbone
        self.composition = composition
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        if self.composition:
            self.model.fc = nn.Identity()
            self.head_verb = nn.Linear(2048, 97)
            self.head_verb_dropout = nn.Dropout(head_drop_rate)
            self.head_noun = nn.Linear(2048, 300)
            self.head_noun_dropout = nn.Dropout(head_drop_rate)
            self.head_verb.weight.data.mul_(init_scale)
            self.head_verb.bias.data.mul_(init_scale)
            self.head_noun.weight.data.mul_(init_scale)
            self.head_noun.bias.data.mul_(init_scale)
        else:
            self.model.fc = nn.Identity()
            self.head = nn.Linear(2048, num_classes)
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)
            
        self.feature_agg = feature_agg  # 'mean' or 'sum'
    
    def get_num_layers(self):
        return len(self.model.layer1) + len(self.model.layer2) + len(self.model.layer3) + len(self.model.layer4)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'embeddings'}

    def forward(self, x, **kwargs):  # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        # (1) Frame 단위로 CLIP backbone 통과 (batch concat)
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # [B*T, C, H, W]
        outputs = self.model(x)
        features = outputs.view(B, T, -1)  # [B, T, D]
        # (2) 모든 frame feature를 sum/mean pooling
        if self.feature_agg == 'mean':
            vid_feat = features.mean(dim=1)
        elif self.feature_agg == 'sum':
            vid_feat = features.sum(dim=1)
        else:
            raise NotImplementedError
        if self.composition:
            s_x = self.head_noun_dropout(vid_feat)
            s_x = self.head_noun(s_x)
            t_x = self.head_verb_dropout(vid_feat)
            t_x = self.head_verb(t_x)
            return s_x, t_x
        else:
            out = self.head(vid_feat)  # [B, num_classes]
            return out
            

@register_model
def resnet50_model(pretrained=False, **kwargs):
    print('num_classes = %s' % kwargs.get('num_classes', None))
    model = VideoResNetWithHead(num_classes=kwargs.get('num_classes', 400), 
                                feature_agg='mean', 
                                img_size=kwargs.get('img_size',224), 
                                composition=False,
                                head_drop_rate=kwargs.get('head_drop_rate', 0.),
                                init_scale=kwargs.get('init_scale', 0.))
    return model

@register_model
def compo_resnet50_model(pretrained=False, **kwargs):
    print('num_classes = %s' % kwargs.get('num_classes', None))
    model = VideoResNetWithHead(num_classes=kwargs.get('num_classes', 400), 
                                feature_agg='mean', 
                                img_size=kwargs.get('img_size',224), 
                                composition=True,
                                head_drop_rate=kwargs.get('head_drop_rate', 0.),
                                init_scale=kwargs.get('init_scale', 0.))
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
    # model = resnet50_vmae_base_patch16_224(num_classes=400).cuda().half()
    model = resnet50_model(num_classes=400).cuda().half()
    model.eval()
    print(model)

    dummy_input = torch.rand(1, 3, num_frames, img_size, img_size).cuda().half()
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