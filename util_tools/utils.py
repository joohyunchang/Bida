import io
import os
import math
import time
import json
from collections import defaultdict, deque, OrderedDict
from typing import Optional, Sequence
import datetime
import numpy as np
from timm.utils import get_state_dict
from timm.models.layers import trunc_normal_
from torch.utils.data._utils.collate import default_collate
from pathlib import Path
import subprocess
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from torch import inf
import random
import requests
from typing import Dict
import pandas as pd


from tensorboardX import SummaryWriter


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def flush(self):
        self.writer.flush()

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process, except for error messages
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    # elif 'SLURM_PROCID' in os.environ:
    #     if "WORLD_SIZE" in os.environ:
    #         args.world_size = int(os.environ["WORLD_SIZE"])
    #     ngpus_per_node = torch.cuda.device_count()
    #     args.rank = int(os.environ['SLURM_PROCID'])
    #     args.gpu = args.rank % torch.cuda.device_count()
    #     args.world_size = int(os.environ['SLURM_NTASKS'])
    #     # os.environ['RANK'] = str(args.rank)
    #     # os.environ['LOCAL_RANK'] = str(args.gpu)
    #     # os.environ['WORLD_SIZE'] = str(args.world_size)

    #     node_list = os.environ['SLURM_NODELIST']
    #     addr = subprocess.getoutput(
    #         f'scontrol show hostname {node_list} | head -n1')
    #     os.environ['MASTER_ADDR'] = addr
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank, timeout=datetime.timedelta(seconds=200000))
    torch.distributed.barrier()
    # assert torch.distributed.is_initialized()
    setup_for_distributed(args.rank == 0)


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index",freeze_list=None):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        # module._load_from_state_dict(
        #     state_dict=state_dict, prefix=prefix, local_metadata=local_metadata,
        #     strict=True, missing_keys=missing_keys, unexpected_keys=unexpected_keys,
        #     error_msgs=error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    before_key = model.state_dict().keys()
    load_key = state_dict.keys()
    load(model, prefix=prefix)
    if freeze_list is not None:
        loaded_key = set(before_key).intersection(set(load_key))
        freeze_but_notload = set(freeze_list).difference(loaded_key)
        print("Weights of {} are freeze but not initialized from pretrained model: {}".format(
            model.__class__.__name__, sorted(freeze_but_notload)))
    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))
        
def load_bidir_weights(model, args, freeze_list=None):
    if args.vmae_finetune.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.vmae_finetune, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(args.vmae_finetune, map_location='cpu')

    print("Load VideoMAE ckpt from %s" % args.vmae_finetune)
    checkpoint_model = None
    clip_checkpoint = torch.jit.load(args.clip_finetune, map_location='cpu')
    print("Load CLIP ckpt from %s" % args.clip_finetune)
    checkpoint_clip = clip_checkpoint.visual.state_dict()
    for model_key in args.model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model:
            if getattr(args, 'videomae_v2', False) or checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

    all_keys = list(checkpoint_model.keys())
    clip_all_keys = list(checkpoint_clip.keys())
    new_dict = OrderedDict()
    if getattr(args, 'videomae_v2', False):
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('fc_norm.'):
                new_dict[key.replace('fc_norm', 'vmae_fc_norm')] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
    else:
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.norm.'):
                new_dict[key.replace('encoder.', 'vmae_fc_')] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
    
    # add new code for load clip weight <blocks load, for Video Encoder>
    for key in clip_all_keys:
        if key.startswith('transformer.'):
            if key[23] == '.':
                new_dict['blocks.'+ key[22] + '.clip_' + key[24:]] = checkpoint_clip[key]
            else : # layer10 ~ 11 process
                new_dict['blocks.'+ key[22:24] + '.clip_' + key[25:]] = checkpoint_clip[key]
        else:
            new_dict['clip_' + key] = checkpoint_clip[key]
    
    new_dict['clip_noun_proj'] = checkpoint_clip['proj']
    new_dict['clip_verb_proj'] = checkpoint_clip['proj']
    new_dict['clip_ov_verb_proj'] = checkpoint_clip['proj']
    
    # add new code for load clip weight <blocks load, for Text Encoder>
    if args.text_finetune is not None:
        lavila = torch.load(args.text_finetune, map_location='cpu')
        checkpoint_clip = lavila['state_dict']
        for key in checkpoint_clip:
            if key.startswith('module.transformer.'):
                if key[30] == '.':
                    new_dict['text_blocks.'+ key[29] + '.clip_text_' + key[31:]] = checkpoint_clip[key]
                    new_dict['blocks.'+ key[29] + '.clip_text_' + key[31:]] = checkpoint_clip[key]
                else : # layer10 ~ 11 process
                    new_dict['text_blocks.'+ key[29:31] + '.clip_text_' + key[32:]] = checkpoint_clip[key]
                    new_dict['blocks.'+ key[29:31] + '.clip_text_' + key[32:]] = checkpoint_clip[key]
            elif not key.startswith('module.visual.'):
                new_dict['clip_text_' + key[7:]] = checkpoint_clip[key]
        new_dict['clip_text_text_projection'] = checkpoint_clip['module.text_projection']
    elif args.audio_finetune is not None:
        ckeckpoint = torch.load(args.audio_finetune, map_location='cpu')
        ckeckpoint = ckeckpoint['model']
        for key in ckeckpoint:
            if key.startswith('encoder.layers.'):
                new_dict['blocks.'+ key[15:]] = ckeckpoint[key]
                # if key[14] == '.' and key[16] == '.':
                #     new_dict['blocks.'+ key[15] + '.' + key[17:]] = ckeckpoint[key]
                # elif key[14] == '.' and key[17] == '.':
                #     new_dict['blocks.'+ key[15:17] + '.' + key[18:]] = ckeckpoint[key]
            elif key.startswith('encoder.layer_norm.'):
                new_dict[key[8:18]+'_first'+key[18:]] = ckeckpoint[key]
            elif key.startswith('encoder.pos_conv'):
                new_dict[key[8:]] = ckeckpoint[key]
            else:
                new_dict[key] = ckeckpoint[key]
    elif args.audio_path is not None:
        for key in clip_all_keys:
            if key.startswith('transformer.'):
                if key[23] == '.':
                    new_dict['text_blocks.'+ key[22] + '.clip_text_' + key[24:]] = checkpoint_clip[key]
                    new_dict['blocks.'+ key[22] + '.clip_text_' + key[24:]] = checkpoint_clip[key]
                else : # layer10 ~ 11 process
                    new_dict['text_blocks.'+ key[22:24] + '.clip_text_' + key[25:]] = checkpoint_clip[key]
                    new_dict['blocks.'+ key[22:24] + '.clip_text_' + key[25:]] = checkpoint_clip[key]
            else:
                new_dict['clip_text_' + key] = checkpoint_clip[key]
        new_dict['audio_ln_post.weight'] = checkpoint_clip['ln_post.weight']
        new_dict['audio_ln_post.bias'] = checkpoint_clip['ln_post.bias']
    else:
        checkpoint_clip = clip_checkpoint.state_dict()
        for key in checkpoint_clip:
            if key.startswith('transformer.'):
                if key[23] == '.':
                    new_dict['text_blocks.'+ key[22] + '.clip_text_' + key[24:]] = checkpoint_clip[key]
                    new_dict['blocks.'+ key[22] + '.clip_text_' + key[24:]] = checkpoint_clip[key]
                else : # layer10 ~ 11 process
                    new_dict['text_blocks.'+ key[22:24] + '.clip_text_' + key[25:]] = checkpoint_clip[key]
                    new_dict['blocks.'+ key[22:24] + '.clip_text_' + key[25:]] = checkpoint_clip[key]
            elif not key.startswith('visual.'):
                new_dict['clip_text_' + key] = checkpoint_clip[key]
                
    if args.prompt_weight is not None:
        prompt = torch.load(args.prompt_weight, map_location='cpu')
        keys = prompt['module']
        new_dict['prompt_embedding.weight'] = keys['prompt_embedding.weight']
    
    if getattr(args, 'ast_finetune', None) is not None:
        ast_key = torch.load(args.ast_finetune, map_location='cpu')
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in ast_key:
                del ast_key[k]
        for key, value in ast_key.items():
            if key.startswith('module.v.'):
                if key.startswith('module.v.blocks.'):
                    new_key = key.replace('module.v.', '')
                    if new_key[8] == '.':
                        new_dict[new_key[:9] + 'ast_' + new_key[9:]] = value
                    else:
                        new_dict[new_key[:10] + 'ast_' + new_key[10:]] = value
                else:
                    new_key = key.replace('module.v.', 'ast_')
                    new_dict[new_key] = value
                    
        f_dim, t_dim = model.get_shape(args.stride, args.stride, args.audio_height, args.audio_width)
        num_patches = f_dim * t_dim
        p_f_dim, p_t_dim = model.get_shape(args.stride, args.stride, 128, 1024)
        p_num_patches = p_f_dim * p_t_dim
        original_embedding_dim = ast_key['module.v.pos_embed'].shape[-1]

        new_pos_embed = ast_key['module.v.pos_embed'][:, 2:, :].detach().reshape(1, p_num_patches, original_embedding_dim).transpose(1, 2).reshape(1, original_embedding_dim, p_f_dim, p_t_dim)
        # if the input sequence length is larger than the original audioset (10s), then cut the positional embedding
        if t_dim < p_t_dim:
            new_pos_embed = new_pos_embed[:, :, :, int(p_t_dim/2) - int(t_dim / 2): int(p_t_dim/2) - int(t_dim / 2) + t_dim]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(8, t_dim), mode='bilinear')
        if f_dim < p_f_dim:
            new_pos_embed = new_pos_embed[:, :, int(p_f_dim/2) - int(f_dim / 2): int(p_f_dim/2) - int(f_dim / 2) + t_dim, :]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
        new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
        new_dict['ast_pos_embed'] = nn.Parameter(torch.cat([ast_key['module.v.pos_embed'][:, :2, :].detach(), new_pos_embed], dim=1))
                    
    
    if getattr(args, 'audio_only_finetune', None) is not None:
        audio_key = torch.load(args.audio_only_finetune, map_location='cpu')['module']
        for k in ['head.weight', 'head.bias', 'head_noun.weight', 'head_noun.bias', 'head_verb.weight', 'head_verb.bias', 'audio_ln_post.weight' 'audio_ln_post.bias']:
            if k in audio_key:
                del audio_key[k]
        for key, value in audio_key.items():
            new_key = key.replace('clip', 'audio')
            new_dict[new_key] = value
    
    if getattr(args, 'enable_audio_stride', None):
        new_dict['audio_conv1.weight'] = new_dict['clip_conv1.weight']
            
    # load로 불러온 pre-trained weight를 new_dict에 담아주고
    checkpoint_model = new_dict

    # interpolate position embedding
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
        num_patches = model.patch_embed.num_patches # 
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1

        # height (== width) for the checkpoint position embedding
        orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(args.num_frames // model.patch_embed.tubelet_size)) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int((num_patches // (args.num_frames // model.patch_embed.tubelet_size) )** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # B, L, C -> BT, H, W, C -> BT, C, H, W
            pos_tokens = pos_tokens.reshape(-1, args.num_frames // model.patch_embed.tubelet_size, orig_size, orig_size, embedding_size)
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, args.num_frames // model.patch_embed.tubelet_size, new_size, new_size, embedding_size) 
            pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

    load_state_dict(model, checkpoint_model, prefix=args.model_prefix,freeze_list=freeze_list)
    
    # with torch.no_grad():#! module_layers에 들어가는 것만 한다.
    #     for i in range(12):
    #         model.blocks[i].time_attn.out_proj.weight.copy_(model.blocks[i].clip_attn.out_proj.weight)
    #         model.blocks[i].time_attn.out_proj.bias.copy_(model.blocks[i].clip_attn.out_proj.bias)
    #         model.blocks[i].time_attn.in_proj_weight.copy_(model.blocks[i].clip_attn.in_proj_weight)
    #         model.blocks[i].time_attn.in_proj_bias.copy_(model.blocks[i].clip_attn.in_proj_bias)
    #     print("copy attn layer")
        
def load_clip_weights(model,load_path: str) -> Dict[str, torch.Tensor]:
    clip_model = torch.jit.load(load_path, map_location='cpu')
    clip_model = clip_model.visual
    src_state_dict = clip_model.state_dict()
    src_state_dict = dict((k, v.float()) for k, v in src_state_dict.items())

    dst_state_dict = {}
    
    dst_state_dict['cls_token'] = src_state_dict['class_embedding']
    dst_state_dict['pos_embed'] = src_state_dict['positional_embedding']
    dst_state_dict['patch_embed.proj.weight'] = src_state_dict['conv1.weight'].flatten(1)
    dst_state_dict['patch_embed.proj.bias'] = torch.zeros([src_state_dict['conv1.weight'].size(0)])
    
    dst_state_dict['ln_pre.weight'] = src_state_dict['ln_pre.weight']
    dst_state_dict['ln_pre.bias'] = src_state_dict['ln_pre.bias']
    
    dst_state_dict['ln_post.weight'] = src_state_dict['ln_post.weight']
    dst_state_dict['ln_post.bias'] = src_state_dict['ln_post.bias']

    block_idx = 0
    while True:
        src_prefix = 'transformer.resblocks.%d.' % block_idx
        dst_prefix = 'blocks.%d.' % block_idx

        src_block_state_dict = dict((k[len(src_prefix):], v) for k, v in src_state_dict.items() if k.startswith(src_prefix))
        if len(src_block_state_dict) == 0:
            break

        dst_block_state_dict = {}
        feat_dim = src_block_state_dict['ln_1.weight'].size(0)

        for i, dst_name in enumerate(('q', 'k', 'v')):
            dst_block_state_dict['attn.%s_proj.weight' % dst_name] = src_block_state_dict['attn.in_proj_weight'][feat_dim * i: feat_dim * (i + 1)]
            dst_block_state_dict['attn.%s_proj.bias' % dst_name] = src_block_state_dict['attn.in_proj_bias'][feat_dim * i: feat_dim * (i + 1)]
        
        dst_block_state_dict['attn.out_proj.weight'] = src_block_state_dict['attn.out_proj.weight']
        dst_block_state_dict['attn.out_proj.bias'] = src_block_state_dict['attn.out_proj.bias']

        dst_block_state_dict['mlp.fc1.weight'] = src_block_state_dict['mlp.c_fc.weight']
        dst_block_state_dict['mlp.fc1.bias'] = src_block_state_dict['mlp.c_fc.bias']
        dst_block_state_dict['mlp.fc2.weight'] = src_block_state_dict['mlp.c_proj.weight']
        dst_block_state_dict['mlp.fc2.bias'] = src_block_state_dict['mlp.c_proj.bias']

        dst_block_state_dict['norm1.weight'] = src_block_state_dict['ln_1.weight']
        dst_block_state_dict['norm1.bias'] = src_block_state_dict['ln_1.bias']
        dst_block_state_dict['norm2.weight'] = src_block_state_dict['ln_2.weight']
        dst_block_state_dict['norm2.bias'] = src_block_state_dict['ln_2.bias']

        dst_state_dict.update(dict((dst_prefix + k, v) for k, v in dst_block_state_dict.items()))
        block_idx += 1
    
    load_state_dict(model, dst_state_dict)      

        
def laod_vmae_weights(model, pre_trained_weight, args):
    checkpoint = torch.load(pre_trained_weight, map_location='cpu')
    print("Load ckpt from %s" % pre_trained_weight)
    for model_key in args.model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    for key in all_keys:
        if key.startswith('backbone.'):
            new_dict[key[9:]] = checkpoint_model[key]
        elif key.startswith('encoder.'):
            new_dict[key[8:]] = checkpoint_model[key]
        else:
            new_dict[key] = checkpoint_model[key]
    checkpoint_model = new_dict
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
        num_patches = model.patch_embed.num_patches # 
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1

        # height (== width) for the checkpoint position embedding 
        orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(args.num_frames // model.patch_embed.tubelet_size)) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int((num_patches // (args.num_frames // model.patch_embed.tubelet_size) )** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # B, L, C -> BT, H, W, C -> BT, C, H, W
            pos_tokens = pos_tokens.reshape(-1, args.num_frames // model.patch_embed.tubelet_size, orig_size, orig_size, embedding_size)
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, args.num_frames // model.patch_embed.tubelet_size, new_size, new_size, embedding_size) 
            pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
            
    load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
    
def laod_eval_weights(model, pre_trained_weight, args):
    checkpoint = torch.load(pre_trained_weight, map_location='cpu')
    print("Load ckpt from %s" % pre_trained_weight)
    for model_key in args.model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    if any(key.startswith('clipmodel.') for key in state_dict.keys()):
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]
            elif key.startswith('module.'):
                new_dict[key[7:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
    else:
        for key in all_keys:
            if key.startswith('clipmodel.'):
                if key.startswith('clipmodel.transformer.'):
                    if key[23] == '.':
                        new_dict['text_blocks.'+ key[32] + '.clip_text_' + key[34:]] = checkpoint_model[key]
                    else : # layer10 ~ 11 process
                        new_dict['text_blocks.'+ key[32:34] + '.clip_text_' + key[35:]] = checkpoint_model[key]
                elif not key.startswith('clipmodel.visual.'):
                    new_dict['clip_text_' + key] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
                
    checkpoint_model = new_dict
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
        num_patches = model.patch_embed.num_patches # 
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1

        # height (== width) for the checkpoint position embedding 
        orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(args.num_frames // model.patch_embed.tubelet_size)) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int((num_patches // (args.num_frames // model.patch_embed.tubelet_size) )** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # B, L, C -> BT, H, W, C -> BT, C, H, W
            pos_tokens = pos_tokens.reshape(-1, args.num_frames // model.patch_embed.tubelet_size, orig_size, orig_size, embedding_size)
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, args.num_frames // model.patch_embed.tubelet_size, new_size, new_size, embedding_size) 
            pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
            
    model_state_dict = model.state_dict()
    for key in list(checkpoint_model.keys()):
        # Check for the temporal positional embedding parameters in CrossAttentionT2S
        if 't2s_cross.clip_time_pos' in key or 't2s_cross.vmae_time_pos' in key:
            param_checkpoint = checkpoint_model[key]
            param_model = model_state_dict[key]

            # If the temporal length is different, perform interpolation
            if param_checkpoint.shape[0] != param_model.shape[0]:
                
                # Reshape for 1D interpolation: (L, C) -> (B, C, L)
                param_to_interp = param_checkpoint.unsqueeze(0).transpose(1, 2)
                
                # Perform linear interpolation
                interpolated_param = F.interpolate(
                    param_to_interp,
                    size=param_model.shape[0],  # Target temporal length
                    mode='linear',
                    align_corners=False
                )
                
                # Reshape back: (B, C, L) -> (L, C)
                final_param = interpolated_param.transpose(1, 2).squeeze(0)
                
                # Update the parameter in the checkpoint dictionary
                checkpoint_model[key] = final_param
        
            
    load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
            

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, real_epoch=None):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    epoch = real_epoch if real_epoch is not None else epoch
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    output_dir = Path(args.output_dir)
    if loss_scaler is not None:
        # torch.amp
        if args.auto_resume and len(args.resume) == 0:
            import glob
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            print("Auto resume checkpoint: %s" % args.resume)

        if args.resume:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
            print("Resume checkpoint %s" % args.resume)
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = checkpoint['epoch'] + 1
                if hasattr(args, 'model_ema') and args.model_ema:
                    _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")
    else:
        # deepspeed, only support '--auto_resume'.
        if args.auto_resume:
            import glob
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-best')
                _, client_states = model.load_checkpoint(args.output_dir, tag='checkpoint-best')
                if client_states['epoch'] == 'best' or int(client_states['epoch']) < latest_ckpt:
                    args.resume = os.path.join(output_dir, 'checkpoint-%d' % latest_ckpt)
                    _, client_states = model.load_checkpoint(args.output_dir, tag='checkpoint-%d' % latest_ckpt)
                    print("Auto resume checkpoint: %d" % latest_ckpt)
                else:
                    print("Auto resume checkpoint: %d from best ckpt" % int(client_states['epoch']))
                args.start_epoch = client_states['epoch'] + 1
                if model_ema is not None:
                    if args.model_ema:
                        _load_checkpoint_for_ema(model_ema, client_states['model_ema'])


def create_ds_config(args):
    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * args.update_freq * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                    "betas": [
                        0.9,
                        0.999
                    ],
                    "eps": 1e-8
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 7,
                "loss_scale_window": 128
            }
        }

        writer.write(json.dumps(ds_config, indent=2))
        
def audio_collate_fn(batch):
    """
    Collate function for audio data.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    inputs, labels, video_idx, audios, caption, idx = zip(*batch)
    inputs, labels, video_idx, audios, caption, idx = list(inputs), list(labels), list(video_idx), list(audios), list(caption), list(idx)
    caption = None if all(cap is None for cap in caption) else caption
    inputs, labels, video_idx, audios, idx = (
        default_collate(inputs),
        default_collate(labels),
        default_collate(video_idx),
        default_collate(audios),
        default_collate(idx),
    )
    
    return inputs, labels, video_idx, audios, caption, idx

def test_audio_collate_fn(batch):
    inputs, labels, video_idx, chunk_nb, split_nb, audios, caption, idx = zip(*batch)
    inputs, labels, video_idx, chunk_nb, split_nb, audios, caption, idx = list(inputs), list(labels), list(video_idx), list(chunk_nb), list(split_nb), list(audios), list(caption), list(idx)
    caption = None if all(cap is None for cap in caption) else caption
    inputs, labels, video_idx, chunk_nb, split_nb, audios, idx = (
        default_collate(inputs),
        default_collate(labels),
        default_collate(video_idx),
        default_collate(chunk_nb),
        default_collate(split_nb),
        default_collate(audios),
        default_collate(idx),
    )
    
    return inputs, labels, video_idx, chunk_nb, split_nb, audios, caption, idx

def audio_list_collate_fn(batch):
    """
    Collate function for audio data.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    inputs, labels, video_idx, audios, caption, idx = zip(*batch)
    inputs, labels, video_idx, audios, caption, idx = list(inputs), list(labels), list(video_idx), list(audios), list(caption), list(idx)
    caption = None if all(cap is None for cap in caption) else caption
    inputs, labels, video_idx, idx = (
        default_collate(inputs),
        default_collate(labels),
        default_collate(video_idx),
        default_collate(idx),
    )
    
    return inputs, labels, video_idx, audios, caption

def test_audio_list_collate_fn(batch):
    inputs, labels, video_idx, chunk_nb, split_nb, audios, caption, idx = zip(*batch)
    inputs, labels, video_idx, chunk_nb, split_nb, audios, caption, idx = list(inputs), list(labels), list(video_idx), list(chunk_nb), list(split_nb), list(audios), list(caption), list(idx)
    caption = None if all(cap is None for cap in caption) else caption
    inputs, labels, video_idx, chunk_nb, split_nb, idx = (
        default_collate(inputs),
        default_collate(labels),
        default_collate(video_idx),
        default_collate(chunk_nb),
        default_collate(split_nb),
        default_collate(idx),
    )
    
    return inputs, labels, video_idx, chunk_nb, split_nb, audios, caption

def multiple_samples_collate(batch, fold=False):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    inputs, labels, video_idx, extra_data = zip(*batch)
    inputs = [item for sublist in inputs for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    video_idx = [item for sublist in video_idx for item in sublist]
    inputs, labels, video_idx, extra_data = (
        default_collate(inputs),
        default_collate(labels),
        default_collate(video_idx),
        default_collate(extra_data),
    )
    if fold:
        return [inputs], labels, video_idx, extra_data
    else:
        return inputs, labels, video_idx, extra_data

def cross_multiple_samples_collate(batch, fold=False):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    s_inputs, t_inputs, labels, video_idx, extra_data = zip(*batch)
    s_inputs = [item for item in s_inputs for i in range(2)] # sample을 2개씩 sampling하니까 range2로 반복시켜줬다. 나중에 num_sample숫자에 맞춰 코드 짜줘야 한다.
    t_inputs = [item for sublist in t_inputs for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    video_idx = [item for sublist in video_idx for item in sublist]
    s_inputs, t_inputs, labels, video_idx, extra_data = (
        default_collate(s_inputs),
        default_collate(t_inputs),
        default_collate(labels),
        default_collate(video_idx),
        default_collate(extra_data),
    )
    if fold:
        return [s_inputs, t_inputs], labels, video_idx, extra_data
    else:
        return s_inputs, t_inputs, labels, video_idx, extra_data
    
def freeze_block(model,block_list):
    freeze_list = []
    for name, param in model.named_parameters():
        for block in block_list:#if block in block_list
            if block in name:
                param.requires_grad = False
                freeze_list.append(name)
                break
            else:
                param.requires_grad = True
    return model, freeze_list

def unfreeze_block(model, block_list):
    unfreeze_list = []
    for name, param in model.named_parameters():
        for block in block_list:#if block in block_list
            if block in name or 'ALLIN' in block_list:
                param.requires_grad = True
                unfreeze_list.append(name)
                break
            else:
                param.requires_grad = False
    return model, unfreeze_list

def freeze_block_list(model, block_list):
    freeze_list = []
    for name, param in model.named_parameters():
        if 'ALLIN' in block_list:
            freeze_list.append(name)
            continue
        should_freeze = True 
        for block in block_list:
            if block in name:
                should_freeze = False
                break
        if should_freeze:
            freeze_list.append(name)
    return freeze_list

def notice_message(token, channel, text, attachments):
    attachments = json.dumps(attachments) # 리스트는 Json 으로 덤핑 시켜야 Slack한테 제대로 간다.
    response = requests.post("https://slack.com/api/chat.postMessage",
        headers={"Authorization": "Bearer "+token},
        data={"channel": channel, "text": text ,"attachments": attachments})