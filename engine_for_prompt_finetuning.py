import os
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
from util_tools.mixup import Mixup
from timm.utils import accuracy, ModelEma
import util_tools.utils as utils
from scipy.special import softmax
from einops import rearrange, repeat
import random


def train_class_batch(model, samples, target, criterion, textlist, device, textFeature = None):
    featnorm = 1
    randContrast = False
    
    if randContrast:
        numContrast = 300
        
        # sample positive and negative
        target = [textlist[i] for i in target]
        uniqname = np.unique(target)
        numNeg = numContrast - len(uniqname)
        complement = list(set(textlist) - set(uniqname))
        inp_list = uniqname.tolist() + random.sample(complement, min(numNeg, len(complement)))
        targets = torch.tensor([inp_list.index(n) for n in target]).to(device)
    else:
        targets = target
    
    if textFeature is not None:
        outputs_video = model(samples)
    else:
        outputs_video, textFeature = model(samples, textlist)
        
    if featnorm:
        outputs_video = outputs_video / outputs_video.norm(dim=-1, keepdim=True)
        textFeature = textFeature / textFeature.norm(dim=-1, keepdim=True)
        logits = outputs_video @ textFeature.t() / 0.07 if textFeature.dim() == 2 else torch.bmm(textFeature, outputs_video.unsqueeze(-1)).squeeze(-1) / 0.07
    else:
        logits = outputs_video @ textFeature.t() if textFeature.dim() == 2 else torch.bmm(textFeature, outputs_video.unsqueeze(-1)).squeeze(-1)
        
    loss = criterion(logits, targets)
    return loss, outputs_video, logits


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale

def train_one_epoch(args, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, class_list=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    textlist, textdict, texttoken = class_list
    if not args.kd:
        textdict = None
    
    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()
    for data_iter_step, (samples, targets, _, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = samples.shape[0]
        target = targets
        
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            samples = samples.half()
            loss, output, logits = train_class_batch(
                model, samples, targets, criterion, textlist, args.device, textFeature = textdict)
        else:
            with torch.cuda.amp.autocast():
                samples = samples.half() 
                loss, output = train_class_batch(
                    model, samples, targets, criterion)

        loss_value = loss.item()
        
        top1_acc, top5_acc = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if mixup_fn is None:
            pass
            class_acc = None
            # class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)
        metric_logger.update(acc1=top1_acc.item())
        metric_logger.update(acc5=top5_acc.item())
        metric_logger.meters['acc1'].update(top1_acc.item(), n=batch_size)
        metric_logger.meters['acc5'].update(top5_acc.item(), n=batch_size)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(args, data_loader, model, device, class_list):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'
    
    # prompt setting
    textlist, textdict, texttoken = class_list
    featnorm = 1

    # switch to evaluation mode
    model.eval()
    for idx, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        videos = batch[0]
        target = batch[1]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # compute output
        with torch.cuda.amp.autocast():
            if idx == 0:
                outputs_video, textFeature = model(videos, textlist)
            else:
                outputs_video, _ = model(videos, textlist)
                
            if featnorm:
                outputs_video = outputs_video / outputs_video.norm(dim=-1, keepdim=True)
                textFeature = textFeature / textFeature.norm(dim=-1, keepdim=True)
                logits = outputs_video @ textFeature.t() / 0.07 if textFeature.dim() == 2 else torch.bmm(textFeature, outputs_video.unsqueeze(-1)).squeeze(-1) / 0.07
            else:
                logits = outputs_video @ textFeature.t() if textFeature.dim() == 2 else torch.bmm(textFeature, outputs_video.unsqueeze(-1)).squeeze(-1)
        
            loss = criterion(logits, target)

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.update(acc1=acc1.item())
        metric_logger.update(acc5=acc5.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def speedup_one_epoch(args,data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    import time 
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # metric_logger = utils.MetricLogger(delimiter="  ")
    # header = 'Val:'
    IMG_SIZE = 224
    x = np.random.randn(args.batch_size, 3,args.num_frames, IMG_SIZE, IMG_SIZE).astype(np.float32)
    y = np.random.randint(0, args.nb_classes, args.batch_size, dtype=np.int32)
    tx=torch.tensor(x,dtype=torch.float32).cuda(device)
    ty=torch.tensor(y,dtype=torch.long).cuda(device)
    ave_forward_throughput=[]
    # switch to evaluation mode
    model.eval()
    steps=100
    # ave_start=time.time()
    for i in range(steps):
        videos = tx
        target = ty
        videos = videos.to(device, non_blocking=True)
        batch_size = videos.shape[0]
        target = target.to(device, non_blocking=True)
        # compute output
        with torch.cuda.amp.autocast():
        # start=time.time()
            starter.record()
            output = model(videos)
            ender.record()
            # loss = criterion(output, target)
            # end=time.time()
            torch.cuda.synchronize()
            curr_time = 1e-3*starter.elapsed_time(ender)
            fwd_throughput= batch_size/(curr_time)
            print(fwd_throughput)
            ave_forward_throughput.append(fwd_throughput)
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # metric_logger.update(loss=loss.item())
        # metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    # ave_end=time.time()
    #print(end-start)
    # throughput = steps*batch_size/(ave_end-ave_start)
    ave_fwd_throughput=np.mean(ave_forward_throughput[2:])
    return ave_fwd_throughput

@torch.no_grad()
def final_test(args,data_loader, model, device, file, class_list):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    # prompt setting
    textlist, textdict, texttoken = class_list
    featnorm = 1

    # switch to evaluation mode
    model.eval()
    final_result = []
    
    for idx, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        videos = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            if idx == 0:
                outputs_video, textFeature = model(videos, textlist)
            else:
                outputs_video, _ = model(videos, textlist)
            if featnorm:
                outputs_video = outputs_video / outputs_video.norm(dim=-1, keepdim=True)
                textFeature = textFeature / textFeature.norm(dim=-1, keepdim=True)
                logits = outputs_video @ textFeature.t() / 0.07 if textFeature.dim() == 2 else torch.bmm(textFeature, outputs_video.unsqueeze(-1)).squeeze(-1) / 0.07
            else:
                logits = outputs_video @ textFeature.t() if textFeature.dim() == 2 else torch.bmm(textFeature, outputs_video.unsqueeze(-1)).squeeze(-1)
        
            loss = criterion(logits, target)

        for i in range(outputs_video.size(0)):
            string = "{} {} {} {} {}\n".format(ids[i], \
                                                str(logits.data[i].cpu().numpy().tolist()), \
                                                str(int(target[i].cpu().numpy())), \
                                                str(int(chunk_nb[i].cpu().numpy())), \
                                                str(int(split_nb[i].cpu().numpy())))
            final_result.append(string)

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.update(acc1=acc1.item())
        metric_logger.update(acc5=acc5.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(acc1, acc5))
        for line in final_result:
            f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks, return_result = False):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(line.split('[')[1].split(']')[0], dtype=float, sep=',')
            data = softmax(data)
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    final_top1 ,final_top5 = np.mean(top1), np.mean(top5)
    if return_result:
        pred = [x[0] for x in ans]
        label = [x[3] for x in ans]
        video_ids = [x[4] for x in ans]
        confidence = [x[5] for x in ans]
        return final_top1*100 ,final_top5*100, pred, label, video_ids, confidence
    return final_top1*100 ,final_top5*100

def compute_video(lst):
    i, video_id, data, label = lst
    video_ids = [x for x in video_id]
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    confidence = np.max(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, label, video_ids, confidence]
