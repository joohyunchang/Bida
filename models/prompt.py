from models import clip
import torch
import pandas as pd
import numpy as np
import os
from collections import OrderedDict
import json


def convert_to_token(xh):
    xh_id = clip.tokenize(xh).cpu().data.numpy()
    return xh_id


def text_prompt(dataset='HMDB51', data_path = None ,clipbackbone='ViT-B/16', device='cpu'):
    actionlist, actionprompt, actiontoken = [], {}, []
    numC = {'HMDB51-feature-30fps-center': 51,}

    # load the CLIP model
    clipmodel, _ = clip.load(clipbackbone, device=device, jit=False)
    for paramclip in clipmodel.parameters():
        paramclip.requires_grad = False

    # convert to token, will automatically padded to 77 with zeros
    if dataset == 'HMDB51-feature-30fps-center':
        meta = open("../data/HMDB51/HMDB51_action.list", 'rb')
        actionlist = meta.readlines()
        meta.close()
        actionlist = np.array([a.decode('utf-8').split('\n')[0] for a in actionlist])
        actiontoken = np.array([convert_to_token(a) for a in actionlist])
    # More datasets to be continued

    elif dataset == 'EPIC':
        noun_anno_path = os.path.join(data_path, 'epic100_noun_classes.csv')
        verb_anno_path = os.path.join(data_path, 'epic100_verb_classes.csv')
        noun_cleaned = pd.read_csv(noun_anno_path, header=None, delimiter=',')
        verb_cleaned = pd.read_csv(verb_anno_path, header=None, delimiter=',')
        nounlist = list(noun_cleaned.values[:, 0])
        verblist = list(verb_cleaned.values[:, 0])
        nountoken = np.array([convert_to_token(a) for a in nounlist])
        verbtoken = np.array([convert_to_token(a) for a in verblist])
    
        # query the vector from dictionary
        with torch.no_grad():
            nounembed = clipmodel.encode_text_light(torch.tensor(nountoken).to(device))
            verbembed = clipmodel.encode_text_light(torch.tensor(verbtoken).to(device))

        noundict = OrderedDict((nounlist[i], nounembed[i].cpu().data.numpy()) for i in range(300))
        verbdict = OrderedDict((verblist[i], verbembed[i].cpu().data.numpy()) for i in range(97))
        nountoken = OrderedDict((nounlist[i], nountoken[i]) for i in range(300))
        verbtoken = OrderedDict((verblist[i], verbtoken[i]) for i in range(97))

        del clipmodel
        torch.cuda.empty_cache()
        
        return [nounlist, noundict, nountoken, verblist, verbdict, verbtoken]
    
    elif dataset == 'diving-48':
        anno_path = os.path.join(data_path, 'class.csv')
        cleaned = pd.read_csv(anno_path, header=None, delimiter=',')
        # list_0 = list(cleaned.values[:, 0])
        # list_1 = list(cleaned.values[:, 1])
        # list_2 = list(cleaned.values[:, 2])
        # list_3 = list(cleaned.values[:, 3])
        actionlist = list(cleaned.values[:, 4])
        actiontoken = np.array([convert_to_token(a) for a in actionlist])
    
        # query the vector from dictionary
        with torch.no_grad():
            actionembed = clipmodel.encode_text_light(torch.tensor(actiontoken).to(device))

        actiondict = OrderedDict((actionlist[i], actionembed[i].cpu().data.numpy()) for i in range(300))
        actiontoken = OrderedDict((actionlist[i], actiontoken[i]) for i in range(300))

        del clipmodel
        torch.cuda.empty_cache()
        
        return [actionlist, actiondict, actiontoken]
    
    # query the vector from dictionary
    with torch.no_grad():
        actionembed = clipmodel.encode_text_light(torch.tensor(actiontoken).to(device))

    actiondict = OrderedDict((actionlist[i], actionembed[i].cpu().data.numpy()) for i in range(numC[dataset]))
    actiontoken = OrderedDict((actionlist[i], actiontoken[i]) for i in range(numC[dataset]))
    
    del clipmodel
    torch.cuda.empty_cache()
    
    return [actionlist, actiondict, actiontoken]