from models.med import BertConfig, BertModel, BertLMHeadModel
from models.blip import create_vit, init_tokenizer, load_checkpoint
from models.blip_vqa import BLIP_VQA
from models.vit import interpolate_pos_embed

import copy
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np

class BLIP_VQA_LLF(nn.Module):
    def __init__(self,
                 net,
                 pre_split_reset_factor,
                 post_split_reset_factor,
                 reset_split_layer
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.pre_split_reset_factor = pre_split_reset_factor
        self.post_split_reset_factor = post_split_reset_factor
        self.reset_split_layer = reset_split_layer

        self.net = net
        self.net_init_state = None

    def save_wts_for_reset(self):
        self.net_init_state = copy.deepcopy(self.net.state_dict())

    def reset_params(self):
        state_dict = self.net.state_dict()
        post_split = False
        for (name, param), init_param in zip(state_dict.items(), self.net_init_state.values()):
            if self.reset_split_layer in name:
                post_split = True
            if not post_split:
                reset_factor = self.pre_split_reset_factor
            else:
                reset_factor = self.post_split_reset_factor
            if reset_factor < 1.0:
                param.copy_(
                    ((1.0 - reset_factor) * init_param.to(param)) + (reset_factor * param)
                )

        if self.reset_split_layer is not None and not post_split:
            raise ValueError(
                f"reset_split_layer value {self.reset_split_layer} is not a valid "
                "parameter name in the model"
            )

    def reset_params_module(self, reset_split_layer, module):
        state_dict = self.net.state_dict()
        post_split = False
        for (name, param), init_param in zip(state_dict.items(), self.net_init_state.values()):
            if module not in name:
                continue

            if reset_split_layer in name:
                post_split = True
            if not post_split:
                reset_factor = self.pre_split_reset_factor
            else:
                reset_factor = self.post_split_reset_factor
            if reset_factor < 1.0:
                param.copy_(
                    ((1.0 - reset_factor) * init_param.to(param)) + (reset_factor * param)
                )

        if reset_split_layer is not None and not post_split:
            raise ValueError(
                f"reset_split_layer value {reset_split_layer} is not a valid "
                "parameter name in the model"
            )

    def reset_params_all_modules(self):
        if isinstance(self.reset_split_layer, dict):
            for module in ['visual_encoder', 'text_encoder', 'text_decoder']:
                if self.reset_split_layer[module] is not None:
                    self.reset_params_module(self.reset_split_layer[module], module)
        else:
            self.reset_params()

    def forward(self, image, question, answer=None, n=None, weights=None, train=True, inference='rank', k_test=128):
        return self.net(image, question, answer, n, weights, train, inference, k_test)


def blip_vqa_llf(resume_ckpt='', **kwargs):
    model = BLIP_VQA_LLF(**kwargs)

    if not resume_ckpt:
        return model

    checkpoint = torch.load(resume_ckpt, map_location='cpu')
    state_dict = checkpoint['model']

    state_dict['net.visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['net.visual_encoder.pos_embed'], model.net.visual_encoder)
    if 'net.visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['net.visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['net.visual_encoder_m.pos_embed'],
                                                                         model.net.visual_encoder_m)

    # TODO: see if this is necessary
    # for key in model.state_dict().keys():
    #     if key in state_dict.keys():
    #         if state_dict[key].shape!=model.state_dict()[key].shape:
    #             del state_dict[key]

    msg = model.load_state_dict(state_dict, strict=True)
    print('load checkpoint from %s'%resume_ckpt)
    return model


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))
