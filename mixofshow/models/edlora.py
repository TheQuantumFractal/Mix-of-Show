import math

import torch
import torch.nn as nn
from diffusers.models.attention_processor import AttnProcessor
from diffusers.utils.import_utils import is_xformers_available
import torch.nn.functional as F

if is_xformers_available():
    import xformers


def remove_edlora_unet_attention_forward(unet):
    def change_forward(unet):  # omit proceesor in new diffusers
        for name, layer in unet.named_children():
            if layer.__class__.__name__ == 'Attention' and name == 'attn2':
                layer.set_processor(AttnProcessor())
            else:
                change_forward(layer)
    change_forward(unet)


class EDLoRA_Control_AttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """
    def __init__(self, cross_attention_idx, place_in_unet, controller, attention_op=None):
        self.cross_attention_idx = cross_attention_idx
        self.place_in_unet = place_in_unet
        self.controller = controller
        self.attention_op = attention_op

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if encoder_hidden_states is None:
            is_cross = False
            encoder_hidden_states = hidden_states
        else:
            is_cross = True
            if len(encoder_hidden_states.shape) == 4:  # multi-layer embedding
                encoder_hidden_states = encoder_hidden_states[:, self.cross_attention_idx, ...]
            else:  # single layer embedding
                encoder_hidden_states = encoder_hidden_states

        assert not attn.norm_cross

        batch_size, sequence_length, _ = encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        if is_xformers_available() and not is_cross:
            hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
            hidden_states = hidden_states.to(query.dtype)
        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            attention_probs = self.controller(attention_probs, is_cross, self.place_in_unet)
            hidden_states = torch.bmm(attention_probs, value)

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class EDLoRA_AttnProcessor:
    def __init__(self, cross_attention_idx, attention_op=None):
        self.attention_op = attention_op
        self.cross_attention_idx = cross_attention_idx

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            if len(encoder_hidden_states.shape) == 4:  # multi-layer embedding
                encoder_hidden_states = encoder_hidden_states[:, self.cross_attention_idx, ...]
            else:  # single layer embedding
                encoder_hidden_states = encoder_hidden_states

        assert not attn.norm_cross

        batch_size, sequence_length, _ = encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        if is_xformers_available():
            hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=attention_mask)
            hidden_states = hidden_states.to(query.dtype)
        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def revise_edlora_unet_attention_forward(unet):
    def change_forward(unet, count):
        for name, layer in unet.named_children():
            if layer.__class__.__name__ == 'Attention' and 'attn2' in name:
                layer.set_processor(EDLoRA_AttnProcessor(count))
                count += 1
            else:
                count = change_forward(layer, count)
        return count

    # use this to ensure the order
    cross_attention_idx = change_forward(unet.down_blocks, 0)
    cross_attention_idx = change_forward(unet.mid_block, cross_attention_idx)
    cross_attention_idx = change_forward(unet.up_blocks, cross_attention_idx)
    print(f'Number of attention layer registered {cross_attention_idx}')


def revise_edlora_unet_attention_controller_forward(unet, controller):
    class DummyController:
        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def change_forward(unet, count, place_in_unet):
        for name, layer in unet.named_children():
            if layer.__class__.__name__ == 'Attention' and 'attn2' in name:  # only register controller for cross-attention
                layer.set_processor(EDLoRA_Control_AttnProcessor(count, place_in_unet, controller))
                count += 1
            else:
                count = change_forward(layer, count, place_in_unet)
        return count

    # use this to ensure the order
    cross_attention_idx = change_forward(unet.down_blocks, 0, 'down')
    cross_attention_idx = change_forward(unet.mid_block, cross_attention_idx, 'mid')
    cross_attention_idx = change_forward(unet.up_blocks, cross_attention_idx, 'up')
    print(f'Number of attention layer registered {cross_attention_idx}')
    controller.num_att_layers = cross_attention_idx


class LoRALinearLayer(nn.Module):
    def __init__(self, name, original_module, rank=4, alpha=1, lorsa_count=1, lorsa_val=None):
        super().__init__()

        self.name = name
        self.class_name = original_module.__class__.__name__
        self.lorsa_val = [0] if lorsa_val is None else lorsa_val

        if original_module.__class__.__name__ == 'Conv2d':
            in_channels, out_channels = original_module.in_channels, original_module.out_channels
            self.stride = original_module.stride
            self.padding = original_module.padding
            self.dilation = original_module.dilation
        else:
            in_channels, out_channels = original_module.in_features, original_module.out_features
        self.rank = rank
        self.lora_down = []
        self.lora_up = []
        self.lora_sparse = []
        for i in range(lorsa_count):
            if rank > 0:
                self.lora_down.append(torch.nn.Parameter(torch.zeros(rank, in_channels)))
                self.lora_up.append(torch.nn.Parameter(torch.zeros(out_channels, rank)))
                nn.init.normal_(self.lora_down[i], std=1 / rank)
                nn.init.zeros_(self.lora_up[i])
            self.lora_sparse.append(torch.nn.Parameter(torch.zeros(out_channels, in_channels)))
            nn.init.zeros_(self.lora_sparse[i])
        self.lora_down = nn.ParameterList(self.lora_down)
        self.lora_up = nn.ParameterList(self.lora_up)
        self.lora_sparse = nn.ParameterList(self.lora_sparse)

        self.register_buffer('alpha', torch.tensor(alpha))

        
        

        # self.original_forward = original_module.forward)
        self.original_weights = torch.nn.Parameter(original_module.weight.detach(), requires_grad=False)
        self.original_bias = torch.nn.Parameter(original_module.bias.detach(), requires_grad=False) if original_module.bias is not None else None
        original_module.forward = self.forward

        self.enable_drop = False

    def forward(self, hidden_states):
        lorsa_idx = self.lorsa_val[0]
        # print(lorsa_idx)
        if self.enable_drop and self.training:
            drop_mul = 0
        else:
            drop_mul = 1
        if self.class_name == 'Conv2d':
            new_weights = self.original_weights + (drop_mul * self.alpha * (self.lora_up[lorsa_idx] @ self.lora_down[lorsa_idx] + self.lora_sparse[lorsa_idx])).reshape(self.up.shape[0], self.down.shape[1], 1, 1)
            hidden_states = F.conv2d(hidden_states, new_weights, self.original_bias, stride=self.stride, padding=self.padding, dilation=self.dilation)
        else:
            if self.rank > 0:
                new_weights = self.original_weights + (drop_mul * self.alpha * (self.lora_up[lorsa_idx] @ self.lora_down[lorsa_idx] + self.lora_sparse[lorsa_idx]))
            else:
                new_weights = self.original_weights + (drop_mul * self.alpha * self.lora_sparse[lorsa_idx])
            hidden_states = F.linear(hidden_states, new_weights, self.original_bias)
        return hidden_states
