import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def lora_moved(save_weight):
    layer_move = []
    for name, params in save_weight.items():
        if 'lora.up' in name:
            lora_up = params
            lora_down = save_weight[name.replace('lora.up', 'lora.down')]
            weight = lora_up.squeeze() @ lora_down.squeeze()
            dist = weight.flatten().abs().mean().item()
            layer_move.append(dist)
    return sum(layer_move) / len(layer_move)


class LoRALinearLayer(nn.Module):

    def __init__(self, name, original_module, rank=4, alpha=1):
        super().__init__()

        self.name = name
        self.class_name = original_module.__class__.__name__

        if original_module.__class__.__name__ == 'Conv2d':
            in_channels, out_channels = original_module.in_channels, original_module.out_channels
            self.stride = original_module.stride
            self.padding = original_module.padding
            self.dilation = original_module.dilation
        else:
            in_channels, out_channels = original_module.in_features, original_module.out_features
        self.down = torch.nn.Parameter(torch.zeros(rank, in_channels))
        self.up = torch.nn.Parameter(torch.zeros(out_channels, rank))

        self.register_buffer('alpha', torch.tensor(alpha))

        nn.init.normal_(self.down, std=1 / rank)
        nn.init.zeros_(self.up)

        # self.original_forward = original_module.forward
        self.original_weights = torch.nn.Parameter(original_module.weight.detach(), requires_grad=False)
        self.original_bias = torch.nn.Parameter(original_module.bias.detach(), requires_grad=False) if original_module.bias is not None else None
        original_module.forward = self.forward

        self.enable_drop = False

    def forward(self, hidden_states):
        if self.enable_drop and self.training:
            drop_mul = 0
        else:
            drop_mul = 1
        if self.class_name == 'Conv2d':
            new_weights = self.original_weights + (drop_mul * self.alpha * self.up @ self.down).reshape(self.up.shape[0], self.down.shape[1], 1, 1)
            hidden_states = F.conv2d(hidden_states, new_weights, self.original_bias, stride=self.stride, padding=self.padding, dilation=self.dilation)
        else:
            new_weights = self.original_weights + (drop_mul * self.alpha * self.up @ self.down)
            hidden_states = F.linear(hidden_states, new_weights, self.original_bias)
        return hidden_states


class LoRSALinearLayer(nn.Module):

    def __init__(self, name, original_module, rank=4, alpha=1):
        super().__init__()

        self.name = name
        self.class_name = original_module.__class__.__name__

        if original_module.__class__.__name__ == 'Conv2d':
            in_channels, out_channels = original_module.in_channels, original_module.out_channels
            self.stride = original_module.stride
            self.padding = original_module.padding
            self.dilation = original_module.dilation
        else:
            in_channels, out_channels = original_module.in_features, original_module.out_features
        self.down = torch.nn.Parameter(torch.zeros(rank, in_channels))
        self.up = torch.nn.Parameter(torch.zeros(out_channels, rank))
        self.sparse = torch.nn.Parameter(torch.zeros(out_channels, in_channels))

        self.register_buffer('alpha', torch.tensor(alpha))

        nn.init.normal_(self.down, std=1 / rank)
        nn.init.zeros_(self.up)
        nn.init.zeros_(self.sparse)

        # self.original_forward = original_module.forward)
        self.original_weights = torch.nn.Parameter(original_module.weight.detach(), requires_grad=False)
        self.original_bias = torch.nn.Parameter(original_module.bias.detach(), requires_grad=False) if original_module.bias is not None else None
        original_module.forward = self.forward

        self.enable_drop = False

    def forward(self, hidden_states):
        if self.enable_drop and self.training:
            drop_mul = 0
        else:
            drop_mul = 1
        if self.class_name == 'Conv2d':
            new_weights = self.original_weights + (drop_mul * self.alpha * (self.up @ self.down + self.sparse)).reshape(self.up.shape[0], self.down.shape[1], 1, 1)
            hidden_states = F.conv2d(hidden_states, new_weights, self.original_bias, stride=self.stride, padding=self.padding, dilation=self.dilation)
        else:
            new_weights = self.original_weights + (drop_mul * self.alpha * (self.up @ self.down + self.sparse))
            hidden_states = F.linear(hidden_states, new_weights, self.original_bias)
        return hidden_states
