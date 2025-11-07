from __future__ import absolute_import, division
"""
OSNet IBN x1.0 — ARQUITETURA COMPLETA para INFERÊNCIA (versão limpa)

- Mantém exatamente os blocos e a topologia do OSNet original (ICCV'19)
- Remove dependências externas (ex.: gdown) e código de treinamento/visualização
- Compatível com pesos .pth oficiais (ex.: osnet_ibn_x1_0_imagenet.pth)
- Foco: uso em produção para extração de embeddings (eval mode)

Notas de uso:
    • Instancie via os construtores abaixo (ex.: osnet_ibn_x1_0())
    • Carregue seus pesos manualmente (model.load_state_dict(...))
    • Em eval(), o forward retorna diretamente o vetor de características (v)
"""

import os
import warnings
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    'osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25', 'osnet_ibn_x1_0'
]


# -----------------------------
# Camadas básicas
# -----------------------------
class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, IN=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, bias=False, groups=groups)
        self.bn = nn.InstanceNorm2d(out_channels, affine=True) if IN else nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0,
                              bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (sem não-linearidade)."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv3x3(nn.Module):
    """3x3 convolution + bn + relu."""
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1,
                              bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution: 1x1 (linear) + dw 3x3 (nonlinear)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# -----------------------------
# Blocos para Omni-Scale feature learning
# -----------------------------
class ChannelGate(nn.Module):
    """Gera gates por canal condicionados no tensor de entrada."""
    def __init__(self, in_channels, num_gates=None, return_gates=False,
                 gate_activation='sigmoid', reduction=16, layer_norm=False):
        super().__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True, padding=0)
        self.norm1 = None
        if layer_norm:
            # Pylance pode apontar tipo aqui, PyTorch aceita esse shape 3D
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))  # type: ignore[arg-type]
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, num_gates, kernel_size=1, bias=True, padding=0)
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError(f"Unknown gate activation: {gate_activation}")

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x


class OSBlock(nn.Module):
    """Omni-Scale feature learning block."""
    def __init__(self, in_channels, out_channels, IN=False, bottleneck_reduction=4, **kwargs):
        super().__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = Conv1x1Linear(in_channels, out_channels) if in_channels != out_channels else None
        self.IN = nn.InstanceNorm2d(out_channels, affine=True) if IN else None

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return F.relu(out)


# -----------------------------
# Arquitetura da rede
# -----------------------------
class OSNet(nn.Module):
    """Omni-Scale Network (ICCV'19, arXiv'19).

    Em eval(): forward retorna o vetor de características (embedding).
    """
    def __init__(self, num_classes, blocks, layers, channels, feature_dim=512, loss='softmax', IN=False, **kwargs):
        super().__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1
        self.loss = loss

        # Backbone convolucional
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(blocks[0], layers[0], channels[0], channels[1], reduce_spatial_size=True, IN=IN)
        self.conv3 = self._make_layer(blocks[1], layers[1], channels[1], channels[2], reduce_spatial_size=True)
        self.conv4 = self._make_layer(blocks[2], layers[2], channels[2], channels[3], reduce_spatial_size=False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        # Camada totalmente conectada (FC) para construir o vetor final (feature_dim)
        self.fc = self._construct_fc_layer(feature_dim, channels[3], dropout_p=None)
        # Classificador de identidade (somente útil em treino)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self._init_params()

    def _make_layer(self, block, layer, in_channels, out_channels, reduce_spatial_size, IN=False):
        layers = []
        layers.append(block(in_channels, out_channels, IN=IN))
        for _ in range(1, layer):
            layers.append(block(out_channels, out_channels, IN=IN))
        if reduce_spatial_size:
            layers.append(nn.Sequential(Conv1x1(out_channels, out_channels), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None
        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]
        layers = []
        for dim in fc_dims:
            layers += [nn.Linear(input_dim, dim), nn.BatchNorm1d(dim), nn.ReLU(inplace=True)]
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.feature_dim = fc_dims[-1]
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x, return_featuremaps=False):
        x = self.featuremaps(x)
        if return_featuremaps:
            return x
        v = self.global_avgpool(x).view(x.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError(f"Unsupported loss: {self.loss}")


# -----------------------------
# Utilitário: carregar pesos pré-treinados (opcional, sem gdown)
# -----------------------------
_pretrained_cache = {
    'osnet_x1_0': 'osnet_x1_0_imagenet.pth',
    'osnet_x0_75': 'osnet_x0_75_imagenet.pth',
    'osnet_x0_5': 'osnet_x0_5_imagenet.pth',
    'osnet_x0_25': 'osnet_x0_25_imagenet.pth',
    'osnet_ibn_x1_0': 'osnet_ibn_x1_0_imagenet.pth',
}


def _try_init_from_local_cache(model: nn.Module, key: str) -> None:
    """Tenta carregar pesos de ~/.cache/torch/checkpoints se existir.
    Se não existir, apenas avisa e segue SEM baixar nada (remoção de gdown).
    """
    filename = _pretrained_cache.get(key)
    if not filename:
        warnings.warn(f"[OSNet] chave de pretrained desconhecida: {key}")
        return
    torch_home = os.path.expanduser(os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch'))
    model_dir = os.path.join(torch_home, 'checkpoints')
    fpath = os.path.join(model_dir, filename)
    if not os.path.exists(fpath):
        warnings.warn(
            f"[OSNet] Peso pré-treinado '{filename}' não encontrado em '{model_dir}'. "
            f"Carregue manualmente via model.load_state_dict(...)."
        )
        return
    # Carrega estado
    state_dict = torch.load(fpath, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    new_state = OrderedDict((k[7:] if k.startswith('module.') else k, v) for k, v in state_dict.items())
    model.load_state_dict(new_state, strict=False)
    print(f"[OSNet] Pesos pré-treinados carregados de '{fpath}'")


# -----------------------------
# Fábricas de modelos
# -----------------------------

def osnet_x1_0(num_classes=1000, pretrained=False, loss='softmax', **kwargs):
    model = OSNet(num_classes, blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2],
                  channels=[64, 256, 384, 512], loss=loss, **kwargs)
    if pretrained:
        _try_init_from_local_cache(model, key='osnet_x1_0')
    return model


def osnet_x0_75(num_classes=1000, pretrained=False, loss='softmax', **kwargs):
    model = OSNet(num_classes, blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2],
                  channels=[48, 192, 288, 384], loss=loss, **kwargs)
    if pretrained:
        _try_init_from_local_cache(model, key='osnet_x0_75')
    return model


def osnet_x0_5(num_classes=1000, pretrained=False, loss='softmax', **kwargs):
    model = OSNet(num_classes, blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2],
                  channels=[32, 128, 192, 256], loss=loss, **kwargs)
    if pretrained:
        _try_init_from_local_cache(model, key='osnet_x0_5')
    return model


def osnet_x0_25(num_classes=1000, pretrained=False, loss='softmax', **kwargs):
    model = OSNet(num_classes, blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2],
                  channels=[16, 64, 96, 128], loss=loss, **kwargs)
    if pretrained:
        _try_init_from_local_cache(model, key='osnet_x0_25')
    return model


def osnet_ibn_x1_0(num_classes=1000, pretrained=False, loss='softmax', **kwargs):
    """Versão padrão (x1.0) com IBN nas primeiras camadas."""
    model = OSNet(num_classes, blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2],
                  channels=[64, 256, 384, 512], loss=loss, IN=True, **kwargs)
    if pretrained:
        _try_init_from_local_cache(model, key='osnet_ibn_x1_0')
    return model
