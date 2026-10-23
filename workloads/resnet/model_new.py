import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

# Heavily optimize for inference speed.
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def _fuse_conv_bn_eval(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fold BatchNorm2d into Conv2d (eval-mode only).
    Returns (weight, bias) to be used with F.conv2d.
    """
    assert isinstance(conv, nn.Conv2d)
    assert isinstance(bn, nn.BatchNorm2d)

    w = conv.weight
    if conv.bias is None:
        b = torch.zeros((w.size(0),), device=w.device, dtype=w.dtype)
    else:
        b = conv.bias

    # BN params/buffers
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    # Compute in fp32 for stability, cast back at end.
    w_f = w.float()
    b_f = b.float()
    gamma_f = gamma.float()
    beta_f = beta.float()
    mean_f = mean.float()
    var_f = var.float()

    invstd = torch.rsqrt(var_f + eps)
    scale = (gamma_f * invstd).reshape(-1, 1, 1, 1)
    w_fold = (w_f * scale).to(w.dtype)
    b_fold = (beta_f + (b_f - mean_f) * gamma_f * invstd).to(w.dtype)
    return w_fold, b_fold


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, dtype=torch.float16)
        self.bn1 = nn.BatchNorm2d(out_channels, dtype=torch.float16)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, dtype=torch.float16)
        self.bn2 = nn.BatchNorm2d(out_channels, dtype=torch.float16)
        self.downsample = downsample
        self.stride = stride

        # Inference-only caches (NOT registered buffers/params).
        self._fused_w1: torch.Tensor | None = None
        self._fused_b1: torch.Tensor | None = None
        self._fused_w2: torch.Tensor | None = None
        self._fused_b2: torch.Tensor | None = None
        self._fused_dw: torch.Tensor | None = None
        self._fused_db: torch.Tensor | None = None

    def _invalidate_fused(self) -> None:
        self._fused_w1 = None
        self._fused_b1 = None
        self._fused_w2 = None
        self._fused_b2 = None
        self._fused_dw = None
        self._fused_db = None

    def rebuild_fused(self) -> None:
        """
        Precompute fused conv/bn weights for inference.
        Must be called outside torch.compile (e.g., right after load_state_dict + eval()).
        """
        w1, b1 = _fuse_conv_bn_eval(self.conv1, self.bn1)
        w2, b2 = _fuse_conv_bn_eval(self.conv2, self.bn2)
        self._fused_w1, self._fused_b1 = w1, b1
        self._fused_w2, self._fused_b2 = w2, b2
        if self.downsample is not None:
            ds_conv = self.downsample[0]
            ds_bn = self.downsample[1]
            dw, db = _fuse_conv_bn_eval(ds_conv, ds_bn)
            self._fused_dw, self._fused_db = dw, db

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fast path: use precomputed fused weights (no module mutation in forward).
        if self._fused_w1 is not None and self._fused_w2 is not None:
            identity = x
            out = F.conv2d(
                x,
                self._fused_w1,
                self._fused_b1,
                stride=self.conv1.stride,
                padding=self.conv1.padding,
                dilation=self.conv1.dilation,
                groups=self.conv1.groups,
            )
            out = F.relu(out, inplace=True)
            out = F.conv2d(
                out,
                self._fused_w2,
                self._fused_b2,
                stride=self.conv2.stride,
                padding=self.conv2.padding,
                dilation=self.conv2.dilation,
                groups=self.conv2.groups,
            )

            if self.downsample is not None:
                # If downsample exists but fused weights missing, fall back to original.
                if self._fused_dw is None:
                    identity = self.downsample(x)
                else:
                    ds_conv = self.downsample[0]
                    identity = F.conv2d(
                        x,
                        self._fused_dw,
                        self._fused_db,
                        stride=ds_conv.stride,
                        padding=ds_conv.padding,
                        dilation=ds_conv.dilation,
                        groups=ds_conv.groups,
                    )

            out = out + identity
            out = F.relu(out, inplace=True)
            return out

        # Fallback: original implementation (correct, slower).
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False, dtype=torch.float16)
        self.bn1 = nn.BatchNorm2d(64, dtype=torch.float16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes, dtype=torch.float16)

        # Inference-only cache for stem (conv1+bn1).
        self._stem_w: torch.Tensor | None = None
        self._stem_b: torch.Tensor | None = None

        self.register_load_state_dict_post_hook(self._post_load_hook)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False, dtype=torch.float16),
                nn.BatchNorm2d(out_channels * block.expansion, dtype=torch.float16),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _post_load_hook(self, module: "Model", _incompatible_keys) -> None:
        module._stem_w = None
        module._stem_b = None
        # Rebuild fused weights after all params are loaded.
        # (This runs outside torch.compile.)
        module.rebuild_fused()

    def rebuild_fused(self) -> None:
        # Stem
        w, b = _fuse_conv_bn_eval(self.conv1, self.bn1)
        self._stem_w, self._stem_b = w, b
        # Blocks
        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.rebuild_fused()

    def train(self, mode: bool = True):
        super().train(mode)
        # When switching to eval, (re)build inference caches.
        if not mode:
            self.rebuild_fused()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fast path: use precomputed fused stem weights.
        if self._stem_w is not None:
            x = F.conv2d(
                x,
                self._stem_w,
                self._stem_b,
                stride=self.conv1.stride,
                padding=self.conv1.padding,
                dilation=self.conv1.dilation,
                groups=self.conv1.groups,
            )
            x = F.relu(x, inplace=True)
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        else:
            # Fallback: original stem.
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Replace AdaptiveAvgPool2d((1,1)) with a reduction.
        x = x.mean(dim=(2, 3))
        x = self.fc(x)
        return x


# Test code
batch_size = 4
num_classes = 1000
input_shape = (batch_size, 3, 224, 224)


def get_inputs():
    return [torch.randn(input_shape, dtype=torch.float16)]


def get_init_inputs():
    return [num_classes]


if __name__ == "__main__":
    torch.set_default_device("mlu" if torch.mlu.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    m = Model(*get_init_inputs()).eval()
    xs = get_inputs()
    with torch.no_grad():
        y = m(*xs)
    print(y.shape)
    print(y.sum())

