"""Fourier Neural Operator 2-D inference workload.

Architecture references:
- Paper: https://arxiv.org/abs/2010.08895
- Official project: https://github.com/neuraloperator/neuraloperator
- Documentation: https://neuraloperator.github.io/dev/

The implementation is self-contained and keeps the defining FNO operations:
real FFT, learned complex spectral multiplication, inverse FFT and pointwise
residual projections.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / max(1, in_channels * out_channels)
        shape = (in_channels, out_channels, modes1, modes2, 2)
        self.weight_positive = nn.Parameter(
            scale * torch.randn(shape, dtype=torch.float32)
        )
        self.weight_negative = nn.Parameter(
            scale * torch.randn(shape, dtype=torch.float32)
        )

    @staticmethod
    def _multiply(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Current MLU StridedBMM does not accept complex compute types.  Keep
        # the exact complex multiplication but lower it into four real-valued
        # contractions, which are portable across CPU/CUDA/MLU.
        real = torch.einsum("bixy,ioxy->boxy", x.real, weight[..., 0])
        real = real - torch.einsum("bixy,ioxy->boxy", x.imag, weight[..., 1])
        imaginary = torch.einsum("bixy,ioxy->boxy", x.real, weight[..., 1])
        imaginary = imaginary + torch.einsum("bixy,ioxy->boxy", x.imag, weight[..., 0])
        return torch.view_as_complex(torch.stack((real, imaginary), dim=-1).contiguous())

    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep FFT at an explicit graph boundary on the current MLU stack.  Its
        # runtime output stride differs from the backend fake/meta declaration,
        # so compiling across this boundary produces an invalid stride guard.
        # The operation still runs on MLU; only the surrounding pointwise path
        # is handed to torch.compile.
        batch, _, height, width = x.shape
        # MLU FFT currently returns the two frequency axes with a non-standard
        # physical stride, while its fake/meta kernel describes contiguous
        # NCHW-like storage.  Materialize that contract explicitly so eager and
        # torch.compile observe the same layout.
        x_ft = torch.fft.rfft2(x, norm="ortho").contiguous()
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            height,
            width // 2 + 1,
            device=x.device,
            dtype=x_ft.dtype,
        )
        modes1 = min(self.modes1, height // 2)
        modes2 = min(self.modes2, width // 2 + 1)
        out_ft[:, :, :modes1, :modes2] = self._multiply(
            x_ft[:, :, :modes1, :modes2],
            self.weight_positive[:, :, :modes1, :modes2, :],
        )
        out_ft[:, :, -modes1:, :modes2] = self._multiply(
            x_ft[:, :, -modes1:, :modes2],
            self.weight_negative[:, :, :modes1, :modes2, :],
        )
        return torch.fft.irfft2(out_ft, s=(height, width), norm="ortho")


class FourierLayer(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.pointwise(x))


class Model(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        width: int = 64,
        modes1: int = 16,
        modes2: int = 16,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.lift = nn.Conv2d(in_channels + 2, width, kernel_size=1)
        self.layers = nn.ModuleList(
            FourierLayer(width, modes1, modes2) for _ in range(num_layers)
        )
        self.project = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width * 2, out_channels, kernel_size=1),
        )

    @staticmethod
    def _grid(x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        grid_y = torch.linspace(0, 1, height, device=x.device, dtype=x.dtype)
        grid_x = torch.linspace(0, 1, width, device=x.device, dtype=x.dtype)
        y, x_coordinate = torch.meshgrid(grid_y, grid_x, indexing="ij")
        grid = torch.stack((x_coordinate, y), dim=0).unsqueeze(0)
        return grid.expand(batch, -1, -1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(torch.cat((x, self._grid(x)), dim=1))
        for layer in self.layers:
            x = layer(x)
        return self.project(x)


def get_inputs() -> list[torch.Tensor]:
    return [torch.randn(4, 3, 64, 64, dtype=torch.float32)]


def get_init_inputs() -> list[int]:
    return [3, 1, 64, 16, 16, 4]
