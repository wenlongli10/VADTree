import torch
import torch.nn as nn


class SGPMixer(nn.Module):

    def __init__(
        self,
        dim,
        diff_idx,
        kernel_size=1,
        k=1.5
    ):
        super().__init__()
        self.dim = dim
        self.diff_idx = diff_idx
        self.kernel_size = kernel_size
        # up_size = round((kernel_size + 1) * k)
        # up_size = up_size + 1 if up_size % 2 == 0 else up_size
        up_size = 3

        self.norm = nn.BatchNorm1d(dim)
        self.psi = nn.Conv1d(dim, dim, kernel_size, stride=1, padding=kernel_size // 2, groups=dim)
        self.fc = nn.Conv1d(dim, dim, 1, stride=1, padding=0, groups=dim)
        self.convw = nn.Conv1d(dim, dim, kernel_size, stride=1, padding=kernel_size // 2, groups=dim)
        self.convkw = nn.Conv1d(dim, dim, up_size, stride=1, padding=up_size // 2, groups=dim)
        self.diff_norm = nn.BatchNorm1d(dim)
        self.global_fc = nn.Conv1d(dim, dim, 1, stride=1, padding=0, groups=dim)

    def diff(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.diff(x, n=1, dim=2)
        x = torch.nn.functional.pad(x, (1, 0), mode='replicate')
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # X shape: B, C, T
        B, C, T = x.shape

        out = self.norm(x)
        psi = self.psi(out)
        fc = self.fc(out)
        convw = self.convw(out)
        convkw = self.diff_norm(self.convkw(self.diff(out)))
        phi = torch.relu(self.global_fc(out.mean(dim=-1, keepdim=True)))
        out = fc * phi + (convw + convkw) * psi + out
        # out = (convw + convkw) * psi + out
        # out = (convw + convkw) * psi

        out = x + out

        return x


class DiffMixer(nn.Module):

    def __init__(
        self,
        dim,
        diff_idx=0,
        kernel_size=1,
    ):
        super().__init__()
        self.dim = dim
        self.diff_idx = diff_idx
        self.kernel_size = kernel_size

        self.pre_norm = nn.BatchNorm1d(dim)
        self.d = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(dim, dim, kernel_size, groups=1),
                    nn.BatchNorm1d(dim),
                    nn.GELU()
                )
                for _ in range(4)
            ]
        )

    def diff(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.diff(x, n=1, dim=2)
        x = torch.nn.functional.pad(x, (1, 0), mode='replicate')
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_identity = x
        x = self.pre_norm(x)
        x_d0 = self.d[0](x)
        x_d1 = self.d[1](self.diff(x))
        x_d2 = self.d[3](self.diff(self.d[2](self.diff(x))))

        x_d = x_d0 + x_d1 + x_d2

        x = x_identity + x_d

        return x


class ConvFFN(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module(
            "conv",
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding=kernel_size//2,
                groups=in_channels,
                # groups=1,
                bias=False,
            ),
        )
        self.conv.add_module("bn", nn.BatchNorm1d(num_features=in_channels))
        self.fc1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, groups=in_channels)
        self.act = act_layer()
        # self.fc2 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        # self.drop = nn.Dropout(drop)
        self.identity_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, groups=in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_identity = x
        x = self.conv(x)
        x = self.fc1(x)
        # x = self.act(x)
        # x = self.drop(x)
        # x = self.fc2(x)
        # x = self.drop(x)
        x = x + self.identity_conv(x_identity)
        x = self.act(x)
        return x


class DiffBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        diff_idx: int, 
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.token_mixer = DiffMixer(dim, diff_idx)
        self.convffn = ConvFFN(
            in_channels=dim,
            out_channels=dim,
            kernel_size=1,
            act_layer=act_layer,
            drop=drop,
        )
        # Drop Path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x_diff = self.token_mixer(x)
        x = self.drop_path(self.convffn(x_diff))
        return x


class DiffFormer(nn.Module):

    def __init__(
        self,
        dim: int,
        diff_idx: int,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(DiffBlock(dim=dim, diff_idx=diff_idx))
            self.downs.append(nn.MaxPool1d(kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        # input: [b*nl, c, f]
        x_diff_lists = [x]

        for i in range(self.num_blocks - 1):
            x = self.blocks[i](x)
            x_diff_lists.append(x)
            x = self.downs[i](x)

        x = self.blocks[-1](x)

        return x, x_diff_lists


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'