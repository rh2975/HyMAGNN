import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import GlobalCoreFusion


# Deformable Temporal Conv
class DeformableTemporalConv1D(nn.Module):
    """
    Deformable conv over (N, T) map with kernel (1, k).
    Deform only along temporal axis in practice (still uses 2D offsets).
    """
    def __init__(self, channels, kernel_size=3, stride=1, pad=None, bias=True, modulated=False):
        super().__init__()
        self.k = kernel_size
        self.stride = stride
        self.modulated = modulated

        if pad is None:
            pad = kernel_size // 2
        self.pad = pad

        self.conv = nn.Conv2d(
            channels, channels,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, self.pad),
            bias=bias
        )

        # offsets: 2*k (since kH=1, kW=k)
        self.offset_conv = nn.Conv2d(
            channels, 2 * kernel_size,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, self.pad),
            bias=True
        )

        if self.modulated:
            self.mask_conv = nn.Conv2d(
                channels, kernel_size,
                kernel_size=(1, kernel_size),
                stride=(1, stride),
                padding=(0, self.pad),
                bias=True
            )

        try:
            from torchvision.ops import deform_conv2d
            self._deform_conv2d = deform_conv2d
        except Exception:
            self._deform_conv2d = None
            print("[WARN] torchvision deform_conv2d not available. Falling back to normal conv.")

    def forward(self, x):
        if self._deform_conv2d is None:
            return self.conv(x)

        offset = self.offset_conv(x)  # (B,2k,N,T_out)
        mask = None
        if self.modulated:
            mask = torch.sigmoid(self.mask_conv(x))  # (B,k,N,T_out)

        out = self._deform_conv2d(
            x,
            offset,
            self.conv.weight,
            self.conv.bias,
            stride=(1, self.stride),
            padding=(0, self.pad),
            dilation=(1, 1),
            mask=mask
        )
        return out


# Helpers
class NConv(nn.Module):
    def forward(self, x, A):
        return torch.einsum('bcnt,nm->bcmt', x, A)

class Linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1,1), bias=bias)
    def forward(self, x):
        return self.mlp(x)

class MixProp(nn.Module):
    def __init__(self, c_in, c_out, gdep=2, dropout=0.3, alpha=0.05):
        super().__init__()
        self.nconv = NConv()
        self.mlp = Linear((gdep+1)*c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, A):
        A = A / (A.sum(dim=1, keepdim=True) + 1e-8)
        h = x
        out = [h]
        for _ in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, A)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return F.dropout(ho, p=self.dropout, training=self.training)


class PyramidLayer(nn.Module):
    def __init__(self, channels, ksize, stride=2, use_deform=False):
        super().__init__()

        
        if ksize % 2 == 0:
            pad = (ksize // 2) - 1  
        else:
            pad = ksize // 2

        if use_deform:
            self.rec = DeformableTemporalConv1D(
                channels,
                kernel_size=ksize,
                stride=stride,
                pad=pad,
                modulated=False
            )
        else:
            self.rec = nn.Conv2d(
                channels, channels,
                kernel_size=(1, ksize),
                stride=(1, stride),
                padding=(0, pad)
            )

        self.norm = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))  
        )
        self.act = nn.ReLU()

    def forward(self, x):
        xr = self.act(self.rec(x))
        xn = self.norm(x)

        
        T = min(xr.size(-1), xn.size(-1))
        xr = xr[..., -T:]
        xn = xn[..., -T:]
        return xr + xn


class MAGNN(nn.Module):
    def __init__(self, num_nodes, seq_in_len=168, horizon=3,
                 num_scales=4, topk=20, embed_dim=40,
                 conv_channels=32, ds=32, gcn_depth=2,
                 dropout=0.3, alpha=0.05,
                 use_hypergraph=False, num_hyperedges=32,
                 use_deform_conv=False, use_core_fusion=False,
                 core_hidden_dim=64):
        super().__init__()

        self.use_core_fusion = use_core_fusion

        self.start_conv = nn.Conv2d(1, conv_channels, kernel_size=(1,1))

        self.pyr2 = PyramidLayer(conv_channels, 7, use_deform=use_deform_conv)
        self.pyr3 = PyramidLayer(conv_channels, 6, use_deform=use_deform_conv)
        self.pyr4 = PyramidLayer(conv_channels, 3, use_deform=use_deform_conv)

        self.in_proj = nn.ModuleList([nn.Conv2d(conv_channels, ds, kernel_size=(1,1))
                                      for _ in range(num_scales)])
        self.gnn_in  = nn.ModuleList([MixProp(ds, ds, gcn_depth, dropout, alpha)
                                      for _ in range(num_scales)])
        self.gnn_out = nn.ModuleList([MixProp(ds, ds, gcn_depth, dropout, alpha)
                                      for _ in range(num_scales)])

        if use_core_fusion:
            self.core_fusion = GlobalCoreFusion(ds, core_hidden_dim)

        self.tcn = nn.ModuleList([nn.Conv1d(ds, ds, kernel_size=3, padding=1)
                                  for _ in range(num_scales)])

        self.fuse_fc1 = nn.Linear(num_nodes * ds, 128)
        self.fuse_fc2 = nn.Linear(128, num_scales)

        self.end_conv_1 = nn.Conv2d(ds, 64, kernel_size=(1,1))
        self.end_conv_2 = nn.Conv2d(64, horizon, kernel_size=(1,1))

    def _build_scales(self, x):
        x = self.start_conv(x)
        x1 = x
        x2 = self.pyr2(x1)
        x3 = self.pyr3(x2)
        x4 = self.pyr4(x3)
        return [x1, x2, x3, x4]

    def forward(self, x):
        scales = self._build_scales(x)
        hs = []

        for k in range(len(scales)):
            z = self.in_proj[k](scales[k])
            z = self.gnn_in[k](z, torch.eye(z.size(2), device=z.device))
            z = z.mean(dim=-1)
            hs.append(z)

        # hstack = torch.stack(hs, dim=1)
        # if self.use_core_fusion:
        #     hstack = self.core_fusion(hstack)
        hstack = torch.stack(hs, dim=1)  # (B, K, ds, N)
        
        if self.use_core_fusion:
            if not hasattr(self, "_dbg_core"):
                print("core_fusion input shape:", tuple(hstack.shape))
                self._dbg_core = True
        
            # core_fusion expects (B, C, N, T) with C=ds and T=K
            hstack = hstack.permute(0, 2, 3, 1).contiguous()  # (B, ds, N, K)
            hstack = self.core_fusion(hstack)                 # (B, ds, N, K)
            hstack = hstack.permute(0, 3, 1, 2).contiguous()  # (B, K, ds, N)
        
        hpool = hstack.mean(dim=1)  # (B, ds, N)

        z = hpool.reshape(hpool.size(0), -1)
        alpha = torch.sigmoid(self.fuse_fc2(F.relu(self.fuse_fc1(z))))

        hm = sum(alpha[:,k].view(-1,1,1) * hs[k] for k in range(len(hs)))
        # hm = F.relu(hm).permute(0,2,1).unsqueeze(-1)

        # y = self.end_conv_2(F.relu(self.end_conv_1(hm)))
        hm = F.relu(hm).unsqueeze(-1)   # (B, ds, N, 1)
        y = self.end_conv_2(F.relu(self.end_conv_1(hm)))

    
        return y, alpha
