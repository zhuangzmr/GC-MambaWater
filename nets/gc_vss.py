##########################################################
# Complete VMamba with GC_SS2D Integration
# This version includes all necessary components and the new GC_SS2D module
##########################################################

import os
import time
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import warnings
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

# Check for Triton support
WITH_TRITON = True
try:
    import triton
    import triton.language as tl
except:
    WITH_TRITON = False
    warnings.warn("Triton not installed, fall back to pytorch implements.")

# Check for selective scan CUDA support
WITH_SELECTIVESCAN_MAMBA = True
try:
    import selective_scan_cuda
except ImportError:
    WITH_SELECTIVESCAN_MAMBA = False
    warnings.warn("selective_scan_cuda not installed, falling back to pytorch implementation.")

if WITH_TRITON:
    try:
        from functools import cached_property
    except:
        warnings.warn("if you are using py37, add this line to functools.py: "
            "cached_property = lambda func: property(lru_cache()(func))")

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

##########################################################
# Basic Layers
##########################################################

class Linear(nn.Linear):
    def __init__(self, *args, channel_first=False, groups=1, **kwargs):
        nn.Linear.__init__(self, *args, **kwargs)
        self.channel_first = channel_first
        self.groups = groups
    
    def forward(self, x: torch.Tensor):
        if self.channel_first:
            if len(x.shape) == 4:
                return F.conv2d(x, self.weight[:, :, None, None], self.bias, groups=self.groups)
            elif len(x.shape) == 3:
                return F.conv1d(x, self.weight[:, :, None], self.bias, groups=self.groups)
        else:
            return F.linear(x, self.weight, self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self_state_dict = self.state_dict()
        load_state_dict_keys = list(state_dict.keys())
        if prefix + "weight" in load_state_dict_keys:
            state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view_as(self_state_dict["weight"])
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, channel_first=None, in_channel_first=False, out_channel_first=False, **kwargs):
        nn.LayerNorm.__init__(self, *args, **kwargs)
        if channel_first is not None:
            in_channel_first = channel_first
            out_channel_first = channel_first
        self.in_channel_first = in_channel_first
        self.out_channel_first = out_channel_first

    def forward(self, x: torch.Tensor):
        if self.in_channel_first:
            x = x.permute(0, 2, 3, 1)
        x = nn.LayerNorm.forward(self, x)
        if self.out_channel_first:
            x = x.permute(0, 3, 1, 2)
        return x


class PatchMerge(nn.Module):
    def __init__(self, channel_first=True, in_channel_first=False, out_channel_first=False,):
        nn.Module.__init__(self)
        if channel_first is not None:
            in_channel_first = channel_first
            out_channel_first = channel_first
        self.in_channel_first = in_channel_first
        self.out_channel_first = out_channel_first

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        if not self.in_channel_first:
            B, H, W, C = x.shape
        
        if (W % 2 != 0) or (H % 2 != 0):
            PH, PW = H - H % 2, W - W % 2
            pad_shape = (PW // 2, PW - PW // 2, PH // 2, PH - PH // 2)
            pad_shape = (*pad_shape, 0, 0, 0, 0) if self.in_channel_first else (0, 0, *pad_shape, 0, 0)
            x = nn.functional.pad(x, pad_shape)
        
        xs = [
            x[..., 0::2, 0::2], x[..., 1::2, 0::2], 
            x[..., 0::2, 1::2], x[..., 1::2, 1::2],
        ] if self.in_channel_first else [
            x[..., 0::2, 0::2, :], x[..., 1::2, 0::2, :], 
            x[..., 0::2, 1::2, :], x[..., 1::2, 1::2, :],
        ]

        xs = torch.cat(xs, (1 if self.out_channel_first else -1))
        return xs


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channel_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features, channel_first=channel_first)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features, channel_first=channel_first)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

##########################################################
# Cross Scan Operations (PyTorch Implementation)
##########################################################

def cross_scan_fwd(x: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if in_channel_first:
        B, C, H, W = x.shape
        if scans == 0:
            y = x.new_empty((B, 4, C, H * W))
            y[:, 0, :, :] = x.flatten(2, 3)
            y[:, 1, :, :] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
            y[:, 2:4, :, :] = torch.flip(y[:, 0:2, :, :], dims=[-1])
        elif scans == 1:
            y = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1)
        elif scans == 2:
            y = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
            y = torch.cat([y, y.flip(dims=[-1])], dim=1)
        elif scans == 3:
            y = x.new_empty((B, 4, C, H * W))
            y[:, 0, :, :] = x.flatten(2, 3)
            y[:, 1, :, :] = torch.rot90(x, 1, dims=(2, 3)).flatten(2, 3)
            y[:, 2, :, :] = torch.rot90(x, 2, dims=(2, 3)).flatten(2, 3)
            y[:, 3, :, :] = torch.rot90(x, 3, dims=(2, 3)).flatten(2, 3)
    else:
        B, H, W, C = x.shape
        if scans == 0:
            y = x.new_empty((B, H * W, 4, C))
            y[:, :, 0, :] = x.flatten(1, 2)
            y[:, :, 1, :] = x.transpose(dim0=1, dim1=2).flatten(1, 2)
            y[:, :, 2:4, :] = torch.flip(y[:, :, 0:2, :], dims=[1])
        elif scans == 1:
            y = x.view(B, H * W, 1, C).repeat(1, 1, 4, 1)
        elif scans == 2:
            y = x.view(B, H * W, 1, C).repeat(1, 1, 2, 1)
            y = torch.cat([y, y.flip(dims=[1])], dim=2)
        elif scans == 3:
            y = x.new_empty((B, H * W, 4, C))
            y[:, :, 0, :] = x.flatten(1, 2)
            y[:, :, 1, :] = torch.rot90(x, 1, dims=(1, 2)).flatten(1, 2)
            y[:, :, 2, :] = torch.rot90(x, 2, dims=(1, 2)).flatten(1, 2)
            y[:, :, 3, :] = torch.rot90(x, 3, dims=(1, 2)).flatten(1, 2)

    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()

    return y


def cross_merge_fwd(y: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0):
    if out_channel_first:
        B, K, D, H, W = y.shape
        y = y.view(B, K, D, -1)
        if scans == 0:
            y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = y[:, 0] + y[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        elif scans == 1:
            y = y.sum(1)
        elif scans == 2:
            y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = y.sum(1)
        elif scans == 3:
            oy = y[:, 0, :, :].contiguous().view(B, D, -1)
            oy = oy + torch.rot90(y.view(B, K, D, W, H)[:, 1, :, :, :], -1, dims=(2, 3)).flatten(2, 3)
            oy = oy + torch.rot90(y.view(B, K, D, H, W)[:, 2, :, :, :], -2, dims=(2, 3)).flatten(2, 3)
            oy = oy + torch.rot90(y.view(B, K, D, W, H)[:, 3, :, :, :], -3, dims=(2, 3)).flatten(2, 3)
            y = oy
    else:
        B, H, W, K, D = y.shape
        y = y.view(B, -1, K, D)
        if scans == 0:
            y = y[:, :, 0:2] + y[:, :, 2:4].flip(dims=[1]).view(B, -1, 2, D)
            y = y[:, :, 0] + y[:, :, 1].view(B, W, H, -1).transpose(dim0=1, dim1=2).contiguous().view(B, -1, D)        
        elif scans == 1:
            y = y.sum(2)
        elif scans == 2:
            y = y[:, :, 0:2] + y[:, :, 2:4].flip(dims=[1]).view(B, -1, 2, D)
            y = y.sum(2)
        elif scans == 3:
            oy = y[:, :, 0, :].contiguous().view(B, -1, D)
            oy = oy + torch.rot90(y.view(B, W, H, K, D)[:, :, :, 1, :], -1, dims=(1, 2)).flatten(1, 2)
            oy = oy + torch.rot90(y.view(B, H, W, K, D)[:, :, :, 2, :], -2, dims=(1, 2)).flatten(1, 2)
            oy = oy + torch.rot90(y.view(B, W, H, K, D)[:, :, :, 3, :], -3, dims=(1, 2)).flatten(1, 2)
            y = oy
            
    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 2, 1).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 1).contiguous()
    
    return y


class CrossScanF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans

        if one_by_one:
            B, K, C, H, W = x.shape
            if not in_channel_first:
                B, H, W, K, C = x.shape
        else:
            B, C, H, W = x.shape
            if not in_channel_first:
                B, H, W, C = x.shape
        ctx.shape = (B, C, H, W)

        y = cross_scan_fwd(x, in_channel_first, out_channel_first, scans)
        return y
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape

        ys = ys.view(B, -1, C, H, W) if out_channel_first else ys.view(B, H, W, -1, C)
        y = cross_merge_fwd(ys, in_channel_first, out_channel_first, scans)
        
        if one_by_one:
            y = y.view(B, 4, -1, H, W) if in_channel_first else y.view(B, H, W, 4, -1)
        else:
            y = y.view(B, -1, H, W) if in_channel_first else y.view(B, H, W, -1)

        return y, None, None, None, None


class CrossMergeF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0):
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans

        B, K, C, H, W = ys.shape
        if not out_channel_first:
            B, H, W, K, C = ys.shape
        ctx.shape = (B, C, H, W)
        
        y = cross_merge_fwd(ys, in_channel_first, out_channel_first, scans)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        in_channel_first = ctx.in_channel_first
        out_channel_first = ctx.out_channel_first
        one_by_one = ctx.one_by_one
        scans = ctx.scans
        B, C, H, W = ctx.shape
    
        if not one_by_one:
            if in_channel_first:
                x = x.view(B, C, H, W)
            else:
                x = x.view(B, H, W, C)
        else:
            if in_channel_first:
                x = x.view(B, 4, C, H, W)
            else:
                x = x.view(B, H, W, 4, C)   
                     
        x = cross_scan_fwd(x, in_channel_first, out_channel_first, scans)
        x = x.view(B, 4, C, H, W) if out_channel_first else x.view(B, H, W, 4, C)
    
        return x, None, None, None, None


def cross_scan_fn(x: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, force_torch=False):
    return CrossScanF.apply(x, in_channel_first, out_channel_first, one_by_one, scans)


def cross_merge_fn(y: torch.Tensor, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0, force_torch=False):
    return CrossMergeF.apply(y, in_channel_first, out_channel_first, one_by_one, scans)

##########################################################
# Selective Scan
##########################################################

def selective_scan_torch(
    u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
    D: torch.Tensor = None, delta_bias: torch.Tensor = None, delta_softplus=True, oflex=True,
    *args, **kwargs
):
    dtype_in = u.dtype
    Batch, K, N, L = B.shape
    KCdim = u.shape[1]
    Cdim = int(KCdim / K)
    
    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)
            
    u, delta, A, B, C = u.float(), delta.float(), A.float(), B.float(), C.float()
    B = B.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)
    C = C.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    
    x = A.new_zeros((Batch, KCdim, N))
    ys = []
    for i in range(L):
        x = deltaA[:, :, i, :] * x + deltaB_u[:, :, i, :]
        y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        ys.append(y)
    y = torch.stack(ys, dim=2)
    
    out = y if D is None else y + u * D.unsqueeze(-1)
    return out if oflex else out.to(dtype=dtype_in)


class SelectiveScanCuda(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, oflex=True, backend=None):
        ctx.delta_softplus = delta_softplus
        backend = "mamba" if WITH_SELECTIVESCAN_MAMBA and (backend is None) else backend
        ctx.backend = backend
        if backend == "mamba":
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        else:
            out = selective_scan_torch(u, delta, A, B, C, D, delta_bias, delta_softplus, oflex)
            x = None
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        backend = ctx.backend
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if backend == "mamba":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus, False
            )
        else:
            # Simplified backward pass for torch backend
            du = ddelta = dA = dB = dC = dD = ddelta_bias = None
        return du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None


def selective_scan_fn(
    u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
    D: torch.Tensor = None, delta_bias: torch.Tensor = None, delta_softplus=True, oflex=True, backend=None,
):
    if backend == "torch" or (not WITH_SELECTIVESCAN_MAMBA):
        return selective_scan_torch(u, delta, A, B, C, D, delta_bias, delta_softplus, oflex)
    else:
        return SelectiveScanCuda.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, oflex, backend)

##########################################################
# Mamba Initialization
##########################################################

class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))
        del dt_projs
            
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)
        return A_logs, Ds, dt_projs_weight, dt_projs_bias

##########################################################
# GC_SS2D Module (New Improved Version)
##########################################################

class GC_SS2D(nn.Module):
    """带GRN统计与按通道方向路由的改进SS2D块
    严格实现：Y = α ⊙ Y^(h) + β ⊙ Y^(v)
    """
    def __init__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        forward_type="gc",
        channel_first=True,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(ssm_ratio * d_model)
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.k_group = 4

        # 输入投影
        self.in_proj = Linear(d_model, self.d_inner * 2, bias=bias, channel_first=channel_first)
        self.act = act_layer()

        # 深度卷积
        self.with_dconv = d_conv > 1
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                self.d_inner, self.d_inner,
                kernel_size=d_conv, padding=(d_conv - 1) // 2,
                groups=self.d_inner, bias=conv_bias
            )

        # === GRN 统计 + 方向路由（这里的 route_proj 会被“零初始化”为均匀路由起步）===
        self.grn_gamma = nn.Parameter(torch.ones(self.d_inner))
        self.grn_beta  = nn.Parameter(torch.zeros(self.d_inner))

        self.route_proj = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 2),
            nn.GELU(),
            nn.Linear(self.d_inner // 2, 2 * self.d_inner)  # 输出 2*C 的路由 logits
        )
        # 关键初始化：让初始路由接近均匀（alpha≈beta≈0.5）
        nn.init.zeros_(self.route_proj[-1].weight)
        nn.init.zeros_(self.route_proj[-1].bias)
        self.route_temperature = 2.0  # 温度>1使分布更均匀；需要更激进可调小

        # x/Δt 的投影
        self.x_proj = Linear(
            self.d_inner,
            self.k_group * (self.dt_rank + self.d_state * 2),
            groups=self.k_group, bias=False, channel_first=True
        )
        self.dt_projs = Linear(
            self.dt_rank, self.k_group * self.d_inner,
            groups=self.k_group, bias=False, channel_first=True
        )

        # SSM 参数初始化
        if initialize == "v0":
            self.A_logs, self.Ds, dt_w, dt_b = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner,
                dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                k_group=self.k_group
            )
            self.dt_projs.weight.data = dt_w.data.view(self.dt_projs.weight.shape)
            self.dt_projs_bias = dt_b
        else:
            self.Ds = nn.Parameter(torch.ones(self.k_group * self.d_inner))
            self.A_logs = nn.Parameter(torch.randn(self.k_group * self.d_inner, self.d_state))
            dt_w = nn.Parameter(torch.randn(self.k_group, self.d_inner, self.dt_rank))
            self.dt_projs.weight.data = dt_w.data.view(self.dt_projs.weight.shape)
            self.dt_projs_bias = nn.Parameter(torch.randn(self.k_group, self.d_inner))

        # 输出
        self.out_norm = LayerNorm(self.d_inner, channel_first=channel_first)
        self.out_proj = Linear(self.d_inner, d_model, bias=bias, channel_first=channel_first)
        self.dropout = nn.Dropout(dropout)

    def compute_grn_stats(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        channel_norms = torch.norm(x_flat, p=2, dim=2)
        mean_norm = channel_norms.mean(dim=1, keepdim=True)
        std_norm  = channel_norms.std(dim=1, keepdim=True) + 1e-6
        normalized = (channel_norms - mean_norm) / std_norm
        stats = normalized * self.grn_gamma + self.grn_beta
        return stats  # [B, C]

    def forward(self, x):
        # 输入两分支
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        z = self.act(z)

        # NHWC->NCHW（如有）
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()

        # 深度卷积
        if self.with_dconv:
            x = self.conv2d(x)
        x = self.act(x)

        B, D, H, W = x.shape
        K, N, R = self.k_group, self.d_state, self.dt_rank
        L = H * W

        # GRN 统计 + 路由（这里的路由会先用温度做softmax，初始更均匀）
        stats = self.compute_grn_stats(x)                  # [B, D]
        route_logits = self.route_proj(stats)              # [B, 2*D]

        # Cross-scan
        xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=0)

        # 参数投影
        x_dbl = self.x_proj(xs.view(B, -1, L)).view(B, K, -1, L)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)

        dts = dts.contiguous().view(B, -1, L)
        dts = self.dt_projs(dts)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -self.A_logs.float().exp()
        Ds = self.Ds.float()
        Bs = Bs.contiguous()
        Cs = Cs.contiguous()
        delta_bias = self.dt_projs_bias.view(-1).float()

        # 选择性扫描
        ys = selective_scan_fn(
            xs, dts, As, Bs, Cs, Ds,
            delta_bias=delta_bias,
            delta_softplus=True,
            backend="mamba"
        ).view(B, K, D, H, W)

        # 横/纵向融合
        y_h = 0.5 * (ys[:, 0] + ys[:, 1])
        y_v = 0.5 * (ys[:, 2] + ys[:, 3])

        # 按通道路由权重（加入温度）
        route = route_logits.view(B, 2, D, 1, 1)
        route = torch.softmax(route / self.route_temperature, dim=1)
        alpha, beta = route[:, 0], route[:, 1]

        y = alpha * y_h + beta * y_v

        if not self.channel_first:
            y = y.permute(0, 2, 3, 1).contiguous()

        y = self.out_norm(y)
        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


##########################################################
# SS2D Module (Original)
##########################################################

class SS2Dv2:
    def __initv2__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        forward_type="v2",
        channel_first=False,
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.k_group = 4
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank)
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        self.forward = self.forwardv2

        # tags for forward_type
        checkpostfix = self.checkpostfix
        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        self.out_norm, forward_type = self.get_outnorm(forward_type, self.d_inner, channel_first)

        # forward_type selection
        FORWARD_TYPES = dict(
            v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True, selective_scan_backend="mamba"),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, partial(self.forward_corev2, force_fp32=False, selective_scan_backend="mamba"))

        # in proj
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(self.d_model, d_proj, bias=bias, channel_first=channel_first)
        self.act: nn.Module = act_layer()
        
        # conv
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj
        self.x_proj = Linear(self.d_inner, self.k_group * (self.dt_rank + self.d_state * 2), groups=self.k_group, bias=False, channel_first=True)
        self.dt_projs = Linear(self.dt_rank, self.k_group * self.d_inner, groups=self.k_group, bias=False, channel_first=True)
          
        # out proj
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias, channel_first=channel_first)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=self.k_group,
            )
        else:
            self.Ds = nn.Parameter(torch.ones((self.k_group * self.d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.k_group * self.d_inner, self.d_state)))
            self.dt_projs_weight = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, self.d_inner)))
        
        self.dt_projs.weight.data = self.dt_projs_weight.data.view(self.dt_projs.weight.shape)
        del self.dt_projs_weight

    def forward_corev2(
        self,
        x: torch.Tensor=None, 
        force_fp32=False,
        ssoflex=True,
        selective_scan_backend = "mamba",
        scan_mode = "cross2d",
        scan_force_torch = False,
        no_einsum = False,
        **kwargs,
    ):
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, scan_mode) if isinstance(scan_mode, str) else scan_mode
        delta_softplus = True
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        force_fp32 = force_fp32 or ((not ssoflex) and self.training)

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W

        xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)
        x_dbl = self.x_proj(xs.view(B, -1, L))
        dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
        dts = dts.contiguous().view(B, -1, L)
        dts = self.dt_projs(dts)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -self.A_logs.to(torch.float).exp()
        Ds = self.Ds.to(torch.float)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        ys: torch.Tensor = selective_scan_fn(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, ssoflex, backend=selective_scan_backend
        ).view(B, K, -1, H, W)
        
        y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.permute(0, 2, 3, 1).contiguous()
        y = self.out_norm(y)

        return y.to(x.dtype)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out

    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner, channel_first=channel_first),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner, channel_first=channel_first)

        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value


class SS2D(nn.Module, SS2Dv2):
    def __init__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        forward_type="v2",
        channel_first=False,
        **kwargs,
    ):
        nn.Module.__init__(self)
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,
        )
        self.__initv2__(**kwargs)

##########################################################
# VSSBlock
##########################################################

class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        channel_first=False,
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        use_checkpoint: bool = False,
        post_norm: bool = False,
        use_gc_ss2d: bool = True,  # New parameter to choose between SS2D and GC_SS2D
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = LayerNorm(hidden_dim, channel_first=channel_first)
            
            # Choose between SS2D and GC_SS2D
            SSM_Module = GC_SS2D if use_gc_ss2d else SS2D
            
            self.op = SSM_Module(
                d_model=hidden_dim, 
                d_state=ssm_d_state, 
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
            )
        
        self.drop_path = DropPath(drop_path)
        
        if self.mlp_branch:
            self.norm2 = LayerNorm(hidden_dim, channel_first=channel_first)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, 
                          act_layer=mlp_act_layer, drop=mlp_drop_rate, channel_first=channel_first)

    def _forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)











































##########################################################
# VSSM Model
##########################################################

# class VSSM(nn.Module):
#     def __init__(
#         self, 
#         patch_size=4, 
#         in_chans=3, 
#         num_classes=1000, 
#         depths=[2, 2, 9, 2], 
#         dims=[96, 192, 384, 768], 
#         ssm_d_state=16,
#         ssm_ratio=2.0,
#         ssm_dt_rank="auto",
#         ssm_act_layer="silu",        
#         ssm_conv=3,
#         ssm_conv_bias=True,
#         ssm_drop_rate=0.0, 
#         ssm_init="v0",
#         forward_type="v2",
#         mlp_ratio=4.0,
#         mlp_act_layer="gelu",
#         mlp_drop_rate=0.0,
#         gmlp=False,
#         drop_path_rate=0.1, 
#         patch_norm=True, 
#         norm_layer="LN",
#         downsample_version: str = "v2",
#         patchembed_version: str = "v1",
#         use_checkpoint=False,  
#         posembed=False,
#         imgsize=224,
#         use_gc_ss2d=False,  # New parameter
#         **kwargs,
#     ):
#         super().__init__()
#         self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
#         self.num_classes = num_classes
#         self.num_layers = len(depths)
#         if isinstance(dims, int):
#             dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
#         self.num_features = dims[-1]
#         self.dims = dims
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
# 
#         _ACTLAYERS = dict(
#             silu=nn.SiLU, 
#             gelu=nn.GELU, 
#             relu=nn.ReLU, 
#             sigmoid=nn.Sigmoid,
#         )
#         ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
#         mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)
# 
#         self.pos_embed = self._pos_embed(dims[0], patch_size, imgsize) if posembed else None
#         self.patch_embed = self._make_patch_embed(in_chans, dims[0], patch_size, patch_norm, 
#                                                   channel_first=self.channel_first, version=patchembed_version)
# 
#         self.layers = nn.ModuleList()
#         for i_layer in range(self.num_layers):
#             downsample = self._make_downsample(
#                 self.dims[i_layer], 
#                 self.dims[i_layer + 1], 
#                 channel_first=self.channel_first,
#                 version=downsample_version,
#             ) if (i_layer < self.num_layers - 1) else nn.Identity()
# 
#             self.layers.append(self._make_layer(
#                 dim = self.dims[i_layer],
#                 drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
#                 use_checkpoint=use_checkpoint,
#                 downsample=downsample,
#                 channel_first=self.channel_first,
#                 ssm_d_state=ssm_d_state,
#                 ssm_ratio=ssm_ratio,
#                 ssm_dt_rank=ssm_dt_rank,
#                 ssm_act_layer=ssm_act_layer,
#                 ssm_conv=ssm_conv,
#                 ssm_conv_bias=ssm_conv_bias,
#                 ssm_drop_rate=ssm_drop_rate,
#                 ssm_init=ssm_init,
#                 forward_type=forward_type,
#                 mlp_ratio=mlp_ratio,
#                 mlp_act_layer=mlp_act_layer,
#                 mlp_drop_rate=mlp_drop_rate,
#                 gmlp=gmlp,
#                 use_gc_ss2d=use_gc_ss2d,
#             ))
# 
#         self.classifier = nn.Sequential(OrderedDict(
#             norm=LayerNorm(self.num_features, channel_first=self.channel_first),
#             permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
#             avgpool=nn.AdaptiveAvgPool2d(1),
#             flatten=nn.Flatten(1),
#             head=nn.Linear(self.num_features, num_classes),
#         ))
# 
#         self.apply(self._init_weights)
# 
#     @staticmethod
#     def _pos_embed(embed_dims, patch_size, img_size):
#         patch_height, patch_width = (img_size // patch_size, img_size // patch_size)
#         pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
#         trunc_normal_(pos_embed, std=0.02)
#         return pos_embed
# 
#     def _init_weights(self, m: nn.Module):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
# 
#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {"pos_embed"}
# 
#     @torch.jit.ignore
#     def no_weight_decay_keywords(self):
#         return {}
# 
#     @staticmethod
#     def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, channel_first=False, version="v1"):
#         if version == "v1":
#             return nn.Sequential(
#                 nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
#                 nn.Identity(),
#                 (LayerNorm(embed_dim, in_channel_first=True, out_channel_first=channel_first) 
#                     if patch_norm else (nn.Identity() if channel_first else Permute(0, 2, 3, 1))),
#             )
#         elif version == "v2":
#             stride = patch_size // 2
#             kernel_size = stride + 1
#             padding = 1
#             return nn.Sequential(
#                 nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
#                 nn.Identity(),
#                 (LayerNorm(embed_dim // 2, channel_first=True) if patch_norm else nn.Identity()),
#                 nn.Identity(),
#                 nn.GELU(),
#                 nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
#                 nn.Identity(),
#                 (LayerNorm(embed_dim, in_channel_first=True, out_channel_first=channel_first) 
#                     if patch_norm else (nn.Identity() if channel_first else Permute(0, 2, 3, 1))),
#             )
#         raise NotImplementedError
# 
#     @staticmethod
#     def _make_downsample(dim=96, out_dim=192, norm=True, channel_first=False, version="v1"):
#         if version == "v1":
#             return nn.Sequential(
#                 PatchMerge(channel_first),
#                 LayerNorm(4 * dim, channel_first=channel_first) if norm else nn.Identity(),
#                 Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False, channel_first=channel_first),
#             )
#         elif version == "v2":
#             return nn.Sequential(
#                 (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
#                 nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
#                 nn.Identity(),
#                 LayerNorm(out_dim, in_channel_first=True, out_channel_first=channel_first) if norm else 
#                     (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
#             )
#         elif version == "v3":
#             return nn.Sequential(
#                 (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
#                 nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
#                 nn.Identity(),
#                 LayerNorm(out_dim, in_channel_first=True, out_channel_first=channel_first) if norm else 
#                     (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
#             )
#         raise NotImplementedError
# 
#     @staticmethod
#     def _make_layer(
#         dim=96, 
#         drop_path=[0.1, 0.1], 
#         use_checkpoint=False, 
#         downsample=nn.Identity(),
#         channel_first=False,
#         ssm_d_state=16,
#         ssm_ratio=2.0,
#         ssm_dt_rank="auto",       
#         ssm_act_layer=nn.SiLU,
#         ssm_conv=3,
#         ssm_conv_bias=True,
#         ssm_drop_rate=0.0, 
#         ssm_init="v0",
#         forward_type="v2",
#         mlp_ratio=4.0,
#         mlp_act_layer=nn.GELU,
#         mlp_drop_rate=0.0,
#         use_gc_ss2d=False,
#         **kwargs,
#     ):
#         depth = len(drop_path)
#         blocks = []
#         for d in range(depth):
#             blocks.append(VSSBlock(
#                 hidden_dim=dim, 
#                 drop_path=drop_path[d],
#                 channel_first=channel_first,
#                 ssm_d_state=ssm_d_state,
#                 ssm_ratio=ssm_ratio,
#                 ssm_dt_rank=ssm_dt_rank,
#                 ssm_act_layer=ssm_act_layer,
#                 ssm_conv=ssm_conv,
#                 ssm_conv_bias=ssm_conv_bias,
#                 ssm_drop_rate=ssm_drop_rate,
#                 ssm_init=ssm_init,
#                 forward_type=forward_type,
#                 mlp_ratio=mlp_ratio,
#                 mlp_act_layer=mlp_act_layer,
#                 mlp_drop_rate=mlp_drop_rate,
#                 use_checkpoint=use_checkpoint,
#                 use_gc_ss2d=use_gc_ss2d,
#             ))
#         
#         return nn.Sequential(OrderedDict(
#             blocks=nn.Sequential(*blocks,),
#             downsample=downsample,
#         ))
# 
#     def forward(self, x: torch.Tensor):
#         x = self.patch_embed(x)
#         if self.pos_embed is not None:
#             pos_embed = self.pos_embed.permute(0, 2, 3, 1) if not self.channel_first else self.pos_embed
#             x = x + pos_embed
#         for layer in self.layers:
#             x = layer(x)
#         x = self.classifier(x)
#         return x
# 
#     def flops(self, shape=(3, 224, 224), verbose=True):
#         supported_ops={
#             "aten::silu": None,
#             "aten::neg": None,
#             "aten::exp": None,
#             "aten::flip": None,
#             "prim::PythonOp.SelectiveScanCuda": partial(selective_scan_flop_jit, backend="prefixsum", verbose=verbose),
#         }
# 
#         model = copy.deepcopy(self)
#         model.cuda().eval()
# 
#         input = torch.randn((1, *shape), device=next(model.parameters()).device)
#         params = parameter_count(model)[""]
#         Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
# 
#         del model, input
#         return sum(Gflops.values()) * 1e9
























































# -*- coding: utf-8 -*-


# 说明：
# - 这里假定 SS2D / GC_SS2D、LayerNorm / Linear / Permute / PatchMerge / Mlp / DropPath
#   以及 trunc_normal_、parameter_count、flop_count、selective_scan_flop_jit
#   已在同文件或其它模块中定义/导入。

#########################################################
# VSS Block
#########################################################

class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        channel_first: bool = False,
        ssm_d_state: int = 16,
        ssm_ratio: float = 2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer = nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias: bool = True,
        ssm_drop_rate: float = 0.0,
        ssm_init: str = "v0",
        forward_type: str = "gc",         # <<< 改：默认 'gc'
        mlp_ratio: float = 4.0,
        mlp_act_layer = nn.GELU,
        mlp_drop_rate: float = 0.0,
        use_checkpoint: bool = False,
        post_norm: bool = False,
        use_gc_ss2d: bool = True,         # <<< 改：默认使用 GC-SS2D
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        # 当启用 GC-SS2D 时，强制使用 'gc' 前向类型
        self._forward_type = "gc" if use_gc_ss2d else forward_type  # <<< 新增

        if self.ssm_branch:
            self.norm = LayerNorm(hidden_dim, channel_first=channel_first)

            # GC_SS2D or SS2D
            SSM_Module = GC_SS2D if use_gc_ss2d else SS2D

            self.op = SSM_Module(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=self._forward_type,   # <<< 改：使用规范化后的 forward_type
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            self.norm2 = LayerNorm(hidden_dim, channel_first=channel_first)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=hidden_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=mlp_act_layer,
                drop=mlp_drop_rate,
                channel_first=channel_first,
            )

    def _forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

##########################################################
# VSSM Model
##########################################################

class VSSM(nn.Module):
    def __init__(
        self,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 1000,
        depths = [2, 2, 9, 2],
        dims = [96, 192, 384, 768],
        ssm_d_state: int = 16,
        ssm_ratio: float = 2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer: str = "silu",
        ssm_conv: int = 3,
        ssm_conv_bias: bool = True,
        ssm_drop_rate: float = 0.0,
        ssm_init: str = "v0",
        forward_type: str = "gc",         # <<< 改：默认 'gc'
        mlp_ratio: float = 4.0,
        mlp_act_layer: str = "gelu",
        mlp_drop_rate: float = 0.0,
        gmlp: bool = False,
        drop_path_rate: float = 0.1,
        patch_norm: bool = True,
        norm_layer: str = "LN",
        downsample_version: str = "v2",
        patchembed_version: str = "v1",
        use_checkpoint: bool = False,
        posembed: bool = False,
        imgsize: int = 224,
        use_gc_ss2d: bool = True,         # <<< 改：默认使用 GC-SS2D
        **kwargs,
    ):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        _ACTLAYERS = dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )
        ssm_act_layer_mod: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer_mod: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        self.pos_embed = self._pos_embed(dims[0], patch_size, imgsize) if posembed else None
        self.patch_embed = self._make_patch_embed(
            in_chans, dims[0], patch_size, patch_norm,
            channel_first=self.channel_first, version=patchembed_version
        )

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = self._make_downsample(
                self.dims[i_layer],
                self.dims[i_layer + 1],
                channel_first=self.channel_first,
                version=downsample_version,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim=self.dims[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                downsample=downsample,
                channel_first=self.channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer_mod,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=("gc" if use_gc_ss2d else forward_type),  # <<< 改：层内也强制 'gc'
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer_mod,
                mlp_drop_rate=mlp_drop_rate,
                use_gc_ss2d=use_gc_ss2d,                               # <<< 传递
            ))

        self.classifier = nn.Sequential(OrderedDict(
            norm=LayerNorm(self.num_features, channel_first=self.channel_first),
            permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):
        patch_height, patch_width = (img_size // patch_size, img_size // patch_size)
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, channel_first=False, version="v1"):
        if version == "v1":
            return nn.Sequential(
                nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
                nn.Identity(),
                (LayerNorm(embed_dim, in_channel_first=True, out_channel_first=channel_first)
                 if patch_norm else (nn.Identity() if channel_first else Permute(0, 2, 3, 1))),
            )
        elif version == "v2":
            stride = patch_size // 2
            kernel_size = stride + 1
            padding = 1
            return nn.Sequential(
                nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.Identity(),
                (LayerNorm(embed_dim // 2, channel_first=True) if patch_norm else nn.Identity()),
                nn.Identity(),
                nn.GELU(),
                nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.Identity(),
                (LayerNorm(embed_dim, in_channel_first=True, out_channel_first=channel_first)
                 if patch_norm else (nn.Identity() if channel_first else Permute(0, 2, 3, 1))),
            )
        raise NotImplementedError

    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm=True, channel_first=False, version="v1"):
        if version == "v1":
            return nn.Sequential(
                PatchMerge(channel_first),
                LayerNorm(4 * dim, channel_first=channel_first) if norm else nn.Identity(),
                Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False, channel_first=channel_first),
            )
        elif version == "v2":
            return nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
                nn.Identity(),
                LayerNorm(out_dim, in_channel_first=True, out_channel_first=channel_first) if norm \
                    else (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif version == "v3":
            return nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
                nn.Identity(),
                LayerNorm(out_dim, in_channel_first=True, out_channel_first=channel_first) if norm \
                    else (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        raise NotImplementedError

    @staticmethod
    def _make_layer(
        dim=96,
        drop_path=[0.1, 0.1],
        use_checkpoint=False,
        downsample=nn.Identity(),
        channel_first=False,
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="gc",            # <<< 改：默认 'gc'
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        use_gc_ss2d=True,             # <<< 改：默认 True
        **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d],
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=("gc" if use_gc_ss2d else forward_type),  # <<< 同步
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
                use_gc_ss2d=use_gc_ss2d,                                 # <<< 传递
            ))

        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks),
            downsample=downsample,
        ))

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.permute(0, 2, 3, 1) if not self.channel_first else self.pos_embed
            x = x + pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x

    def flops(self, shape=(3, 224, 224), verbose=True):
        supported_ops = {
            "aten::silu": None,
            "aten::neg": None,
            "aten::exp": None,
            "aten::flip": None,
            "prim::PythonOp.SelectiveScanCuda": partial(
                selective_scan_flop_jit, backend="prefixsum", verbose=verbose
            ),
        }
        model = copy.deepcopy(self)
        model.cuda().eval()
        _input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(_input,), supported_ops=supported_ops)
        del model, _input
        return sum(Gflops.values()) * 1e9






















































            














##########################################################
# Flops Calculation Helper
##########################################################

def selective_scan_flop_jit(inputs, outputs, backend="prefixsum", verbose=True):
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = 9 * B * L * D * N + B * D * L
    return flops

##########################################################
# Model Registration
##########################################################

from timm.models import register_model

@register_model
def vmamba_tiny_gc(pretrained=False, **kwargs):
    """VMamba Tiny with GC_SS2D"""
    model = VSSM(
        depths=[2, 2, 9, 2], dims=96, drop_path_rate=0.2,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0,
        ssm_init="v0", forward_type="gc",
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0,
        patch_norm=True, norm_layer="ln",
        downsample_version="v2", patchembed_version="v1",
        use_checkpoint=False, posembed=False, imgsize=224,
        use_gc_ss2d=True,  # Enable GC_SS2D
        **kwargs
    )
    return model

@register_model
def vmamba_small_gc(pretrained=False, **kwargs):
    """VMamba Small with GC_SS2D"""
    model = VSSM(
        depths=[2, 2, 27, 2], dims=96, drop_path_rate=0.3,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0,
        ssm_init="v0", forward_type="gc",
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0,
        patch_norm=True, norm_layer="ln",
        downsample_version="v2", patchembed_version="v1",
        use_checkpoint=False, posembed=False, imgsize=224,
        use_gc_ss2d=True,  # Enable GC_SS2D
        **kwargs
    )
    return model

@register_model
def vmamba_base_gc(pretrained=False, **kwargs):
    """VMamba Base with GC_SS2D"""
    model = VSSM(
        depths=[2, 2, 27, 2], dims=128, drop_path_rate=0.6,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0,
        ssm_init="v0", forward_type="gc",
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0,
        patch_norm=True, norm_layer="ln",
        downsample_version="v2", patchembed_version="v1",
        use_checkpoint=False, posembed=False, imgsize=224,
        use_gc_ss2d=True,  # Enable GC_SS2D
        **kwargs
    )
    return model

# Original VMamba models (for comparison)
@register_model
def vmamba_tiny_s2l5(pretrained=False, channel_first=True, **kwargs):
    model = VSSM(
        depths=[2, 2, 5, 2], dims=96, drop_path_rate=0.2,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
        ssm_init="v0", forward_type="v05_noz",
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"),
        downsample_version="v3", patchembed_version="v2",
        use_checkpoint=False, posembed=False, imgsize=224,
        use_gc_ss2d=False,  # Use original SS2D
        **kwargs
    )
    return model

@register_model
def vmamba_small_s2l15(pretrained=False, channel_first=True, **kwargs):
    model = VSSM(
        depths=[2, 2, 15, 2], dims=96, drop_path_rate=0.3,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
        ssm_init="v0", forward_type="v05_noz",
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"),
        downsample_version="v3", patchembed_version="v2",
        use_checkpoint=False, posembed=False, imgsize=224,
        use_gc_ss2d=False,  # Use original SS2D
        **kwargs
    )
    return model

@register_model
def vmamba_base_s2l15(pretrained=False, channel_first=True, **kwargs):
    model = VSSM(
        depths=[2, 2, 15, 2], dims=128, drop_path_rate=0.6,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
        ssm_init="v0", forward_type="v05_noz",
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"),
        downsample_version="v3", patchembed_version="v2",
        use_checkpoint=False, posembed=False, imgsize=224,
        use_gc_ss2d=False,  # Use original SS2D
        **kwargs
    )
    return model







            
##########################################################
# Testing and Demo Functions
##########################################################

if __name__ == "__main__":
    # Test the model
    import torch
    
    # Test original VMamba
    print("Testing original VMamba Tiny...")
    model_orig = vmamba_tiny_s2l5()
    x = torch.randn(2, 3, 224, 224)
    y_orig = model_orig(x)
    print(f"Original output shape: {y_orig.shape}")
    
    # Test GC-VMamba
    print("\nTesting GC-VMamba Tiny...")
    model_gc = vmamba_tiny_gc()
    y_gc = model_gc(x)
    print(f"GC output shape: {y_gc.shape}")
    
    # Count parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nOriginal VMamba parameters: {count_parameters(model_orig):,}")
    print(f"GC-VMamba parameters: {count_parameters(model_gc):,}")
    
    print("\nModels created successfully!")