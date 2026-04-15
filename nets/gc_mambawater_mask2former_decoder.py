# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .gc_vss import VSSM
except ImportError:
    from gc_vss import VSSM

try:
    from .spcii_connect_blocks import Connect, SPCII_Attention
except ImportError:
    from spcii_connect_blocks import Connect, SPCII_Attention


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, x):
        b, _, h, w = x.shape
        device = x.device
        mask = torch.zeros((b, h, w), dtype=torch.bool, device=device)
        not_mask = ~mask

        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2).contiguous()
        return pos


class MaskEmbeddingMLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.layers(x)


class Mask2FormerDecoderLayer(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, ffn_dim=1024, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(hidden_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, query, query_pos, memory, memory_pos):
        q = query + query_pos
        query2 = self.self_attn(q, q, query, need_weights=False)[0]
        query = self.norm1(query + self.dropout(query2))

        query2 = self.cross_attn(query + query_pos, memory + memory_pos, memory, need_weights=False)[0]
        query = self.norm2(query + self.dropout(query2))

        query2 = self.linear2(self.dropout(F.relu(self.linear1(query), inplace=True)))
        query = self.norm3(query + self.dropout(query2))
        return query


class Mask2FormerSPCIIConnectDecoder(nn.Module):
    """
    Mask2Former-style decoder + one SPCII refinement on mask_features + Connect branch.
    The SPCII and Connect are both attached to the final high-resolution mask feature map.
    """
    def __init__(self,
                 num_classes=2,
                 in_channels=(96, 192, 384, 768),
                 hidden_dim=256,
                 num_queries=16,
                 num_decoder_layers=6,
                 num_heads=8,
                 ffn_dim=1024,
                 dropout_ratio=0.1,
                 use_connect_branch=True,
                 use_mask_spcii=True,
                 num_groups_1d_conv_in_spcii=4):
        super().__init__()
        c1_in, c2_in, c3_in, c4_in = in_channels
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.use_connect_branch = use_connect_branch

        self.input_proj_c1 = ConvBNAct(c1_in, hidden_dim, kernel_size=1)
        self.input_proj_c2 = ConvBNAct(c2_in, hidden_dim, kernel_size=1)
        self.input_proj_c3 = ConvBNAct(c3_in, hidden_dim, kernel_size=1)
        self.input_proj_c4 = ConvBNAct(c4_in, hidden_dim, kernel_size=1)

        self.out_conv_p4 = ConvBNAct(hidden_dim, hidden_dim, kernel_size=3)
        self.out_conv_p3 = ConvBNAct(hidden_dim, hidden_dim, kernel_size=3)
        self.out_conv_p2 = ConvBNAct(hidden_dim, hidden_dim, kernel_size=3)
        self.out_conv_p1 = ConvBNAct(hidden_dim, hidden_dim, kernel_size=3)
        self.mask_feature = ConvBNAct(hidden_dim, hidden_dim, kernel_size=3)

        if use_mask_spcii:
            self.mask_spcii = SPCII_Attention(hidden_dim, num_groups_1d_conv=num_groups_1d_conv_in_spcii)
        else:
            self.mask_spcii = nn.Identity()

        self.position_embedding = PositionEmbeddingSine(num_pos_feats=hidden_dim // 2)
        self.level_embed = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)
        self.query_content = nn.Embedding(num_queries, hidden_dim)
        self.query_position = nn.Embedding(num_queries, hidden_dim)

        self.decoder_layers = nn.ModuleList([
            Mask2FormerDecoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=0.0,
            )
            for _ in range(num_decoder_layers)
        ])

        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.mask_embed = MaskEmbeddingMLP(hidden_dim)

        self.connect = Connect(num_classes, num_neighbor=9, embedding_dim=hidden_dim, dropout_ratio=dropout_ratio) \
            if use_connect_branch else None

    def _flatten_memory(self, feat, level_index):
        pos = self.position_embedding(feat)
        feat = feat.flatten(2).transpose(1, 2).contiguous()
        pos = pos.flatten(2).transpose(1, 2).contiguous()
        feat = feat + self.level_embed[level_index].view(1, 1, -1)
        return feat, pos

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        p4 = self.out_conv_p4(self.input_proj_c4(c4))
        p3 = self.out_conv_p3(self.input_proj_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False))
        p2 = self.out_conv_p2(self.input_proj_c2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False))
        p1 = self.out_conv_p1(self.input_proj_c1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False))

        mask_features = self.mask_feature(p1)
        mask_features = self.mask_spcii(mask_features)

        memory_levels = [p4, p3, p2]
        memories = []
        positions = []
        for level_index, feat in enumerate(memory_levels):
            mem, pos = self._flatten_memory(feat, level_index)
            memories.append(mem)
            positions.append(pos)

        bs = c1.shape[0]
        query = self.query_content.weight.unsqueeze(0).repeat(bs, 1, 1)
        query_pos = self.query_position.weight.unsqueeze(0).repeat(bs, 1, 1)

        for layer_index, layer in enumerate(self.decoder_layers):
            mem_index = layer_index % len(memories)
            query = layer(
                query=query,
                query_pos=query_pos,
                memory=memories[mem_index],
                memory_pos=positions[mem_index],
            )

        class_logits = self.class_embed(query)
        mask_embed = self.mask_embed(query)
        mask_logits = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        seg = torch.einsum("bqc,bqhw->bchw", class_logits, mask_logits)

        if self.connect is not None:
            aux_seg, con0, con1 = self.connect(mask_features)
            seg = seg + aux_seg
            return seg, con0, con1
        return seg, None, None


class SegWater_VMamba_Mask2FormerSPCIIConnectDecoder(nn.Module):
    def __init__(self,
                 num_classes=2,
                 vmamba_variant="vmamba_small_s2l15",
                 embedding_dim=256,
                 decoder_embedding_dim=None,
                 num_queries=16,
                 num_decoder_layers=6,
                 num_heads=8,
                 ffn_dim=1024,
                 use_gradient_checkpointing=False,
                 use_gc_ss2d=True,
                 use_connect_branch=True,
                 use_mask_spcii=True,
                 num_groups_1d_conv_in_spcii=4,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        if decoder_embedding_dim is not None:
            embedding_dim = decoder_embedding_dim
        self.embedding_dim = embedding_dim

        variant_cfgs = {
            "vmamba_tiny_s2l5": dict(depths=[2, 2, 5, 2], dims=96, drop_path_rate=0.2),
            "vmamba_small_s2l15": dict(depths=[2, 2, 15, 2], dims=96, drop_path_rate=0.3),
            "vmamba_base_s2l15": dict(depths=[2, 2, 15, 2], dims=128, drop_path_rate=0.6),
            "vmamba_small_s1l20": dict(depths=[2, 2, 20, 2], dims=96, drop_path_rate=0.3),
            "vmamba_base_s1l20": dict(depths=[2, 2, 20, 2], dims=128, drop_path_rate=0.5),
        }

        if vmamba_variant not in variant_cfgs:
            raise ValueError(f"Unsupported vmamba_variant: {vmamba_variant}")
        cfg = variant_cfgs[vmamba_variant]
        model_depths = cfg["depths"]
        model_dims = cfg["dims"]
        drop_path_rate = cfg["drop_path_rate"]

        if isinstance(model_dims, int):
            encoder_channels = [model_dims, model_dims * 2, model_dims * 4, model_dims * 8]
        else:
            encoder_channels = model_dims
        self.encoder_channels = encoder_channels

        self.encoder = VSSM(
            depths=model_depths,
            dims=model_dims,
            ssm_d_state=1,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=False,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v05_noz",
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            drop_path_rate=drop_path_rate,
            patch_norm=True,
            norm_layer="ln2d",
            downsample_version="v3",
            patchembed_version="v2",
            use_checkpoint=use_gradient_checkpointing,
            posembed=False,
            imgsize=224,
            use_gc_ss2d=use_gc_ss2d,
        )

        self.decode_head = Mask2FormerSPCIIConnectDecoder(
            num_classes=num_classes,
            in_channels=encoder_channels,
            hidden_dim=self.embedding_dim,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            use_connect_branch=use_connect_branch,
            use_mask_spcii=use_mask_spcii,
            num_groups_1d_conv_in_spcii=num_groups_1d_conv_in_spcii,
        )

    def forward(self, x):
        height, width = x.size(2), x.size(3)
        x = self.encoder.patch_embed(x)

        features = []
        for layer in self.encoder.layers:
            x = layer.blocks(x)
            features.append(x)
            x = layer.downsample(x)

        c1, c2, c3, c4 = features
        seg, con0, con1 = self.decode_head((c1, c2, c3, c4))
        seg = F.interpolate(seg, size=(height, width), mode="bilinear", align_corners=False)
        if con0 is not None and con1 is not None:
            con0 = F.interpolate(con0, size=(height, width), mode="bilinear", align_corners=False)
            con1 = F.interpolate(con1, size=(height, width), mode="bilinear", align_corners=False)

        if self.training:
            return seg, con0, con1
        return seg
