"""CBraMod model definition (inference only), vendored from the official implementation.

Source:   https://github.com/wjq-learning/CBraMod  (``models/cbramod.py``, ``models/criss_cross_transformer.py``)
Commit:   b9e961003214326972c567eff390e75b0287e32a (default-branch HEAD on 2026-10-07/08)
Paper:    Wang et al., "CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding", ICLR 2025 (arXiv 2412.07236)
Licence:  MIT, "Copyright (c) 2025 Jiquan Wang". The full MIT licence text follows and applies to this file.

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
    to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions
    of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
    TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
    CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

Modifications from upstream (functional behaviour and ``state_dict`` keys are unchanged; verified against the upstream
code in ``tests/test_embed_cbramod.py::test_vendored_model_matches_upstream`` when the upstream clone is cached):
  * training-only parts removed (weight initialisation, ``__main__`` demos, unused helper functions, CUDA calls);
  * ``torch`` is imported at module level, so import this module lazily (``sortinghat.embed.cbramod`` does);
  * the mask-token comparison ``mask == None`` is written ``mask is None``.

Input to ``CBraMod.forward``: ``x`` of shape (batch, channels, patches, 200): 200-sample patches (1 s at 200 Hz), the 19
TUEG channels in the order FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 FZ CZ PZ, signal in units of 100 uV
(upstream divides uV by 100). Output: (batch, channels, patches, 200) patch representations (``proj_out`` is the
upstream pretraining reconstruction head; downstream use, and this repo, replaces it with the identity).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class Criss_CrossTransformerEncoderLayer(nn.Module):  # upstream name: TransformerEncoderLayer
    """Criss-cross attention: half of the features attend across channels (spatial), half across patches (temporal)."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5):
        super().__init__()
        self.self_attn_s = nn.MultiheadAttention(d_model // 2, nhead // 2, dropout=dropout, bias=True, batch_first=True)
        self.self_attn_t = nn.MultiheadAttention(d_model // 2, nhead // 2, dropout=dropout, bias=True, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        x = src
        x = x + self._sa_block(self.norm1(x))
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self, x):
        bz, ch_num, patch_num, patch_size = x.shape
        xs = x[:, :, :, :patch_size // 2]
        xt = x[:, :, :, patch_size // 2:]
        xs = xs.transpose(1, 2).contiguous().view(bz * patch_num, ch_num, patch_size // 2)
        xt = xt.contiguous().view(bz * ch_num, patch_num, patch_size // 2)
        xs = self.self_attn_s(xs, xs, xs, need_weights=False)[0]
        xs = xs.contiguous().view(bz, patch_num, ch_num, patch_size // 2).transpose(1, 2)
        xt = self.self_attn_t(xt, xt, xt, need_weights=False)[0]
        xt = xt.contiguous().view(bz, ch_num, patch_num, patch_size // 2)
        x = torch.concat((xs, xt), dim=3)
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        return self.dropout2(x)


class Criss_CrossTransformerEncoder(nn.Module):  # upstream name: TransformerEncoder
    def __init__(self, encoder_layer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src):
        out = src
        for mod in self.layers:
            out = mod(out)
        return out


class PatchEmbedding(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, d_model: int, seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(19, 7), stride=(1, 1), padding=(9, 3),
                      groups=d_model))
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25), nn.GELU(),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25), nn.GELU(),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25), nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(nn.Linear(101, d_model), nn.Dropout(0.1))

    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        if mask is None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding          # the mask token is all zeros (not trainable)
        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(mask_x)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)

        mask_x = mask_x.contiguous().view(bz * ch_num * patch_num, patch_size)
        spectral = torch.fft.rfft(mask_x, dim=-1, norm="forward")
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, 101)
        patch_emb = patch_emb + self.spectral_proj(spectral)

        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)
        return patch_emb + positional_embedding


class CBraMod(nn.Module):
    def __init__(self, in_dim: int = 200, out_dim: int = 200, d_model: int = 200, dim_feedforward: int = 800,
                 seq_len: int = 30, n_layer: int = 12, nhead: int = 8):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_dim, out_dim, d_model, seq_len)
        layer = Criss_CrossTransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.encoder = Criss_CrossTransformerEncoder(layer, num_layers=n_layer)
        self.proj_out = nn.Sequential(nn.Linear(d_model, out_dim))

    def forward(self, x, mask=None):
        feats = self.encoder(self.patch_embedding(x, mask))
        return self.proj_out(feats)
