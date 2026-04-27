# coding:utf-8

import math

import torch
import torch.nn as nn

__all__ = ['PatchTST']


class _RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, subtract_last=False):
        super(_RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            self._get_statistics(x)
            return self._normalize(x)
        if mode == 'denorm':
            return self._denormalize(x)
        raise NotImplementedError

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1:, :].detach()
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class _Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super(_Transpose, self).__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, x):
        x = x.transpose(*self.dims)
        return x.contiguous() if self.contiguous else x


def _get_activation_fn(activation):
    if callable(activation):
        return activation()
    if activation.lower() == 'relu':
        return nn.ReLU()
    if activation.lower() == 'gelu':
        return nn.GELU()
    raise ValueError(f'{activation} is not available. Use "relu", "gelu", or a callable.')


class _MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(_MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class _SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(_SeriesDecomp, self).__init__()
        self.moving_avg = _MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


def _sincos_positional_encoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


def _coord_1d_positional_encoding(q_len, exponential=False, normalize=True):
    cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** (0.5 if exponential else 1)) - 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def _positional_encoding(pe, learn_pe, q_len, d_model):
    if pe is None:
        w_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(w_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        w_pos = torch.empty((q_len, 1))
        nn.init.uniform_(w_pos, -0.02, 0.02)
    elif pe == 'zeros':
        w_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(w_pos, -0.02, 0.02)
    elif pe in ('normal', 'gauss'):
        w_pos = torch.zeros((q_len, 1))
        nn.init.normal_(w_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        w_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(w_pos, a=0.0, b=0.1)
    elif pe == 'lin1d':
        w_pos = _coord_1d_positional_encoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d':
        w_pos = _coord_1d_positional_encoding(q_len, exponential=True, normalize=True)
    elif pe == 'sincos':
        w_pos = _sincos_positional_encoding(q_len, d_model, normalize=True)
    else:
        raise ValueError(f'{pe} is not a valid positional encoding.')
    return nn.Parameter(w_pos, requires_grad=learn_pe)


class _PatchTSTFlattenHead(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0.0):
        super(_PatchTSTFlattenHead, self).__init__()
        self.individual = individual
        self.n_vars = n_vars

        if individual:
            self.flattens = nn.ModuleList()
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            for _ in range(n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            return torch.stack(x_out, dim=1)
        x = self.flatten(x)
        x = self.linear(x)
        return self.dropout(x)


class _PatchTSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, norm='BatchNorm', attn_dropout=0.0,
                 dropout=0.0, activation='gelu', pre_norm=False):
        super(_PatchTSTEncoderLayer, self).__init__()
        if d_model % n_heads != 0:
            raise ValueError(f'd_model ({d_model}) must be divisible by n_heads ({n_heads}).')

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True
        )
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        if 'batch' in norm.lower():
            self.norm_attn = nn.Sequential(_Transpose(1, 2), nn.BatchNorm1d(d_model), _Transpose(1, 2))
            self.norm_ffn = nn.Sequential(_Transpose(1, 2), nn.BatchNorm1d(d_model), _Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)
            self.norm_ffn = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            _get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.pre_norm = pre_norm

    def forward(self, src):
        if self.pre_norm:
            src = self.norm_attn(src)
        src2, _ = self.self_attn(src, src, src, need_weights=False)
        src = src + self.dropout_attn(src2)
        if not self.pre_norm:
            src = self.norm_attn(src)

        if self.pre_norm:
            src = self.norm_ffn(src)
        src2 = self.ff(src)
        src = src + self.dropout_ffn(src2)
        if not self.pre_norm:
            src = self.norm_ffn(src)
        return src


class _PatchTSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, norm='BatchNorm', attn_dropout=0.0,
                 dropout=0.0, activation='gelu', n_layers=3, pre_norm=False):
        super(_PatchTSTEncoder, self).__init__()
        self.layers = nn.ModuleList([
            _PatchTSTEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                norm=norm,
                attn_dropout=attn_dropout,
                dropout=dropout,
                activation=activation,
                pre_norm=pre_norm
            )
            for _ in range(n_layers)
        ])

    def forward(self, src):
        output = src
        for layer in self.layers:
            output = layer(output)
        return output


class _PatchTSTChannelEncoder(nn.Module):
    def __init__(self, patch_num, patch_len, n_layers=3, d_model=128, n_heads=8,
                 d_ff=256, norm='BatchNorm', attn_dropout=0.0, dropout=0.0,
                 act='gelu', pre_norm=False, pe='zeros', learn_pe=True):
        super(_PatchTSTChannelEncoder, self).__init__()
        self.W_P = nn.Linear(patch_len, d_model)
        self.W_pos = _positional_encoding(pe, learn_pe, patch_num, d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder = _PatchTSTEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            activation=act,
            n_layers=n_layers,
            pre_norm=pre_norm
        )

    def forward(self, x):
        n_vars = x.shape[1]
        x = x.permute(0, 1, 3, 2)
        x = self.W_P(x)
        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        u = self.dropout(u + self.W_pos)
        z = self.encoder(u)
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))
        return z.permute(0, 1, 3, 2)


class _PatchTSTBackbone(nn.Module):
    def __init__(self, c_in, context_window, target_window, patch_len, stride,
                 n_layers=3, d_model=128, n_heads=8, d_ff=256, norm='BatchNorm',
                 attn_dropout=0.0, dropout=0.0, act='gelu', pre_norm=False,
                 pe='zeros', learn_pe=True, head_dropout=0.0, padding_patch='end',
                 individual=False, revin=True, affine=True, subtract_last=False):
        super(_PatchTSTBackbone, self).__init__()
        if context_window < patch_len:
            raise ValueError(f'context_window ({context_window}) must be >= patch_len ({patch_len}).')
        if stride <= 0:
            raise ValueError('stride must be positive.')

        self.revin = revin
        if revin:
            self.revin_layer = _RevIN(c_in, affine=affine, subtract_last=subtract_last)

        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        self.backbone = _PatchTSTChannelEncoder(
            patch_num=patch_num,
            patch_len=patch_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act=act,
            pre_norm=pre_norm,
            pe=pe,
            learn_pe=learn_pe
        )
        self.head = _PatchTSTFlattenHead(
            individual=individual,
            n_vars=c_in,
            nf=d_model * patch_num,
            target_window=target_window,
            head_dropout=head_dropout
        )

    def forward(self, z):
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)

        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = z.permute(0, 1, 3, 2)

        z = self.backbone(z)
        z = self.head(z)

        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)
        return z


class PatchTST(nn.Module):
    """
    PatchTST forecaster adapted from the official supervised PatchTST design.
    Input can be [batch, seq_len, input_size] or [batch, input_size, seq_len].
    By default it returns [batch, output_size] to match this repository's y shape.
    """
    def __init__(self, input_size=None, output_size=None, seq_len=None, c_in=None, c_out=None,
                 pred_len=None, pred_dim=None, patch_len=16, stride=8, n_layers=3,
                 e_layers=None, d_model=128, n_heads=8, d_ff=256, dropout=0.1,
                 fc_dropout=None, head_dropout=None, attn_dropout=0.0, norm='BatchNorm',
                 act='gelu', padding_patch='end', revin=True, affine=True,
                 subtract_last=False, individual=False, decomposition=False,
                 kernel_size=25, flatten_output=True, pe='zeros', learn_pe=True,
                 pre_norm=False, **kwargs):
        super(PatchTST, self).__init__()

        input_size = input_size if input_size is not None else c_in
        output_size = output_size if output_size is not None else pred_dim
        output_size = output_size if output_size is not None else pred_len
        output_size = output_size if output_size is not None else c_out
        n_layers = e_layers if e_layers is not None else n_layers
        fc_dropout = dropout if fc_dropout is None else fc_dropout
        head_dropout = dropout if head_dropout is None else head_dropout

        if input_size is None or output_size is None or seq_len is None:
            raise ValueError('PatchTST requires input_size/c_in, output_size/pred_len, and seq_len.')

        self.input_size = input_size
        self.output_size = output_size
        self.seq_len = seq_len
        self.flatten_output = flatten_output
        self.decomposition = decomposition

        backbone_kwargs = dict(
            c_in=input_size,
            context_window=seq_len,
            target_window=output_size,
            patch_len=patch_len,
            stride=stride,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act=act,
            pre_norm=pre_norm,
            pe=pe,
            learn_pe=learn_pe,
            head_dropout=head_dropout,
            padding_patch=padding_patch,
            individual=individual,
            revin=revin,
            affine=affine,
            subtract_last=subtract_last
        )

        if decomposition:
            self.decomp_module = _SeriesDecomp(kernel_size)
            self.model_res = _PatchTSTBackbone(**backbone_kwargs)
            self.model_trend = _PatchTSTBackbone(**backbone_kwargs)
        else:
            self.model = _PatchTSTBackbone(**backbone_kwargs)

        if flatten_output:
            self.output_projection = nn.Sequential(
                nn.Dropout(fc_dropout),
                nn.Linear(input_size * output_size, output_size)
            )

    def _as_sequence_first(self, x):
        if x.dim() != 3:
            raise ValueError('PatchTST expects a 3D tensor.')
        if x.size(1) == self.seq_len and x.size(2) == self.input_size:
            return x
        if x.size(1) == self.input_size and x.size(2) == self.seq_len:
            return x.transpose(1, 2)
        raise ValueError(
            f'PatchTST expected [batch, {self.seq_len}, {self.input_size}] '
            f'or [batch, {self.input_size}, {self.seq_len}], got {tuple(x.shape)}.'
        )

    def forward(self, x):
        x = self._as_sequence_first(x)
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            z = self.model_res(res_init.transpose(1, 2)) + self.model_trend(trend_init.transpose(1, 2))
        else:
            z = self.model(x.transpose(1, 2))

        if self.flatten_output:
            return self.output_projection(z.flatten(1))
        return z.transpose(1, 2)
