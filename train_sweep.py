# coding:utf-8

import itertools
import os
import traceback
import time
import numpy as np
import torch

import config as cfg
import utils
import models
import train as train_mod
import eval as eval_mod
from sweep_config import param_grid, options


def _make_run_id(params: dict) -> str:
    def fmt(v):
        if isinstance(v, float):
            s = f"{v:.0e}" if v < 1e-2 or v >= 1 else f"{v}"
            return s.replace('+', '').replace('.', 'p')
        return str(v)
    ordered = ['attention_heads', 'lr', 'hidden_size', 'levels', 'kernel_size', 'dropout', 'rand_seed']
    parts = []
    for k in ordered:
        if k in params:
            parts.append(f"{k[:2]}{fmt(params[k])}")
    return '_'.join(parts)


def _is_valid_combo(params: dict) -> (bool, str):
    # MultiheadAttention 整除约束：embed_dim = hidden_size 必须能被 attention_heads 整除
    model_name = cfg.model_name
    heads = params.get('attention_heads', None)
    hidden = params.get('hidden_size', None)
    if heads is not None and hidden is not None:
        if model_name in ['ImprovedSTCN_Attention', 'AdvancedSTCN_Attention']:
            if hidden % heads != 0:
                return False, f"hidden_size {hidden} % attention_heads {heads} != 0"
    return True, ''


def _product(grid: dict):
    keys = list(grid.keys())
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def _ensure_summary_dir() -> str:
    date_str = time.strftime('%m%d')
    base = os.path.join('./reports', date_str)
    os.makedirs(base, exist_ok=True)
    return base


def _append_summary(base_dir: str, row: dict):
    import csv
    fpath = os.path.join(base_dir, 'summary.csv')
    exists = os.path.exists(fpath)
    fieldnames = [
        'run_id', 'status', 'reason',
        'model_name', 'lr', 'attention_heads', 'hidden_size', 'levels', 'kernel_size', 'dropout', 'rand_seed',
        'rmse', 'mae', 'r2', 'mape', 'smape', 'mase', 'coverage', 'best_epoch', 'early_stop_epoch', 'elapsed_seconds'
    ]
    with open(fpath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        # Ensure percentage fields carry a trailing '%'
        data = {k: row.get(k, '') for k in fieldnames}
        for key in ('mape', 'smape', 'coverage'):
            val = data.get(key, '')
            if val == '' or val is None:
                continue
            # Keep KISS: append '%' without altering precision
            sval = str(val)
            if not sval.endswith('%'):
                data[key] = f"{sval}%"
        writer.writerow(data)


def main():
    base_dir = _ensure_summary_dir()

    # 固定模型名为当前 config
    model_name = cfg.model_name

    for params in _product(param_grid):
        valid, reason = _is_valid_combo(params)
        run_id = _make_run_id(params)

        row = {
            'run_id': run_id,
            'status': 'pending',
            'reason': '',
            'model_name': model_name,
            **{k: params.get(k) for k in param_grid.keys()},
            'elapsed_seconds': '',
            'best_epoch': '',
            'early_stop_epoch': ''
        }

        if not valid:
            row['status'] = 'skipped'
            row['reason'] = reason
            _append_summary(base_dir, row)
            print(f"Skip {run_id}: {reason}")
            continue

        try:
            print(f"\n==== Run {run_id} ====")
            # 设定 cfg 参数
            cfg.lr = params.get('lr', cfg.lr)
            cfg.attention_heads = params.get('attention_heads', cfg.attention_heads)
            cfg.hidden_size = params.get('hidden_size', cfg.hidden_size)
            cfg.levels = params.get('levels', cfg.levels)
            cfg.kernel_size = params.get('kernel_size', cfg.kernel_size)
            cfg.dropout = params.get('dropout', cfg.dropout)
            cfg.rand_seed = params.get('rand_seed', cfg.rand_seed)
            # 更新保存路径
            cfg.model_save_pth = f"./models/model_{cfg.model_name}_{run_id}.pth"

            # 设定报告 run_id
            utils.set_run_id(run_id)

            # 固定随机种子
            np.random.seed(cfg.rand_seed)
            torch.manual_seed(cfg.rand_seed)

            # 加载数据
            print('\nLoading data...\n')
            train_loader, valid_loader, test_loader = utils.load_data(f_x=cfg.f_x, f_y=cfg.f_y, batch_size=cfg.batch_size)

            # 生成模型
            if cfg.model_name == 'RNN':
                net = models.SimpleRNN(input_size=cfg.input_size, hidden_size=cfg.hidden_size, output_size=cfg.output_size, num_layers=cfg.num_layers)
            elif cfg.model_name == 'GRU':
                net = models.SimpleGRU(input_size=cfg.input_size, hidden_size=cfg.hidden_size, output_size=cfg.output_size, num_layers=cfg.num_layers)
            elif cfg.model_name == 'LSTM':
                net = models.SimpleLSTM(input_size=cfg.input_size, hidden_size=cfg.hidden_size, output_size=cfg.output_size, num_layers=cfg.num_layers)
            elif cfg.model_name == 'TCN':
                net = models.TCN(input_size=cfg.input_size, output_size=cfg.output_size, num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout)
            elif cfg.model_name == 'TCN_Attention':
                net = models.TCN_Attention(input_size=cfg.input_size, output_size=cfg.output_size, num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout)
            elif cfg.model_name == 'STCN':
                net = models.STCN(input_size=cfg.input_size, in_channels=cfg.in_channels, output_size=cfg.output_size,
                                  num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout)
            elif cfg.model_name == 'STCN_Attention':
                net = models.STCN_Attention(input_size=cfg.input_size, in_channels=cfg.in_channels, output_size=cfg.output_size,
                                             num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout,
                                             attention_heads=cfg.attention_heads, use_rotary=cfg.use_rotary)
            elif cfg.model_name == 'ImprovedSTCN_Attention':
                net = models.ImprovedSTCN_Attention(input_size=cfg.input_size, in_channels=cfg.in_channels, output_size=cfg.output_size,
                                                     num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout,
                                                     attention_heads=cfg.attention_heads)
            elif cfg.model_name == 'AdvancedSTCN_Attention':
                net = models.AdvancedSTCN_Attention(input_size=cfg.input_size, in_channels=cfg.in_channels, output_size=cfg.output_size,
                                                     num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout,
                                                     attention_heads=cfg.attention_heads)
            elif cfg.model_name == 'STCN_LLAttention':
                net = models.STCN_LLAttention(input_size=cfg.input_size, in_channels=cfg.in_channels, output_size=cfg.output_size,
                                             num_channels=[cfg.hidden_size]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout,
                                             attention_heads=cfg.attention_heads, use_rotary=cfg.use_rotary, htype='weak', base=2)
            else:
                raise ValueError(f"Unsupported model_name: {cfg.model_name}")

            net = net.to(cfg.device)

            # 训练
            t0 = time.time()
            meta = train_mod.train(net, train_loader, valid_loader, test_loader, plot=options.get('plots', True))
            t1 = time.time()

            # 评估
            net.load_state_dict(torch.load(cfg.model_save_pth, map_location=cfg.device))
            metrics = eval_mod.eval(net, test_loader, plot=options.get('plots', True))

            row.update({
                'status': 'ok',
                'rmse': metrics[0],
                'mae': metrics[1],
                'r2': metrics[2],
                'mape': metrics[3],
                'smape': metrics[4],
                'mase': metrics[5],
                'coverage': metrics[6],
                'elapsed_seconds': meta.get('elapsed_seconds', t1 - t0),
                'best_epoch': meta.get('best_epoch'),
                'early_stop_epoch': meta.get('early_stop_epoch'),
            })
        except Exception as e:
            row['status'] = 'failed'
            row['reason'] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            _append_summary(base_dir, row)
            # 清理 run_id，避免影响下一轮
            utils.set_run_id(None)


if __name__ == '__main__':
    main()
