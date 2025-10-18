# coding:utf-8

param_grid = {
    'lr': [1e-3],
    'attention_heads': [1, 2, 4, 8, 16, 32],
    # 'hidden_size': [32, 64],
    # 'levels': [3, 4],
    # 'kernel_size': [3, 4],
    # 'dropout': [0.2, 0.25],
    # 'rand_seed': [314, 42],
}

# 是否启用图表和报告
options = {
    'plots': True,
    'reports': True,
}

