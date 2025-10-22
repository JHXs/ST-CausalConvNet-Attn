# coding:utf-8

param_grid = {
    'attention_heads': [1],
    'lr': [1e-5, 1e-4, 7e-4, 7.5e-4, 8e-4, 8.5e-4, 9e-4, 9.5e-4, 
           1e-3, 1.05e-3, 1.1e-3, 1.15e-3, 1.2e-3, 1.25e-3, 1.3e-3, 1.35e-3, 1.4e-3, 1.45e-3, 1.5e-3, 1.55e-3, 1.6e-3, 1.65e-3, 1.7e-3, 1.75e-3, 1.8e-3, 1.85e-3, 1.9e-3, 1.95e-3, 
           2e-3, 2.05e-3, 2.1e-3, 2.15e-3, 2.2e-3],
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
