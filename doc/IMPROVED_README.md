# ST-CausalConvNet-Attn 项目增强版

## 改进内容

本项目在原有STCN和STCN_Attention模型基础上，增加了两种改进版注意力模型，解决了原始注意力机制性能不佳的问题。

### 新增模型

1. **ImprovedSTCN_Attention**: 使用多头注意力和位置编码的完整改进版
   - 采用PyTorch内置的MultiheadAttention机制
   - 添加位置编码帮助模型理解时序关系
   - 使用Transformer风格的残差连接和层归一化
   - 添加Xavier权重初始化和正则化技术

2. **SimplifiedSTCN_Attention**: 简化版注意力模型
   - 使用简单可学习的注意力权重，减少参数数量
   - 对时间维度求平均而不是求和，避免数值问题
   - 减少过拟合风险

## 模型配置

在 `config.py` 中，可以设置以下模型类型：

- `'STCN'` - 原始模型
- `'STCN_Attention'` - 原始注意力模型
- `'ImprovedSTCN_Attention'` - 改进版多头注意力模型
- `'SimplifiedSTCN_Attention'` - 简化版注意力模型
- `'STCN_LLAttention'` - 对数线性注意力模型

例如：
```python
model_name = 'ImprovedSTCN_Attention'  # 选择改进版模型
```

## 训练和评估

### 训练模型
```bash
python train.py
```

### 评估模型
```bash
python eval.py
```

## 改进效果

通过以下方式改进了原始STCN_Attention模型的性能：

1. **更好的注意力机制**: 使用经过验证的多头注意力机制，而非自定义的复杂注意力
2. **位置编码**: 添加位置信息帮助模型理解时序
3. **正则化技术**: 包括残差连接、层归一化、Dropout等，提高模型稳定性
4. **权重初始化**: 使用Xavier初始化防止梯度消失/爆炸
5. **简化的注意力**: 提供参数更少的简化版本，降低过拟合风险

## 项目结构
```
ST-CausalConvNet-Attn/
├── models.py          # 包含所有模型定义（包括新增的改进模型）
├── config.py          # 项目配置文件（已更新支持新模型）
├── train.py           # 训练脚本（已更新支持新模型）
├── eval.py            # 评估脚本（已更新支持新模型）
├── utils.py           # 工具函数
├── attention_utils.py # 注意力相关工具
└── data/              # 数据目录
```

## 配置参数

所有配置参数都在 `config.py` 中定义，包括：
- 模型参数（input_size, hidden_size, levels等）
- 训练参数（lr, batch_size, n_epochs等）
- 正则化参数（dropout, early_stopping等）
- 模型选择（model_name）