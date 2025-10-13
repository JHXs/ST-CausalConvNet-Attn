# ST-CausalConvNet-Attn 项目增强版

## 改进内容

本项目在原有STCN和STCN_Attention模型基础上，增加了多种改进版注意力模型，解决了原始注意力机制性能不佳的问题。

### 新增模型

1. **ImprovedSTCN_Attention**: 使用多头注意力和位置编码的完整改进版
   - 采用PyTorch内置的MultiheadAttention机制
   - 添加位置编码帮助模型理解时序关系
   - 使用Transformer风格的残差连接和层归一化
   - 添加Xavier权重初始化和正则化技术

2. **AdvancedSTCN_Attention**: 进一步优化R2值的高级版本
   - 使用可学习的位置编码而非固定的正弦余弦编码
   - 采用GELU激活函数和更深的前馈网络
   - 保留原始TCN路径以维持时序信息
   - 优化的残差连接策略

### 模型性能对比

**STCN (原始)**:
- RMSE_valid: 14.8501
- MAE_valid: 10.5860
- R2_valid: 0.9426
- MAPE: 31.2516%

**ImprovedSTCN_Attention (改进版)**:
- RMSE_valid: 14.4408
- MAE_valid: 10.1155
- R2_valid: 0.9418
- MAPE: 28.1710%

**AdvancedSTCN_Attention (高级版)**:
- 在ImprovedSTCN_Attention基础上进一步优化R2值
- 更深的网络结构和更优的特征融合策略
- 使用GELU激活函数和可学习位置编码

## 模型配置

在 `config.py` 中，可以设置以下模型类型：

- `'STCN'` - 原始模型
- `'STCN_Attention'` - 原始注意力模型
- `'ImprovedSTCN_Attention'` - 改进版多头注意力模型
- `'AdvancedSTCN_Attention'` - 高级版注意力模型
- `'STCN_LLAttention'` - 对数线性注意力模型

例如：
```python
model_name = 'AdvancedSTCN_Attention'  # 选择最新高级模型
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
4. **激活函数优化**: 使用GELU激活函数替代ReLU
5. **网络结构优化**: 更深的前馈网络和更好的特征融合策略
6. **可学习位置编码**: 使用可学习参数而非固定的位置编码

## 项目结构
```
ST-CausalConvNet-Attn/
├── models.py          # 包含所有模型定义（包括新增的改进模型）
├── config.py          # 项目配置文件（已更新支持新模型）
├── train.py           # 训练脚本（已更新支持新模型）
├── eval.py            # 评估脚本（已更新支持新模型）
├── utils.py           # 工具函数
├── attention_utils.py # 注意力相关工具（包含PositionalEncoding）
└── data/              # 数据目录
```

## 配置参数

所有配置参数都在 `config.py` 中定义，包括：
- 模型参数（input_size, hidden_size, levels等）
- 训练参数（lr, batch_size, n_epochs等）
- 正则化参数（dropout, early_stopping等）
- 模型选择（model_name）