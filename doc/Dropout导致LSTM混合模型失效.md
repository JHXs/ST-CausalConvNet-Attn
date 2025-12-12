这是正常的，尤其是当数据量不大或者模型层数较深时。

  简单来说，当你在 num_layers=4 这样较深的模型中开启 dropout=0.25，每层之间都会随机丢弃 25% 的信息。对于时间序列预测（特别是 AQI
  这种），如果数据特征不够丰富或者样本量不足以支撑这么深的网络，过强的 Dropout
  会导致模型“学不到东西”（欠拟合），因为它破坏了太多连续的时间依赖关系。

  主要原因如下：

   1. 欠拟合 (Underfitting)： 你的模型有 4 层 LSTM，这对于大多数时间序列任务来说已经很深了。在层与层之间再加上
      Dropout，信号在传递过程中损失严重，导致梯度难以有效传播，模型无法收敛。
   2. 数据依赖性： 时间序列数据非常依赖前后文的连贯性。Dropout
      是随机断开连接，这在图像或文本分类中很有效（防止对某些特征过拟合），但在连续数值预测中，可能会打断这种微妙的时间依赖，尤其是
      在高 Dropout 率下。
   3. PyTorch 实现机制： nn.LSTM 的 dropout 参数只在 num_layers > 1 时生效，并且是作用在除了最后一层之外的每一层输出上。如果你只有
      2 层，第一层输出就被 Drop 了一次；如果有 4 层，前三层输出都会被 Drop。层数越多，累计的信号丢失就越严重。

  建议：

  如果想要防止过拟合但又不想模型失效，可以尝试以下调整：
   1. 降低 Dropout 率： 尝试 dropout=0.1 甚至 0.05，而不是 0.25。
   2. 减少层数： 对于 AQI 预测，通常 1 到 2 层 LSTM 就足够了。4 层往往过于复杂且难以训练。
   3. 使用 Weight Decay： 在优化器（如 Adam）中使用 weight_decay 参数（如 1e-4 或 1e-5）来做正则化，而不是完全依赖 Dropout。

   eg.
   ```python
   # 1. 选择 AdamW 优化器
   # 2. 通过 weight_decay 参数设置权重衰减的值
   LEARNING_RATE = 1e-3
   WEIGHT_DECAY = 1e-4  # 推荐从小值开始，例如 0.01、0.001 或 0.0001
   optimizer = optim.AdamW(
      model.parameters(),
      lr=LEARNING_RATE,
      weight_decay=WEIGHT_DECAY  # 在这里设置 weight_decay
   )
   ```