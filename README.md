# ST-CausalConvNet-Attention
基于改进注意力机制的时空因果卷积网络时间序列预测方法
以中国北京 PM2.5 为例。

## 准备

- Python3
- Numpy
- Pandas
- PyTorch (>= 1.6.0)
- 推荐使用 `miniforge`
  - 安装
    ```shell
    sudo pacman miniforge # 需要添加 arch4edu 源
    ```
    初始化
    ```shell
    conda init
    ```
  - 创建虚拟环境
    ```shell
    conda create -n myproj python=3.10
    conda activate myproj # 激活虚拟环境
    # 按需手动装包： pip install xxx
    ```
    `-n`: 虚拟环境名字
    `python`: 指定 python 版本

## 模型架构

ST-CausalConvNet-Attention 的架构包括三部分：（1）多个监测站时空信息的整合；（2）因果卷积网络（对于以下模型架构示例，核大小 = 3，扩张数 = 1、2 和 4）；（3）注意力层。

![Model structure](./ST-CausalConvNet_Architecture.jpg)

## 文件结构和数据描述

- **data 文件夹**:
  - **microsoft_urban_air_data**: 来自微软研究院城市计算团队的空气质量数据集（有关如何使用该数据集的更多帮助，请参阅[网页](http://research.microsoft.com/en-us/projects/urbanair)）。
  - **stations_data**: 北京各站点的数据分别存储在该目录中。
  - **xy**: X 和 y 矩阵（保存为 pickle 文件格式）是经过 `data_process.py` 处理得到的用于深度学习模型的输入。
- **models 文件夹**: 存储训练得到的最佳模型的文件夹。
- **doc 文件夹**：存储文档。
- **reports 文件夹**：存储生成的可视化以及报告。
- **config.py**: 用于设置输入数据位置、模型参数和模型存储路径的配置文件。
- **data_process.py**: 用于提取选定的中心站和相关性较高的其他站的数据，并将原始数据转化为高维矩阵，以匹配模型的输入结构。
- **models.py**: 用于生成预测任务的 ST-CausalConvNet-Attention 的核心函数。模型结构可参考论文。此外，它还包含其他模型（SimpleRNN、GRU 和 LSTM）以供比较。
- **attention_utils.py**：为 `models.py` 提供生成模型的相应类和函数。
- **train.py**: 读取参数、数据准备和训练程序。
- **eval.py**: 用于评估测试集上的模型性能。
- **utils.py**: 它包含用于训练、验证的数据加载、生成批量数据、可视化和生成报告等功能。

## 使用介绍

#### 配置

所有模型参数都可以在 `config.py` 中设置，例如学习率、批量大小、层数、内核大小、注意力头数、早停等。

#### 数据处理

```python
python data_process_me.py
```

程序对原始数据集进行处理，并将其保存为 `Xy` 文件夹中的pkl文件，以供后续的模型训练和验证。

#### 训练模型

```python
python train.py
```

程序可以自动将最佳（验证集上的 RMSE 最低）模型保存在 `models` 目录中。

#### 评估

```python
python eval.py
```

加载保存的模型并在测试集上进行评估。

## License

[Apache License v2.0](./LICENSE)

## Thanks

We thanks the previous work [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271) by Shaojie Bai, J. Zico Kolter and Vladlen Koltun, as the basic knowledge for TCN architecture in our research.


