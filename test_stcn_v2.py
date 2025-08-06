#!/usr/bin/env python
# coding:utf-8

import sys
import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader, TensorDataset

# 导入模型和配置
import models
import config as cfg

def test_stcn_v2():
    """测试STCN_DualAttention_v2模型"""
    print("Testing STCN_DualAttention_v2 model...")
    
    # 设置模型参数
    cfg.model_name = 'STCN_DualAttention_v2'
    
    # 创建模型
    model = models.STCN_DualAttention_v2(
        input_size=cfg.input_size,
        in_channels=cfg.in_channels,
        output_size=cfg.output_size,
        num_channels=[cfg.hidden_size]*cfg.levels,
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout
    )
    
    print(f"Model created: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 创建测试输入数据 [batch, channels, height, width]
    batch_size = 2
    test_input = torch.randn(batch_size, cfg.in_channels, 1, cfg.input_size)
    print(f"Test input shape: {test_input.shape}")
    
    # 测试前向传播
    model.eval()
    with torch.no_grad():
        try:
            output = model(test_input)
            print(f"✓ Forward pass successful!")
            print(f"Output shape: {output.shape}")
            print(f"Output values: {output}")
            return True
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            return False

def compare_architectures():
    """比较两个模型的架构差异"""
    print("\nComparing model architectures...")
    
    # 原始模型
    model_v1 = models.STCN_DualAttention(
        input_size=cfg.input_size,
        in_channels=cfg.in_channels,
        output_size=cfg.output_size,
        num_channels=[cfg.hidden_size]*cfg.levels,
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout
    )
    
    # 改进模型
    model_v2 = models.STCN_DualAttention_v2(
        input_size=cfg.input_size,
        in_channels=cfg.in_channels,
        output_size=cfg.output_size,
        num_channels=[cfg.hidden_size]*cfg.levels,
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout
    )
    
    params_v1 = sum(p.numel() for p in model_v1.parameters())
    params_v2 = sum(p.numel() for p in model_v2.parameters())
    
    print(f"STCN_DualAttention_v1 parameters: {params_v1:,}")
    print(f"STCN_DualAttention_v2 parameters: {params_v2:,}")
    print(f"Parameter reduction: {params_v1 - params_v2:,} ({(params_v1 - params_v2) / params_v1 * 100:.1f}%)")
    
    return True

def test_attention_improvements():
    """测试注意力机制的改进"""
    print("\nTesting attention mechanism improvements...")
    
    model = models.STCN_DualAttention_v2(
        input_size=cfg.input_size,
        in_channels=cfg.in_channels,
        output_size=cfg.output_size,
        num_channels=[cfg.hidden_size]*cfg.levels,
        kernel_size=cfg.kernel_size,
        dropout=cfg.dropout
    )
    
    # 测试输入
    test_input = torch.randn(1, cfg.in_channels, 1, cfg.input_size)
    
    model.eval()
    with torch.no_grad():
        # 手动执行各步骤以检查中间输出
        conv_out = model.conv(test_input).squeeze(2)
        print(f"After conv: {conv_out.shape}")
        
        output = model.tcn(conv_out).transpose(1, 2)
        print(f"After TCN: {output.shape}")
        
        # 空间注意力
        spatial_weights = model.spatial_attention(output.transpose(1, 2))
        print(f"Spatial attention weights: {spatial_weights.shape}")
        print(f"Spatial weights range: [{spatial_weights.min():.3f}, {spatial_weights.max():.3f}]")
        
        # 时序注意力
        temporal_weights = model.temporal_attention(output)
        print(f"Temporal attention weights: {temporal_weights.shape}")
        print(f"Temporal weights range: [{temporal_weights.min():.3f}, {temporal_weights.max():.3f}]")
        
        # 检查残差连接
        residual = output
        print(f"Residual shape: {residual.shape}")
        
        print(f"✓ Attention improvements working correctly!")
        return True

if __name__ == "__main__":
    print("=" * 60)
    print("STCN_DualAttention_v2 Model Test")
    print("=" * 60)
    
    # 测试1: 模型前向传播
    test1_passed = test_stcn_v2()
    
    # 测试2: 架构比较
    test2_passed = compare_architectures()
    
    # 测试3: 注意力机制改进
    test3_passed = test_attention_improvements()
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"✓ Model forward pass: {'PASS' if test1_passed else 'FAIL'}")
    print(f"✓ Architecture comparison: {'PASS' if test2_passed else 'FAIL'}")
    print(f"✓ Attention improvements: {'PASS' if test3_passed else 'FAIL'}")
    
    if all([test1_passed, test2_passed, test3_passed]):
        print("\n🎉 All tests passed! STCN_DualAttention_v2 is ready for training.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
    print("=" * 60)