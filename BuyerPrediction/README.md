# 买家意图预测比赛

Kaggle 比赛 [Buyer Intent Prediction Competition](https://www.kaggle.com/competitions/buyer-intent-prediction-competition/overview)的代码。

## 项目概述

本项目实现了三种不同的微调方式来预测用户的购买意图：

1. **全参数微调** (Full Parameter Fine-tuning) - 已完成 ✅
2. **LoRA** (Low-Rank Adaptation) - 待实现
3. **待定** - 待确定

## 文件结构

```
BuyerPrediction/
├── full_finetune.py          # 全参数微调主脚本（支持DDP多卡训练）
├── config.py                 # 配置文件
├── requirements.txt          # 依赖包列表
├── example_usage.py          # 使用示例
├── README_full_finetune.md   # 全参数微调详细说明
└── README.md                 # 项目总览（本文件）
```

## 微调方法对比

| 方法 | 状态 | 参数量 | 训练速度 | 内存占用 | 预期性能 |
|------|------|--------|----------|----------|----------|
| 全参数微调 | ✅ 已完成 | 全部 | 慢 | 高 | 最好 |
| LoRA | 🔄 待实现 | 少量 | 快 | 低 | 好 |
| 待定 | ⏳ 待确定 | - | - | - | - |

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活conda环境
conda activate llm

# 安装依赖
pip install -r requirements.txt

# 检查GPU
nvidia-smi
```

### 2. 数据准备

数据文件已解压到 `../data/` 目录：
- `buyer_intent_dataset_kaggle_sample_submission.csv` - 训练数据
- `buyer_intent_dataset_kaggle_test.csv` - 测试数据

数据格式：
- `Query`: 用户查询文本
- `Intent`: 意图标签（7个类别）

### 3. 运行训练

#### 方法一：单卡训练（调试用）

```bash
cd BuyerPrediction
python full_finetune.py
```

#### 方法二：多卡训练（推荐）

```bash
# 设置要使用的GPU（根据你的GPU编号调整）
export CUDA_VISIBLE_DEVICES=1,2,3,5

# 启动4卡分布式训练
torchrun --nproc_per_node=4 full_finetune.py

# 或者指定端口（避免端口冲突）
torchrun --nproc_per_node=4 --master_port=29501 full_finetune.py
```

#### 方法三：使用示例脚本

```bash
cd BuyerPrediction
python example_usage.py
```

## 📋 运行参数说明

### GPU设置
- `CUDA_VISIBLE_DEVICES`: 指定使用的GPU编号
- `--nproc_per_node`: 使用的GPU数量
- `--master_port`: 主进程端口（避免冲突）

### 训练参数（在config.py中调整）
- `BATCH_SIZE`: 批次大小（多卡时是每张卡的批次大小）
- `LEARNING_RATE`: 学习率
- `EPOCHS`: 训练轮数
- `MAX_LENGTH`: 最大序列长度

## 🔧 常见问题

### 1. 内存不足
```bash
# 减小批次大小
# 在config.py中修改 BATCH_SIZE = 4
```

### 2. 端口冲突
```bash
# 使用不同端口
torchrun --nproc_per_node=4 --master_port=29502 full_finetune.py
```

### 3. GPU不可见
```bash
# 检查GPU设置
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
```

### 4. 分布式训练失败
```bash
# 确保所有GPU都可用
nvidia-smi
# 检查网络连接（多机训练时需要）
```

## 📊 输出文件

训练完成后会生成：
- `../models/best_full_finetune_model.pth` - 最佳模型权重
- `../models/full_finetune_model/` - 完整模型保存目录
- `../checkpoints/fig/full_finetune_training_curves.png` - 训练曲线图

## 🎯 性能优化建议

1. **多卡训练**: 使用 `torchrun` 进行分布式训练
2. **混合精度**: 已启用FP16混合精度训练
3. **梯度累积**: 可调整 `GRADIENT_ACCUMULATION_STEPS`
4. **批次大小**: 根据GPU显存调整批次大小

## 📖 详细文档

- [全参数微调详细说明](README_full_finetune.md) - 包含完整的使用指南、配置选项和故障排除

## 🔄 对比结果

待所有方法实现完成后，将在此处展示详细的性能对比结果。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进代码。

## 📄 许可证

本项目仅供学习和研究使用。

---

## 💡 快速命令备忘

```bash
# 激活环境
conda activate llm

# 设置GPU
export CUDA_VISIBLE_DEVICES=1,2,3,5

# 单卡训练
python full_finetune.py

# 4卡训练
torchrun --nproc_per_node=4 full_finetune.py

# 查看GPU
nvidia-smi

# 查看训练日志
tail -f ../models/training.log
```