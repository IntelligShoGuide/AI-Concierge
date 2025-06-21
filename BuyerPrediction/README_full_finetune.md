# 全参数微调 (Full Parameter Fine-tuning)

## 概述

本目录包含用于Kaggle买家意图预测比赛的全参数微调实现。全参数微调是指对预训练模型的所有参数进行更新，通常能获得更好的性能，但需要更多的计算资源和训练时间。

## 文件结构

```
BuyerPrediction/
├── full_finetune.py      # 主要的全参数微调脚本
├── config.py             # 配置文件
├── requirements.txt      # 依赖包列表
├── README_full_finetune.md  # 本说明文档
└── README.md             # 项目总览
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 准备数据

确保您的数据文件位于正确的位置，默认路径为 `data/train.csv`。数据文件应包含以下列：
- `text`: 文本内容
- `label`: 标签（将被自动编码）

### 2. 运行训练

```bash
python full_finetune.py
```

### 3. 自定义配置

您可以通过修改 `config.py` 文件来调整训练参数：

```python
# 修改模型配置
FullFinetuneConfig.MODEL_NAME = "roberta-base"
FullFinetuneConfig.BATCH_SIZE = 32
FullFinetuneConfig.LEARNING_RATE = 3e-5
```

## 主要特性

### 1. 完整的训练流程
- 数据加载和预处理
- 模型训练和验证
- 性能评估和可视化
- 模型保存和加载

### 2. 灵活的配置系统
- 支持多种预训练模型（BERT、RoBERTa、DistilBERT等）
- 可调整的超参数
- 设备自动检测（CPU/GPU）

### 3. 训练监控
- 实时损失和准确率显示
- 训练曲线可视化
- 混淆矩阵生成
- 最佳模型自动保存

### 4. 性能优化
- 混合精度训练（FP16）
- 梯度累积
- 早停机制
- 学习率调度

## 配置选项

### 模型配置
- `MODEL_NAME`: 预训练模型名称
- `NUM_LABELS`: 分类类别数
- `MAX_LENGTH`: 最大序列长度

### 训练配置
- `EPOCHS`: 训练轮数
- `BATCH_SIZE`: 批次大小
- `LEARNING_RATE`: 学习率
- `WEIGHT_DECAY`: 权重衰减

### 数据配置
- `DATA_PATH`: 数据文件路径
- `TRAIN_RATIO`: 训练集比例
- `VAL_RATIO`: 验证集比例

## 输出文件

训练完成后会生成以下文件：
- `best_full_finetune_model.pth`: 最佳模型权重
- `full_finetune_model/`: 完整模型保存目录
- `full_finetune_training_curves.png`: 训练曲线图
- `full_finetune_confusion_matrix.png`: 混淆矩阵图

## 支持的模型

通过 `ModelConfigs` 类可以轻松切换不同的预训练模型：

```python
# 使用BERT Base
config = ModelConfigs.get_config("bert_base")

# 使用RoBERTa Base
config = ModelConfigs.get_config("roberta_base")

# 使用DistilBERT
config = ModelConfigs.get_config("distilbert")
```

## 性能对比

全参数微调相比其他方法的特点：

| 方法 | 参数量 | 训练速度 | 内存占用 | 性能 |
|------|--------|----------|----------|------|
| 全参数微调 | 全部 | 慢 | 高 | 最好 |
| LoRA | 少量 | 快 | 低 | 好 |
| 提示学习 | 极少 | 最快 | 最低 | 一般 |

## 注意事项

1. **计算资源**: 全参数微调需要较大的GPU内存，建议使用至少16GB显存
2. **训练时间**: 相比LoRA等方法，训练时间更长
3. **过拟合风险**: 需要适当调整学习率和正则化参数
4. **数据质量**: 对数据质量要求较高，建议进行充分的数据清洗

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小批次大小
   - 使用梯度累积
   - 启用混合精度训练

2. **训练不收敛**
   - 调整学习率
   - 增加预热步数
   - 检查数据质量

3. **过拟合**
   - 增加权重衰减
   - 减少训练轮数
   - 使用早停机制

## 扩展功能

### 自定义数据集类

您可以继承 `BuyerIntentDataset` 类来支持自定义数据格式：

```python
class CustomDataset(BuyerIntentDataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        super().__init__(texts, labels, tokenizer, max_length)
    
    def __getitem__(self, idx):
        # 自定义数据处理逻辑
        pass
```

### 自定义评估指标

在 `evaluate` 方法中添加自定义评估指标：

```python
def evaluate(self, test_dataset):
    # 现有评估代码...
    
    # 添加自定义指标
    f1_score = f1_score(all_labels, all_predictions, average='weighted')
    print(f"F1 Score: {f1_score:.4f}")
```

## 联系信息

如有问题或建议，请参考项目主README文件或提交Issue。 