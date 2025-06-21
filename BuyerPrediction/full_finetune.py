#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
买家意图预测 - 全参数微调
Kaggle比赛: Buyer Intent Prediction Competition

本脚本实现全参数微调方法，用于预测用户的购买意图。
全参数微调是指对预训练模型的所有参数都进行更新，通常能获得最好的性能。
"""

# 导入操作系统相关的模块，用于文件路径操作
import os
# 导入pandas库，用于数据处理和分析
import pandas as pd
# 导入numpy库，用于数值计算
import numpy as np
# 导入PyTorch深度学习框架
import torch
# 导入PyTorch的神经网络模块
import torch.nn as nn
# 导入PyTorch的优化器模块
import torch.optim as optim
# 导入PyTorch的数据加载器，用于批量处理数据
from torch.utils.data import Dataset, DataLoader
# 导入sklearn的数据分割工具，用于划分训练集和验证集
from sklearn.model_selection import train_test_split
# 导入sklearn的标签编码器，用于将文本标签转换为数字
from sklearn.preprocessing import LabelEncoder, StandardScaler
# 导入sklearn的评估指标，用于计算模型性能
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# 导入matplotlib绘图库，用于可视化
import matplotlib.pyplot as plt
# 导入seaborn绘图库，用于更美观的可视化
import seaborn as sns
# 导入transformers库，用于加载预训练模型
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
# 导入配置
from config import FullFinetuneConfig
# 导入警告模块
import warnings
# 忽略警告信息，让输出更清洁
warnings.filterwarnings('ignore')
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import socket
# 导入日志模块
import logging
from datetime import datetime

def setup_logging():
    """
    设置日志配置
    创建日志目录并配置日志格式
    """
    # 创建日志目录
    log_dir = "../checkpoints/log"
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"full_finetune_{timestamp}.log")
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    # 创建logger
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件路径: {log_file}")
    
    return logger

def set_seed(seed=42):
    """
    设置随机种子以确保结果可重现
    参数:
        seed: 随机种子值，默认为42
    """
    # 设置PyTorch的CPU随机种子
    torch.manual_seed(seed)
    # 设置PyTorch的GPU随机种子（如果有多个GPU）
    torch.cuda.manual_seed_all(seed)
    # 设置numpy的随机种子
    np.random.seed(seed)
    # 设置PyTorch的确定性模式，确保结果可重现
    torch.backends.cudnn.deterministic = True

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def setup_ddp():
    # 只在多卡模式下初始化DDP
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            logger = logging.getLogger(__name__)
            logger.info(f"DDP初始化完成: RANK={os.environ.get('RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}")
    else:
        logger = logging.getLogger(__name__)
        logger.info("单卡模式，跳过DDP初始化")

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()
        logger = logging.getLogger(__name__)
        logger.info("DDP清理完成")

class BuyerIntentDataset(Dataset):
    """
    买家意图数据集类
    继承自PyTorch的Dataset类，用于处理文本分类数据
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=None):
        """
        初始化数据集
        参数:
            texts: 文本列表
            labels: 标签列表
            tokenizer: 分词器，用于将文本转换为模型可理解的数字序列
            max_length: 最大序列长度，如果为None则使用配置文件中的设置
        """
        # 保存文本数据
        self.texts = texts
        # 保存标签数据
        self.labels = labels
        # 保存分词器
        self.tokenizer = tokenizer
        # 保存最大序列长度，如果未指定则使用配置文件中的设置
        self.max_length = max_length if max_length is not None else FullFinetuneConfig.MAX_LENGTH
    
    def __len__(self):
        """
        返回数据集的大小
        """
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        获取指定索引的数据项
        参数:
            idx: 数据索引
        返回:
            包含input_ids、attention_mask和labels的字典
        """
        # 获取指定索引的文本，并转换为字符串
        text = str(self.texts[idx])
        # 获取指定索引的标签
        label = self.labels[idx]
        
        # 使用分词器处理文本
        encoding = self.tokenizer(
            text,                           # 要处理的文本
            truncation=True,                # 如果文本超过最大长度则截断
            padding='max_length',           # 如果文本不足最大长度则填充
            max_length=self.max_length,     # 最大序列长度
            return_tensors='pt'             # 返回PyTorch张量格式
        )
        
        # 返回处理后的数据
        return {
            'input_ids': encoding['input_ids'].flatten(),        # 输入ID序列，展平为一维
            'attention_mask': encoding['attention_mask'].flatten(),  # 注意力掩码，展平为一维
            'labels': torch.tensor(label, dtype=torch.long)      # 标签，转换为长整型张量
        }

class FullFinetuneModel:
    """
    全参数微调模型类
    负责模型的初始化、训练、评估等所有操作
    """
    
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        """
        初始化模型
        参数:
            model_name: 预训练模型名称，默认为BERT基础模型
            num_labels: 分类类别数，默认为2（二分类）
        """
        self.model_name = model_name
        self.num_labels = num_labels
        # DDP: 设置当前进程的GPU
        if dist.is_initialized():
            local_rank = int(os.environ["LOCAL_RANK"])
            self.device = torch.device(f"cuda:{local_rank}")
            logger = logging.getLogger(__name__)
            logger.info(f"DDP模式: 进程 {local_rank} 使用设备 cuda:{local_rank}")
        else:
            # 非DDP模式，使用第一个可见的GPU
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                logger = logging.getLogger(__name__)
                logger.info(f"单卡模式: 使用设备 cuda:0")
            else:
                self.device = torch.device("cpu")
                logger = logging.getLogger(__name__)
                logger.info("使用CPU模式")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            torch_dtype=torch.bfloat16
        )
        # 修复Qwen模型的pad_token问题
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # 确保模型配置也设置了pad_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.to(self.device)
        # DDP: 包裹模型
        if dist.is_initialized():
            self.model = DDP(self.model, device_ids=[self.device.index], output_device=self.device.index, find_unused_parameters=True)
        logger = logging.getLogger(__name__)
        logger.info(f"模型已加载到设备: {self.device}")
        logger.info(f"模型参数设备: {next(self.model.parameters()).device}")
        logger.info(f"可见GPU数量: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            logger.info(f"当前GPU显存: {torch.cuda.get_device_properties(self.device).total_memory / 1024**3:.1f} GB")
        
    def load_data(self, data_path):
        """
        加载数据文件
        参数:
            data_path: 数据文件路径
        返回:
            加载的数据框，如果失败则返回None
        """
        logger = logging.getLogger(__name__)
        logger.info(f"正在加载数据: {data_path}")
        
        # 尝试加载CSV文件
        try:
            # 使用pandas读取CSV文件
            df = pd.read_csv(data_path)
            logger.info(f"数据加载成功，共 {len(df)} 条记录")
            logger.info(f"数据列: {df.columns.tolist()}")
            return df
        except Exception as e:
            # 如果加载失败，打印错误信息
            logger.error(f"数据加载失败: {e}")
            return None
    
    def preprocess_data(self, df):
        """
        数据预处理
        参数:
            df: 原始数据框
        返回:
            处理后的训练集和验证集数据框
        """
        logger = logging.getLogger(__name__)
        logger.info("开始数据预处理...")
        
        # 重命名列名以匹配代码期望的格式
        if 'Query' in df.columns:
            df = df.rename(columns={'Query': 'text'})
        if 'Intent' in df.columns:
            df = df.rename(columns={'Intent': 'label'})
        
        # 删除文本列中的缺失值
        df = df.dropna(subset=['text'])  # 假设文本列名为'text'
        
        # 如果存在标签列，则进行标签编码
        if 'label' in df.columns:
            # 创建标签编码器
            label_encoder = LabelEncoder()
            # 将文本标签转换为数字标签
            df['label_encoded'] = label_encoder.fit_transform(df['label'])
            # 保存标签编码器，用于后续的标签转换
            self.label_encoder = label_encoder
            logger.info(f"标签编码完成，类别: {label_encoder.classes_}")
            logger.info(f"标签映射: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
        
        # 将数据集分割为训练集和验证集
        train_df, val_df = train_test_split(
            df,                    # 要分割的数据
            test_size=0.2,         # 验证集占20%
            random_state=42,       # 随机种子，确保结果可重现
            stratify=df['label_encoded']  # 按标签分层采样，确保各类别比例一致
        )
        
        logger.info(f"训练集大小: {len(train_df)}")
        logger.info(f"验证集大小: {len(val_df)}")
        
        return train_df, val_df
    
    def create_datasets(self, train_df, val_df):
        """
        创建PyTorch数据集
        参数:
            train_df: 训练数据框
            val_df: 验证数据框
        返回:
            训练数据集和验证数据集
        """
        logger = logging.getLogger(__name__)
        logger.info("创建数据集...")
        
        # 创建训练数据集
        train_dataset = BuyerIntentDataset(
            train_df['text'].values,        # 训练文本
            train_df['label_encoded'].values,  # 训练标签
            self.tokenizer                  # 分词器
        )
        
        # 创建验证数据集
        val_dataset = BuyerIntentDataset(
            val_df['text'].values,          # 验证文本
            val_df['label_encoded'].values,    # 验证标签
            self.tokenizer                  # 分词器
        )
        
        return train_dataset, val_dataset
    
    def train(self, train_dataset, val_dataset):
        """
        训练模型
        参数:
            train_dataset: 训练数据集
            val_dataset: 验证数据集
        返回:
            最佳验证准确率
        """
        logger = logging.getLogger(__name__)
        logger.info("开始全参数微调训练...")
        logger.info(f"序列长度: {FullFinetuneConfig.MAX_LENGTH}")
        logger.info(f"混合精度训练: {FullFinetuneConfig.USE_AMP}")
        logger.info(f"梯度累积步数: {FullFinetuneConfig.GRADIENT_ACCUMULATION_STEPS}")
        # DDP: 使用DistributedSampler
        train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
        val_sampler = DistributedSampler(val_dataset) if dist.is_initialized() else None
        train_loader = DataLoader(train_dataset, batch_size=FullFinetuneConfig.BATCH_SIZE, shuffle=(train_sampler is None), sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_size=FullFinetuneConfig.BATCH_SIZE, shuffle=False, sampler=val_sampler)
        optimizer = optim.AdamW(self.model.parameters(), lr=FullFinetuneConfig.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        best_val_acc = 0.0
        train_losses = []
        val_losses = []
        val_accuracies = []
        for epoch in range(FullFinetuneConfig.EPOCHS):
            if dist.is_initialized():
                train_loader.sampler.set_epoch(epoch)
            if is_main_process():
                logger.info(f"\nEpoch {epoch+1}/{FullFinetuneConfig.EPOCHS}")
            self.model.train()
            train_loss = 0.0
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                if batch_idx == 0 and is_main_process():
                    logger.info(f"输入数据设备: {input_ids.device}")
                    logger.info(f"模型设备: {next(self.model.parameters()).device}")
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                if batch_idx % 100 == 0 and is_main_process():
                    logger.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            if is_main_process():
                logger.info(f"训练损失: {avg_train_loss:.4f}")
                logger.info(f"验证损失: {avg_val_loss:.4f}")
                logger.info(f"验证准确率: {val_accuracy:.4f}")
                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    torch.save(self.model.state_dict(), '../models/best_full_finetune_model.pth')
                    logger.info("保存最佳模型")
        if is_main_process():
            self.plot_training_curves(train_losses, val_losses, val_accuracies)
        return best_val_acc
    
    def plot_training_curves(self, train_losses, val_losses, val_accuracies):
        """
        绘制训练曲线
        参数:
            train_losses: 训练损失列表
            val_losses: 验证损失列表
            val_accuracies: 验证准确率列表
        """
        # 创建包含两个子图的图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 绘制损失曲线
        ax1.plot(train_losses, label='训练损失')
        ax1.plot(val_losses, label='验证损失')
        ax1.set_title('训练和验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # 绘制准确率曲线
        ax2.plot(val_accuracies, label='验证准确率')
        ax2.set_title('验证准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        # 调整子图布局
        plt.tight_layout()
        # 保存图片
        plt.savefig('../checkpoints/fig/full_finetune_training_curves.png', dpi=300, bbox_inches='tight')
        # 显示图片
        # plt.show()
    
    def evaluate(self, test_dataset):
        """
        评估模型性能
        参数:
            test_dataset: 测试数据集
        返回:
            测试准确率
        """
        logger = logging.getLogger(__name__)
        logger.info("开始模型评估...")
        
        # 创建测试数据加载器
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        self.model.eval()  # 设置为评估模式
        
        # 初始化预测和真实标签列表
        all_predictions = []
        all_labels = []
        
        # 在测试时不计算梯度
        with torch.no_grad():
            # 遍历测试数据批次
            for batch in test_loader:
                # 将数据移动到指定设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 前向传播
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # 获取预测结果
                _, predicted = torch.max(outputs.logits, 1)
                
                # 收集预测结果和真实标签
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算准确率
        accuracy = accuracy_score(all_labels, all_predictions)
        logger.info(f"测试准确率: {accuracy:.4f}")
        
        # 打印详细的分类报告
        logger.info("\n分类报告:")
        logger.info(classification_report(all_labels, all_predictions))
        
        # 绘制混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.savefig('full_finetune_confusion_matrix.png', dpi=300, bbox_inches='tight')
        # plt.show()
        
        return accuracy
    
    def predict(self, texts):
        """
        对新数据进行预测
        参数:
            texts: 要预测的文本列表
        返回:
            预测结果列表
        """
        self.model.eval()  # 设置为评估模式
        predictions = []
        
        # 遍历每个文本
        for text in texts:
            # 使用分词器处理文本
            encoding = self.tokenizer(
                text,                       # 要处理的文本
                truncation=True,            # 截断过长的文本
                padding='max_length',       # 填充过短的文本
                max_length=FullFinetuneConfig.MAX_LENGTH,  # 使用配置文件中的最大长度
                return_tensors='pt'         # 返回PyTorch张量
            )
            
            # 将数据移动到指定设备
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # 进行预测
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                _, predicted = torch.max(outputs.logits, 1)
                predictions.append(predicted.cpu().item())
        
        return predictions
    
    def save_model(self, model_path):
        """
        保存模型
        参数:
            model_path: 模型保存路径
        """
        # 保存模型权重
        torch.save(self.model.state_dict(), model_path)
        # 保存分词器
        self.tokenizer.save_pretrained(model_path)
        logger = logging.getLogger(__name__)
        logger.info(f"模型已保存到: {model_path}")
    
    def load_model(self, model_path):
        """
        加载模型
        参数:
            model_path: 模型加载路径
        """
        # 加载模型权重
        self.model.load_state_dict(torch.load(model_path))
        logger = logging.getLogger(__name__)
        logger.info(f"模型已从 {model_path} 加载")

def main():
    """
    主函数，程序的入口点
    """
    # 设置日志
    logger = setup_logging()
    logger.info("=== 买家意图预测 - 全参数微调 ===")
    
    # 设置随机种子，确保结果可重现
    set_seed(FullFinetuneConfig.RANDOM_SEED)
    
    # 打印当前配置信息
    logger.info("\n=== 当前配置 ===")
    logger.info(f"模型路径: {FullFinetuneConfig.MODEL_NAME}")
    logger.info(f"分类类别数: {FullFinetuneConfig.NUM_LABELS}")
    logger.info(f"最大序列长度: {FullFinetuneConfig.MAX_LENGTH}")
    logger.info(f"训练轮数: {FullFinetuneConfig.EPOCHS}")
    logger.info(f"批次大小: {FullFinetuneConfig.BATCH_SIZE}")
    logger.info(f"学习率: {FullFinetuneConfig.LEARNING_RATE}")
    logger.info(f"混合精度训练: {FullFinetuneConfig.USE_AMP}")
    logger.info(f"梯度累积步数: {FullFinetuneConfig.GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"数据路径: {FullFinetuneConfig.DATA_PATH}")
    logger.info("=" * 50)
    
    # 初始化模型，使用配置文件中的模型名称进行二分类
    model = FullFinetuneModel(
        model_name=FullFinetuneConfig.MODEL_NAME, 
        num_labels=FullFinetuneConfig.NUM_LABELS
    )
    
    # 数据文件路径（需要根据实际情况调整）
    data_path = FullFinetuneConfig.DATA_PATH  # 请根据实际数据路径修改 
    
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        logger.error(f"数据文件不存在: {data_path}")
        logger.error("请确保数据文件路径正确，或者修改 data_path 变量")
        return
    
    # 加载数据
    df = model.load_data(data_path)
    if df is None:
        return
    
    # 数据预处理
    train_df, val_df = model.preprocess_data(df)
    
    # 创建数据集
    train_dataset, val_dataset = model.create_datasets(train_df, val_df)
    
    # 训练模型
    best_accuracy = model.train(
        train_dataset,      # 训练数据集
        val_dataset         # 验证数据集
        # 其他参数使用配置文件中的默认值
    )
    
    # 打印最佳验证准确率
    logger.info(f"\n最佳验证准确率: {best_accuracy:.4f}")
    
    # 保存模型
    model.save_model(FullFinetuneConfig.MODEL_SAVE_PATH)
    
    logger.info("\n全参数微调完成！")

# 如果直接运行此脚本，则执行main函数
if __name__ == "__main__":
    setup_ddp()
    try:
        main()
    finally:
        cleanup_ddp() 