#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件 - 全参数微调
包含训练参数、模型配置等
这个文件集中管理所有的超参数和配置选项，方便调整和实验
"""

class FullFinetuneConfig:
    """
    全参数微调配置类
    包含所有训练和模型相关的配置参数
    """
    
    # ==================== 模型配置 ====================
    MODEL_NAME = "/home/users/wzr/project/predict/LLM/Qwen/qwen/Qwen2_5-0_5B-Instruct/"  # 本地Qwen模型路径
    NUM_LABELS = 7  # 分类类别数，7表示七分类（不同的购买意图类别）
    MAX_LENGTH = 256  # 最大序列长度，超过此长度的文本会被截断（从512减小到256以减少显存使用）
    
    # ==================== 训练配置 ====================
    EPOCHS = 30  # 训练轮数，即整个数据集被训练的次数
    BATCH_SIZE = 8  # 批次大小，每次处理多少个样本（Qwen模型较大，减小批次大小）
    LEARNING_RATE = 1e-5  # 学习率，控制参数更新的步长（Qwen模型使用较小的学习率）
    WEIGHT_DECAY = 0.01  # 权重衰减，用于防止过拟合的正则化参数
    WARMUP_STEPS = 500  # 预热步数，学习率从0逐渐增加到设定值
    
    # ==================== 数据配置 ====================
    TRAIN_RATIO = 0.8  # 训练集比例，80%的数据用于训练
    VAL_RATIO = 0.2  # 验证集比例，20%的数据用于验证
    RANDOM_SEED = 42  # 随机种子，确保结果可重现
    
    # ==================== 文件路径配置 ====================
    DATA_PATH = "../data/buyer_intent_dataset_kaggle_sample_submission.csv"  # 训练数据文件路径
    MODEL_SAVE_PATH = "../models/full_finetune_model"  # 模型保存路径
    BEST_MODEL_PATH = "../models/best_full_finetune_model.pth"  # 最佳模型权重保存路径
    
    # ==================== 训练监控配置 ====================
    LOGGING_STEPS = 100  # 每多少步记录一次日志
    EVAL_STEPS = 500  # 每多少步进行一次评估
    SAVE_STEPS = 1000  # 每多少步保存一次模型
    
    # ==================== 早停配置 ====================
    EARLY_STOPPING_PATIENCE = 3  # 早停耐心值，连续多少轮验证集性能不提升就停止训练
    EARLY_STOPPING_THRESHOLD = 0.001  # 早停阈值，性能提升小于此值认为没有提升
    
    # ==================== 设备配置 ====================
    DEVICE = "auto"  # 设备选择: "auto"自动检测, "cpu"强制使用CPU, "cuda"强制使用GPU
    
    # ==================== 性能优化配置 ====================
    USE_AMP = True  # 是否使用混合精度训练，可以节省显存并加速训练
    GRADIENT_ACCUMULATION_STEPS = 2  # 梯度累积步数，用于模拟更大的批次大小
    
    # ==================== 学习率调度器配置 ====================
    LR_SCHEDULER_TYPE = "linear"  # 学习率调度器类型：linear, cosine, polynomial等
    NUM_TRAIN_EPOCHS = 3  # 训练轮数（与EPOCHS相同，用于兼容transformers库）
    
    # ==================== 数据增强配置 ====================
    USE_DATA_AUGMENTATION = False  # 是否使用数据增强，如随机删除、替换等
    
    # ==================== 模型检查点配置 ====================
    SAVE_TOTAL_LIMIT = 3  # 保存的检查点数量限制，超过此数量会删除旧的检查点
    
    @classmethod
    def get_training_args(cls):
        """
        获取训练参数字典
        返回transformers库TrainingArguments所需的参数字典
        """
        return {
            "output_dir": cls.MODEL_SAVE_PATH,  # 输出目录
            "num_train_epochs": cls.NUM_TRAIN_EPOCHS,  # 训练轮数
            "per_device_train_batch_size": cls.BATCH_SIZE,  # 每个设备的训练批次大小
            "per_device_eval_batch_size": cls.BATCH_SIZE,  # 每个设备的评估批次大小
            "learning_rate": cls.LEARNING_RATE,  # 学习率
            "weight_decay": cls.WEIGHT_DECAY,  # 权重衰减
            "warmup_steps": cls.WARMUP_STEPS,  # 预热步数
            "logging_steps": cls.LOGGING_STEPS,  # 日志记录步数
            "eval_steps": cls.EVAL_STEPS,  # 评估步数
            "save_steps": cls.SAVE_STEPS,  # 保存步数
            "save_total_limit": cls.SAVE_TOTAL_LIMIT,  # 保存检查点数量限制
            "gradient_accumulation_steps": cls.GRADIENT_ACCUMULATION_STEPS,  # 梯度累积步数
            "lr_scheduler_type": cls.LR_SCHEDULER_TYPE,  # 学习率调度器类型
            "load_best_model_at_end": True,  # 训练结束时加载最佳模型
            "metric_for_best_model": "accuracy",  # 用于选择最佳模型的指标
            "greater_is_better": True,  # 指标是否越大越好
            "fp16": cls.USE_AMP,  # 是否使用混合精度训练
            "dataloader_pin_memory": False,  # 是否将数据加载到固定内存
            "remove_unused_columns": False,  # 是否移除未使用的列
        }
    
    @classmethod
    def get_model_config(cls):
        """
        获取模型配置字典
        返回模型相关的配置参数
        """
        return {
            "model_name": cls.MODEL_NAME,  # 模型名称
            "num_labels": cls.NUM_LABELS,  # 分类类别数
            "max_length": cls.MAX_LENGTH,  # 最大序列长度
        }
    
    @classmethod
    def get_data_config(cls):
        """
        获取数据配置字典
        返回数据相关的配置参数
        """
        return {
            "data_path": cls.DATA_PATH,  # 数据文件路径
            "train_ratio": cls.TRAIN_RATIO,  # 训练集比例
            "val_ratio": cls.VAL_RATIO,  # 验证集比例
            "random_seed": cls.RANDOM_SEED,  # 随机种子
        }

# ==================== 不同模型配置 ====================
class ModelConfigs:
    """
    不同模型的配置类
    提供不同预训练模型的推荐配置参数
    """
    
    # BERT基础模型的配置
    BERT_BASE = {
        "model_name": "bert-base-uncased",  # 模型名称
        "max_length": 512,  # 最大序列长度
        "learning_rate": 2e-5,  # 推荐学习率
        "batch_size": 16,  # 推荐批次大小
    }
    
    # BERT大型模型的配置
    BERT_LARGE = {
        "model_name": "bert-large-uncased",  # 模型名称
        "max_length": 512,  # 最大序列长度
        "learning_rate": 1e-5,  # 推荐学习率（比基础模型小，因为参数量更大）
        "batch_size": 8,  # 推荐批次大小（比基础模型小，因为显存占用更大）
    }
    
    # RoBERTa基础模型的配置
    ROBERTA_BASE = {
        "model_name": "roberta-base",  # 模型名称
        "max_length": 512,  # 最大序列长度
        "learning_rate": 2e-5,  # 推荐学习率
        "batch_size": 16,  # 推荐批次大小
    }
    
    # DistilBERT模型的配置（轻量级BERT）
    DISTILBERT = {
        "model_name": "distilbert-base-uncased",  # 模型名称
        "max_length": 512,  # 最大序列长度
        "learning_rate": 3e-5,  # 推荐学习率（比BERT稍大，因为模型更小）
        "batch_size": 32,  # 推荐批次大小（比BERT大，因为模型更小）
    }
    
    # Qwen模型的配置
    QWEN_LOCAL = {
        "model_name": "/home/users/wzr/project/predict/LLM/Qwen/qwen/Qwen2_5-0_5B-Instruct/",  # 本地Qwen模型路径
        "max_length": 256,  # 最大序列长度（从512减小到256以减少显存使用）
        "learning_rate": 1e-5,  # 推荐学习率
        "batch_size": 8,  # 推荐批次大小
    }
    
    @classmethod
    def get_config(cls, model_type="qwen_local"):
        """
        获取指定模型的配置
        参数:
            model_type: 模型类型，可选值：bert_base, bert_large, roberta_base, distilbert, qwen_local
        返回:
            对应模型的配置字典
        """
        # 定义模型类型到配置的映射
        configs = {
            "bert_base": cls.BERT_BASE,      # BERT基础模型
            "bert_large": cls.BERT_LARGE,    # BERT大型模型
            "roberta_base": cls.ROBERTA_BASE,  # RoBERTa基础模型
            "distilbert": cls.DISTILBERT,    # DistilBERT模型
            "qwen_local": cls.QWEN_LOCAL,    # 本地Qwen模型
        }
        # 返回指定模型的配置，如果不存在则返回Qwen本地模型配置
        return configs.get(model_type, cls.QWEN_LOCAL) 