# 微调文件夹

Qwen3 大模型在本项目里非常重要马，但 Qwen 模型作为基础模型，缺少电商场景下的垂直知识，现使用[亚马逊问答数据集](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/qa/)微调 Qwen 模型。

# 数据集说明
有针对 `Electronic` 的 `314,263` 问题，挑出 "question" 和 "answer"，作为指令微调的数据。

# 训练方式
采用指令微调的方式，根据问题生成答案，采用BLEU评估。
