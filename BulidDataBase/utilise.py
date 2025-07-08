import json

# 读取JSONL文件
def load_jsonl(file_path):
    """读取JSONL格式的文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data