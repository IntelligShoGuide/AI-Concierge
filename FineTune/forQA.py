import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
import gzip
import pandas as pd
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch.nn.functional as F

os.chdir('./FineTune')

def data_to_instruction(df):
    # 将df转化为指令微调的数据集格式
    # 假设df中有'question'和'answer'两列
    instruction_data = []
    for idx, row in df.iterrows():
        if pd.isna(row.get('question')) or pd.isna(row.get('answer')):
            continue
        instruction_data.append({
            "instruction": "You are an electronics sales expert and you need to answer the following question",
            "input": str(row['question']),
            "output": str(row['answer'])
        })
    # 直接返回指令微调格式的数据
    return instruction_data

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def data_process(data_path):
    i = 0
    df = {}
    for d in parse(data_path):
        df[i] = d
        i += 1
    
    df = pd.DataFrame.from_dict(df, orient='index')
    df = df[['question', 'answer']]
    return data_to_instruction(df)


# 构建训练数据集
class InstructionDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length    # 问答句子普遍不长，最大长度设置为256，节省显存

    def __len__(self):
        return len(self.data)

    # __getitem__ 方法会在训练时被调用。PyTorch 的 DataLoader 在每次取一个 batch 的数据时，会调用数据集（Dataset）对象的 __getitem__ 方法来获取指定索引的数据样本。因此，__getitem__ 主要是在训练过程中（包括验证和测试时）被频繁调用，而不是在数据集构造时调用。
    def __getitem__(self, idx):
        item = self.data[idx]
        conversation_text = f"user：{item['input']}\nassistant: "
        prompt = f"instruction：{item['instruction']}\n{conversation_text}"
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        labels = self.tokenizer(
            item['output'],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels['input_ids'].squeeze(0)
        }

data_path = "../data/AmazonReviews/qa_Electronics.json.gz"
instruction_data = data_process(data_path)

# 加载Qwen模型和分词器
qwen_model_path = "/home/users/wzr/project/predict/LLM/Qwen/qwen/Qwen2_5-0_5B-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(qwen_model_path)
model = AutoModelForCausalLM.from_pretrained(
    qwen_model_path, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# 构建数据集
train_dataset = InstructionDataset(instruction_data, tokenizer)

del instruction_data

# 配置LoRA参数
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # 具体模块名称需根据Qwen模型结构调整
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 数据整理器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 准备模型以支持LoRA微调
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

print("LoRA微调模型已准备好。")

# 设置训练参数
training_args = TrainingArguments(
    output_dir="../models/qwen-finetuned-qa",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    learning_rate=2e-5,
    fp16=True,
    remove_unused_columns=False,
    report_to=[],
)

# Trainer会自动从train_dataset的__getitem__返回的dict中查找"labels"键作为标签（label），
# 这里labels在InstructionDataset的__getitem__方法中已返回，无需在Trainer中单独指定label参数。
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
)

print("开始问答微调训练...")
trainer.train()
print("训练完成，模型已保存至:", training_args.output_dir)

# 使用BLEU评估模型
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def evaluate_bleu(model, tokenizer, dataset, num_samples=100):
    bleu_scores = []
    chencherry = SmoothingFunction()
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        input_ids = sample["input_ids"].unsqueeze(0).to(model.device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(model.device)
        # 生成模型输出
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                do_sample=False
            )
        # 解码生成的输出
        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # 获取参考答案
        item = train_dataset.data[i]
        reference = item['output']
        # 只对输出部分做BLEU评估
        # 假设输出格式为：指令：xxx\n输入：xxx\n输出：yyy
        # 取生成文本中最后一个“输出：”后的内容
        if "输出：" in output_text:
            output_text = output_text.split("输出：")[-1].strip()
        # 分词
        reference_tokens = list(reference)
        output_tokens = list(output_text)
        bleu = sentence_bleu(
            [reference_tokens],
            output_tokens,
            smoothing_function=chencherry.method1
        )
        bleu_scores.append(bleu)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"在{min(num_samples, len(dataset))}个样本上的平均BLEU分数为: {avg_bleu:.4f}")

print("开始使用BLEU评估模型...")
evaluate_bleu(model, tokenizer, train_dataset, num_samples=100)

