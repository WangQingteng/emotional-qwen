"""
数据预处理脚本
将 SFT 对话数据转换为模型训练所需的 Tokenize 格式
支持 qwen3 的 Chat Template
"""
import os
import json
import argparse
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer


def load_sft_data(data_path: str) -> Dataset:
    """加载 SFT 格式的数据集"""
    if data_path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=data_path)
    elif data_path.endswith(".json"):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        dataset = Dataset.from_list(data)
    else:
        dataset = load_dataset(data_path)
    return dataset


def process_conversations(example, tokenizer, max_length: int = 2048):
    """处理单条对话，使用 chat template 进行 tokenize"""
    conversations = example.get("conversations", [])

    messages = []
    for msg in conversations:
        role = "user" if msg["from"] == "human" else "assistant"
        messages.append({
            "role": role,
            "content": msg["value"]
        })

    if not messages:
        return {"input_ids": [], "labels": [], "attention_mask": []}

    # 使用 tokenizer 的 chat template 处理
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    tokenized = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    # 构建 labels (仅计算 assistant 的 loss)
    input_ids = tokenized["input_ids"][0]
    attention_mask = tokenized["attention_mask"][0]

    # 找到 assistant 回复的起始位置，之前的部分设为 -100
    labels = input_ids.clone()
    # 简化处理：将最后一条 assistant 回复之前的 token 都设为 -100
    # 更精确的实现见下方
    assistant_start = find_assistant_start(input_ids, tokenizer)
    if assistant_start > 0:
        labels[:assistant_start] = -100

    return {
        "input_ids": input_ids.tolist(),
        "labels": labels.tolist(),
        "attention_mask": attention_mask.tolist(),
    }


def find_assistant_start(input_ids, tokenizer):
    """找到 assistant 回复在 input_ids 中的起始位置"""
    # qwen3 的 assistant 回复通常在 <|im_start|>assistant\n 之后
    # 找到最后一个 user 消息后的 assistant 起始位置
    text = tokenizer.decode(input_ids, skip_special_tokens=False)

    # 找 assistant 标记
    assistant_marker = "<|im_start|>assistant\n"
    last_pos = text.rfind(assistant_marker)

    if last_pos == -1:
        # 尝试其他可能的格式
        for marker in ["assistant\n", "assistant: ", "<|assistant|>"]:
            last_pos = text.rfind(marker)
            if last_pos != -1:
                # 找到对应的 token 位置
                tokens_before = tokenizer.encode(text[:last_pos + len(marker)], add_special_tokens=False)
                return len(tokens_before)
        return 0

    # 找到标记后的 token 位置
    tokens_before = tokenizer.encode(text[:last_pos + len(assistant_marker)], add_special_tokens=False)
    return len(tokens_before)


def preprocess_dataset(
    data_path: str,
    model_name: str,
    output_dir: str,
    max_length: int = 2048,
    split_ratio: float = 0.95
):
    """完整的数据预处理流程"""
    print(f"加载数据集: {data_path}")
    dataset = load_sft_data(data_path)

    if isinstance(dataset, dict):
        # 如果是 DatasetDict，取第一个 split
        split_name = list(dataset.keys())[0]
        dataset = dataset[split_name]

    print(f"数据集大小: {len(dataset)} 条")

    print(f"加载 Tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 处理数据集
    print("开始处理数据...")

    def process_batch(batch):
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        for i in range(len(batch["conversations"])):
            conv = batch["conversations"][i]
            example = {"conversations": conv}
            result = process_conversations(example, tokenizer, max_length)
            input_ids_list.append(result["input_ids"])
            labels_list.append(result["labels"])
            attention_mask_list.append(result["attention_mask"])

        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
            "attention_mask": attention_mask_list,
        }

    processed = dataset.map(
        process_batch,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Processing"
    )

    # 过滤空样本
    def filter_empty(example):
        return len(example["input_ids"]) > 0

    processed = processed.filter(filter_empty)
    print(f"处理后样本数: {len(processed)}")

    # 划分训练集和验证集
    if "train" in dataset and "validation" in dataset:
        train_dataset = processed
        eval_dataset = None
    else:
        split = processed.train_test_split(test_size=1 - split_ratio, shuffle=True, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]

    print(f"训练集大小: {len(train_dataset)}")
    if eval_dataset:
        print(f"验证集大小: {len(eval_dataset)}")

    # 保存处理后的数据
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train_dataset")
    train_dataset.save_to_disk(train_path)
    print(f"训练集保存至: {train_path}")

    if eval_dataset:
        eval_path = os.path.join(output_dir, "eval_dataset")
        eval_dataset.save_to_disk(eval_path)
        print(f"验证集保存至: {eval_path}")

    # 保存 tokenizer 配置
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    print(f"Tokenizer 配置已保存")

    return train_dataset, eval_dataset


def main():
    parser = argparse.ArgumentParser(description="情绪价值数据预处理")
    parser.add_argument("--data_path", type=str, default="./data/combined_emotional_sft.jsonl",
                        help="SFT 数据路径")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="模型名称（用于加载 tokenizer）")
    parser.add_argument("--output_dir", type=str, default="./processed_data",
                        help="预处理后数据输出目录")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="最大序列长度")
    parser.add_argument("--split_ratio", type=float, default=0.95,
                        help="训练集划分比例")

    args = parser.parse_args()

    print("=" * 60)
    print("数据预处理脚本")
    print("=" * 60)

    preprocess_dataset(
        data_path=args.data_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_length=args.max_length,
        split_ratio=args.split_ratio
    )

    print("\n" + "=" * 60)
    print("数据预处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()