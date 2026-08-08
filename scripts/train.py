"""
QLoRA 微调训练脚本
用于微调 Qwen3 0.6B 模型，使其具备提供情绪价值的能力

支持:
- 4bit 量化 (QLoRA)
- LoRA 适配器
- 梯度检查点
- Flash Attention (可选)
"""
import os
import argparse
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
import bitsandbytes as bnb


def load_model_and_tokenizer(model_name: str, load_in_4bit: bool = True):
    """加载模型和 tokenizer，支持 4bit 量化"""
    print(f"加载模型: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    if load_in_4bit:
        # QLoRA: 4bit 量化配置
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    # 启用梯度检查点以节省显存
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    return model, tokenizer


def setup_lora(model, config=None):
    """配置 LoRA 适配器"""
    if config is None:
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    return model


def find_all_linear_names(model):
    """找到模型中所有的线性层名称"""
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def load_training_data(data_dir: str):
    """加载预处理后的训练数据"""
    train_data_path = os.path.join(data_dir, "train_dataset")
    eval_data_path = os.path.join(data_dir, "eval_dataset")

    train_dataset = load_from_disk(train_data_path)
    eval_dataset = load_from_disk(eval_data_path) if os.path.exists(eval_data_path) else None

    print(f"训练集大小: {len(train_dataset)}")
    if eval_dataset:
        print(f"验证集大小: {len(eval_dataset)}")

    return train_dataset, eval_dataset


def create_data_collator(tokenizer):
    """创建数据整理器"""
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )
    return data_collator


def train(args):
    """主训练函数"""
    print("=" * 60)
    print("QLoRA 微调训练")
    print("=" * 60)

    # 1. 加载模型和 tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit
    )

    # 2. 准备模型用于训练
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # 3. 配置 LoRA
    if args.auto_lora:
        # 自动选择目标模块
        target_modules = find_all_linear_names(model)
        print(f"自动检测到 LoRA 目标模块: {target_modules}")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
    else:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )

    model = setup_lora(model, lora_config)

    # 4. 加载训练数据
    train_dataset, eval_dataset = load_training_data(args.data_dir)

    # 5. 创建数据整理器
    data_collator = create_data_collator(tokenizer)

    # 6. 配置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.save_steps,
        evaluation_strategy="steps" if eval_dataset else "no",
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=False,
    )

    # 7. 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 8. 开始训练
    print("\n开始训练...")
    train_result = trainer.train()

    # 9. 保存模型
    print("\n保存模型...")
    final_output_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    # 10. 记录训练结果
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    if eval_dataset:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    print("\n" + "=" * 60)
    print(f"训练完成！模型已保存至: {final_output_dir}")
    print("=" * 60)

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="QLoRA 微调训练脚本")

    # 模型参数
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="模型名称或路径")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="是否使用 4bit 量化")

    # 数据参数
    parser.add_argument("--data_dir", type=str, default="./processed_data",
                        help="预处理后的数据目录")

    # LoRA 参数
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA 秩")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--auto_lora", action="store_true",
                        help="自动选择 LoRA 目标模块")

    # 训练参数
    parser.add_argument("--output_dir", type=str, default="./output/emotional_qwen",
                        help="输出目录")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="批大小")
    parser.add_argument("--gradient_accumulation", type=int, default=4,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="预热比例")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="日志记录步数")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="保存步数")

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()