"""
推理测试脚本
用于测试微调后模型的情绪价值提供能力

支持:
- 交互式对话
- 批量测试用例
- 对比基线模型和微调模型
"""
import os
import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# 测试用例：情绪价值场景
TEST_CASES = [
    {
        "category": "安慰",
        "input": "我今天心情特别差，什么都不想做",
        "expected_focus": "共情、安抚情绪"
    },
    {
        "category": "自我否定",
        "input": "我觉得自己是个废物，一事无成",
        "expected_focus": "重建自信、肯定价值"
    },
    {
        "category": "焦虑",
        "input": "我总是担心未来会出问题，睡不着觉",
        "expected_focus": "缓解焦虑、提供安全感"
    },
    {
        "category": "失恋",
        "input": "男朋友和我分手了，我真的很爱他，活不下去了",
        "expected_focus": "共情、情感支持、引导积极"
    },
    {
        "category": "孤独",
        "input": "我觉得没有人真正理解我，身边没有知心朋友",
        "expected_focus": "共情、陪伴感、鼓励社交"
    },
    {
        "category": "工作压力",
        "input": "工作压力太大了，领导骂我，同事排挤我，我想辞职",
        "expected_focus": "共情、压力疏导、理性分析"
    },
    {
        "category": "家庭矛盾",
        "input": "我和父母经常吵架，他们完全不理解我的生活方式",
        "expected_focus": "共情、换位思考、沟通建议"
    },
    {
        "category": "低自尊",
        "input": "我长得不好看，身材也差，没有男生会喜欢我",
        "expected_focus": "肯定自我价值、外貌焦虑疏导"
    },
]


def load_model(model_path: str, base_model_name: str = None, load_in_4bit: bool = False):
    """加载模型，支持 LoRA 适配器合并"""
    import os

    # 检查是否是 LoRA 适配器目录
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    is_lora = os.path.exists(adapter_config_path)

    if is_lora and base_model_name:
        print(f"检测到 LoRA 适配器，加载基础模型: {base_model_name}")

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 加载 LoRA 适配器
        model = PeftModel.from_pretrained(model, model_path)

        # 可选：合并到基础模型
        # model = model.merge_and_unload()
    else:
        print(f"加载完整模型: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, user_input: str, max_new_tokens: int = 512,
                      temperature: float = 0.7, top_p: float = 0.9):
    """生成回复"""
    messages = [
        {"role": "user", "content": user_input}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
        )

    # 解码并提取回复
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response.strip()


def run_interactive(model, tokenizer):
    """交互式对话模式"""
    print("\n" + "=" * 60)
    print("情绪价值助手 - 交互模式")
    print("=" * 60)
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'clear' 清除上下文")
    print("=" * 60)

    conversation_history = []

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "退出"]:
            print("再见！希望你今天心情愉快 ❤️")
            break

        if user_input.lower() in ["clear", "清除"]:
            conversation_history = []
            print("对话历史已清除")
            continue

        # 构建消息
        messages = conversation_history + [
            {"role": "user", "content": user_input}
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
            )

        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        print(f"\n助手: {response.strip()}")

        # 更新对话历史
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response.strip()})

        # 保持最近 10 轮对话
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]


def run_batch_test(model, tokenizer, output_file: str = None):
    """批量测试用例"""
    print("\n" + "=" * 60)
    print("情绪价值测试 - 批量用例")
    print("=" * 60)

    results = []

    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n--- 测试用例 {i}/{len(TEST_CASES)} ---")
        print(f"分类: {case['category']}")
        print(f"输入: {case['input']}")
        print(f"期望关注: {case['expected_focus']}")

        response = generate_response(model, tokenizer, case["input"])

        print(f"输出: {response}")
        print(f"{'='*40}")

        results.append({
            "index": i,
            "category": case["category"],
            "input": case["input"],
            "expected_focus": case["expected_focus"],
            "response": response,
        })

    # 保存结果
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n测试结果已保存至: {output_file}")

    return results


def compare_models(base_model_path: str, finetuned_model_path: str, base_model_name: str):
    """对比基线模型和微调模型"""
    print("\n" + "=" * 60)
    print("模型对比测试")
    print("=" * 60)

    # 加载基线模型
    print("\n加载基线模型...")
    base_model, base_tokenizer = load_model(base_model_path)

    # 加载微调模型
    print("加载微调模型...")
    ft_model, ft_tokenizer = load_model(finetuned_model_path, base_model_name)

    comparison_results = []

    for case in TEST_CASES[:4]:  # 只测试前4个用例
        print(f"\n{'='*60}")
        print(f"测试: {case['category']}")
        print(f"输入: {case['input']}")
        print("-" * 60)

        # 基线模型回复
        print("\n[基线模型回复]:")
        base_response = generate_response(base_model, base_tokenizer, case["input"])
        print(base_response)

        # 微调模型回复
        print("\n[微调模型回复]:")
        ft_response = generate_response(ft_model, ft_tokenizer, case["input"])
        print(ft_response)

        comparison_results.append({
            "category": case["category"],
            "input": case["input"],
            "base_response": base_response,
            "finetuned_response": ft_response,
        })

    return comparison_results


def main():
    parser = argparse.ArgumentParser(description="情绪价值模型推理测试")

    parser.add_argument("--model_path", type=str, default="./output/emotional_qwen/final_model",
                        help="模型路径（LoRA 适配器或完整模型）")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="基础模型路径（LoRA 模式下需要）")
    parser.add_argument("--mode", type=str, choices=["interactive", "batch", "compare"],
                        default="interactive", help="测试模式")
    parser.add_argument("--output_file", type=str, default="./test_results.json",
                        help="批量测试结果输出文件")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="是否使用 4bit 量化加载")

    args = parser.parse_args()

    if args.mode == "compare":
        # 对比模式
        compare_results = compare_models(
            base_model_path=args.base_model,
            finetuned_model_path=args.model_path,
            base_model_name=args.base_model
        )
        print("\n对比测试完成！")
    else:
        # 加载模型
        model, tokenizer = load_model(
            model_path=args.model_path,
            base_model_name=args.base_model,
            load_in_4bit=args.load_in_4bit
        )

        if args.mode == "interactive":
            run_interactive(model, tokenizer)
        elif args.mode == "batch":
            run_batch_test(model, tokenizer, args.output_file)


if __name__ == "__main__":
    main()