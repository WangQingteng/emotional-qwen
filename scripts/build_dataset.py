"""
开源情绪对话数据集下载与构建脚本
支持多个 HuggingFace 开源中文情绪数据集
"""
import os
import json
import argparse
from datasets import load_dataset, Dataset


DATASET_CONFIGS = {
    "emochat": {
        "name": "miemie/EmoChat",
        "description": "EmoChat 情绪对话数据集",
    },
    "emotional_qa": {
        "name": "siliconflow/Emotional_QA",
        "description": "情感问答数据集",
    },
    "emo_dialogue": {
        "name": "wzy664/EmotionalDialogue",
        "description": "情感对话数据集",
    },
}


def download_dataset(dataset_name: str, output_dir: str):
    """下载指定数据集"""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"未知数据集: {dataset_name}, 可选: {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[dataset_name]
    print(f"下载数据集: {config['name']} ({config['description']})")

    dataset = load_dataset(config["name"], trust_remote_code=True)

    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    dataset.save_to_disk(dataset_dir)
    print(f"数据集已保存至: {dataset_dir}")

    return dataset


def convert_to_sft_format(dataset, output_path: str, max_samples: int = None):
    """
    将数据集转换为 SFT 格式 (conversations)
    统一格式:
    [
        {
            "conversations": [
                {"from": "human", "value": "用户输入"},
                {"from": "gpt", "value": "助理回复"}
            ]
        },
        ...
    ]
    """
    samples = []

    if isinstance(dataset, dict):
        for split_name, split_dataset in dataset.items():
            print(f"处理 split: {split_name}, 样本数: {len(split_dataset)}")
            for item in split_dataset:
                converted = convert_item(item)
                if converted:
                    samples.extend(converted)
    else:
        for item in dataset:
            converted = convert_item(item)
            if converted:
                samples.extend(converted)

    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"转换完成，共 {len(samples)} 条样本，保存至: {output_path}")
    return samples


def convert_item(item: dict) -> list:
    """将单条数据转换为对话格式"""
    results = []

    # EmoChat 格式: {instruction, input, output}
    if "instruction" in item and "output" in item:
        user_input = item.get("input", "") or item.get("instruction", "")
        assistant_output = item["output"]
        if user_input and assistant_output:
            results.append({
                "conversations": [
                    {"from": "human", "value": user_input},
                    {"from": "gpt", "value": assistant_output}
                ]
            })

    # Emotional_QA 格式: {question, answer}
    elif "question" in item and "answer" in item:
        results.append({
            "conversations": [
                {"from": "human", "value": item["question"]},
                {"from": "gpt", "value": item["answer"]}
            ]
        })

    # EmotionalDialogue 格式: {prompt, response}
    elif "prompt" in item and "response" in item:
        prompt = item["prompt"]
        response = item["response"]
        if isinstance(prompt, str):
            results.append({
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": response}
                ]
            })
        elif isinstance(prompt, list):
            conv = []
            for msg in prompt:
                role = "human" if msg.get("role") == "user" else "gpt"
                conv.append({"from": role, "value": msg.get("content", "")})
            conv.append({"from": "gpt", "value": response})
            results.append({"conversations": conv})

    # 通用格式: {user, assistant} 或 {query, reply} 等
    else:
        user_key = None
        assistant_key = None

        for key in ["user", "query", "question", "input", "prompt"]:
            if key in item:
                user_key = key
                break
        for key in ["assistant", "reply", "answer", "response", "output"]:
            if key in item:
                assistant_key = key
                break

        if user_key and assistant_key:
            results.append({
                "conversations": [
                    {"from": "human", "value": str(item[user_key])},
                    {"from": "gpt", "value": str(item[assistant_key])}
                ]
            })

    return results


def build_combined_dataset(dataset_names: list, output_dir: str, max_per_dataset: int = 50000):
    """
    下载并合并多个数据集
    """
    combined_samples = []

    for name in dataset_names:
        print(f"\n{'='*50}")
        print(f"处理数据集: {name}")
        print('=' * 50)

        try:
            dataset = download_dataset(name, output_dir)

            temp_path = os.path.join(output_dir, f"{name}_sft.jsonl")
            samples = convert_to_sft_format(dataset, temp_path, max_samples=max_per_dataset)
            combined_samples.extend(samples)
            print(f"✓ {name} 处理完成，获得 {len(samples)} 条样本")

        except Exception as e:
            print(f"✗ {name} 处理失败: {e}")
            continue

    # 保存合并后的数据
    if combined_samples:
        combined_path = os.path.join(output_dir, "combined_emotional_sft.jsonl")
        with open(combined_path, "w", encoding="utf-8") as f:
            for sample in combined_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"\n合并完成，共 {len(combined_samples)} 条样本")
        print(f"保存至: {combined_path}")
        return combined_samples
    else:
        print("没有成功获取任何样本")
        return []


def add_custom_emotional_samples(output_dir: str):
    """
    添加高质量的自定义情绪价值对话样本
    涵盖安慰、鼓励、共情、情感支持等场景
    """
    custom_samples = [
        # 安慰类
        {"conversations": [
            {"from": "human", "value": "我今天心情不好，工作压力太大了"},
            {"from": "gpt", "value": "我能感受到你现在承受着很大的压力，真的很不容易。工作压力大的时候，情绪低落是很正常的，你不用强迫自己马上好起来。\n\n或许我们可以一起想想，是什么让你感到特别疲惫？是任务太多，还是对自己的要求太高了呢？"}
        ]},
        {"conversations": [
            {"from": "human", "value": "我觉得自己什么都做不好，没有自信"},
            {"from": "gpt", "value": "你会这样想，说明你对自己有很高的期望，这本身就说明你是一个有追求的人。很多时候，我们对自己最苛刻，却对别人很宽容。\n\n能不能告诉我，最近发生了什么让你有这样的感受？也许我们可以一起梳理一下，看看事情是否真的像你想的那么糟糕。"}
        ]},
        {"conversations": [
            {"from": "human", "value": "我失恋了，很难过"},
            {"from": "gpt", "value": "失恋真的是一件非常痛苦的事情，允许自己难过、哭泣都是很正常的。一段感情的结束不代表你不值得被爱，只是双方可能不再适合了。\n\n你愿意说说你们之间发生了什么吗？倾诉出来可能会让你感觉好一些。无论如何，我都在这里陪着你。"}
        ]},
        # 鼓励类
        {"conversations": [
            {"from": "human", "value": "我要放弃了，感觉坚持不下去了"},
            {"from": "gpt", "value": "你能走到今天，已经证明了你的坚持和勇气。想要放弃的念头恰恰说明你一直在努力着，这本身就很了不起。\n\n可以告诉我是什么让你觉得特别累吗？有时候我们不需要一个解决方案，只是需要有人理解我们的感受。你愿意给自己再多一点时间吗？"}
        ]},
        {"conversations": [
            {"from": "human", "value": "我总是担心未来，焦虑到睡不着"},
            {"from": "gpt", "value": "对未来感到焦虑是很正常的反应，尤其是当我们面临不确定的时候。你试着在脑海中预演各种可能的结果，但这反而增加了心理负担。\n\n或许我们可以尝试一个方法：把你担心的事情写下来，然后问自己，'这些事情真的都在我的控制范围内吗？'很多时候，我们焦虑的大部分事情其实并不一定会发生。\n\n你想试试告诉我你具体在担心什么吗？"}
        ]},
        # 共情类
        {"conversations": [
            {"from": "human", "value": "今天和朋友吵架了，心里很不是滋味"},
            {"from": "gpt", "value": "和朋友发生争执后心里难受是很自然的，毕竟你在乎这段关系。能感觉到你重视友谊，这是一件美好的事情。\n\n你愿意告诉我事情的经过吗？有时候把心里的委屈说出来，就已经是在慢慢疗愈了。无论谁对谁错，你的感受都是真实和重要的。"}
        ]},
        {"conversations": [
            {"from": "human", "value": "我觉得没有人理解我"},
            {"from": "gpt", "value": "那种觉得不被理解的孤独感真的很让人难过。我想让你知道，在这里，我会认真地听你说，努力去理解你的感受。\n\n你愿意告诉我，你最想被理解的是什么吗？有时候，我们不需要别人的建议，只是需要有人愿意倾听。我就在这里。"}
        ]},
        # 积极引导
        {"conversations": [
            {"from": "human", "value": "我讨厌现在的生活状态"},
            {"from": "gpt", "value": "对当前的生活感到不满意，其实是内心在呼唤改变，这是一件好事！它说明你还没有放弃对更好生活的追求。\n\n能不能告诉我，具体是哪些方面让你感到不满？是工作、人际关系，还是个人成长？当我们能够清楚地表达出来的时候，改变的第一步就已经迈出了。"}
        ]},
        {"conversations": [
            {"from": "human", "value": "为什么我总是这么倒霉，事事不顺心"},
            {"from": "gpt", "value": "连续遇到不顺心的事情时，有这样的想法是很正常的。你不是一个倒霉的人，只是人生中总有一些阶段性的低谷期。\n\n我想让你知道，困难是暂时的，就像天气一样，阴雨天过后总会有晴天。你能走到现在已经很不容易了，这说明你有足够的韧性去面对这些挑战。\n\n你愿意和我说说最近具体遇到了哪些事情吗？"}
        ]},
        # 自我关怀
        {"conversations": [
            {"from": "human", "value": "我总是对自己要求太高，活得好累"},
            {"from": "gpt", "value": "能够意识到自己对自己要求太高，这是一种很好的自我觉察。你值得被温柔对待，特别是来自你自己的温柔。\n\n试试对自己说：'我已经尽力了，我值得休息和被爱。'允许自己不完美，允许自己偶尔偷懒，这不是懒惰，而是自我关怀。\n\n你可以告诉我，你通常用什么标准来要求自己吗？我们可以一起看看这些标准是否过于苛刻了。"}
        ]},
        {"conversations": [
            {"from": "human", "value": "我害怕失败，所以不敢尝试任何新事物"},
            {"from": "gpt", "value": "害怕失败是人之常情，因为我们都希望避免痛苦和失望。但是你知道吗？很多时候，我们害怕的并不是失败本身，而是失败后别人的评价，或者对自己的否定。\n\n或许我们可以重新定义'失败'：它不是终点，而是通往成功路上的反馈信息。每一次'失败'都在告诉你，'下次可以试试别的方法'。\n\n你愿意和我聊聊，有什么想尝试但一直没敢开始的事情吗？我们可以一起分析，把恐惧拆分成小块。"}
        ]},
    ]

    output_path = os.path.join(output_dir, "custom_emotional_sft.jsonl")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in custom_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"自定义样本已保存至: {output_path}, 共 {len(custom_samples)} 条")

    return custom_samples


def main():
    parser = argparse.ArgumentParser(description="情绪价值对话数据集构建工具")
    parser.add_argument(
        "--datasets", nargs="+",
        default=["emochat", "emotional_qa"],
        choices=list(DATASET_CONFIGS.keys()),
        help="要下载的数据集名称"
    )
    parser.add_argument("--output_dir", type=str, default="./data", help="输出目录")
    parser.add_argument("--max_per_dataset", type=int, default=50000, help="每个数据集最大样本数")
    parser.add_argument("--add_custom", action="store_true", help="是否添加自定义情绪样本")

    args = parser.parse_args()

    print("=" * 60)
    print("情绪价值对话数据集构建工具")
    print("=" * 60)

    # 1. 下载并处理开源数据集
    combined = build_combined_dataset(
        dataset_names=args.datasets,
        output_dir=args.output_dir,
        max_per_dataset=args.max_per_dataset
    )

    # 2. 添加自定义高质量样本
    if args.add_custom:
        custom = add_custom_emotional_samples(args.output_dir)
        if combined:
            # 合并到 combined 文件
            combined_path = os.path.join(args.output_dir, "combined_emotional_sft.jsonl")
            with open(combined_path, "a", encoding="utf-8") as f:
                for sample in custom:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            print(f"自定义样本已追加至合并文件，总计 {len(combined) + len(custom)} 条")

    print("\n" + "=" * 60)
    print("数据集构建完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()