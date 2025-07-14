import os
import torch
import json
import argparse
import pandas as pd
import math
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch.nn.functional as F

class Selector:
    """基于相似度的数据选择器"""
    
    def __init__(self, model_path: str, max_length: int = 512, mode: str = 'entk'):
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        
        print(f"🔄 加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✅ 模型加载完成，设备: {self.device}")
        print(f"🧮 当前相似性计算模式: {self.mode}")
    
    def get_gradient_for_input(self, messages) -> torch.Tensor:
        """计算输入消息列表的梯度向量"""
        self.model.zero_grad()
        try:
            # 用chat模板格式化
            chat_str = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            # 分词
            inputs = self.tokenizer(
                chat_str,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            )
            with torch.enable_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                y_hat = torch.mean(logits)
            y_hat.backward()
            gradients = []
            for param in self.model.parameters():
                if param.grad is not None:
                    gradients.append(param.grad.flatten().cpu())
            grad_vector = torch.cat(gradients)
            return grad_vector.detach().clone()
        finally:
            # 清理计算图和中间变量
            self.model.zero_grad()
            torch.cuda.empty_cache()
    
    def compute_entk_similarity(self, messages1, messages2) -> float:
        """计算两个消息列表的ENTK内积相似性"""
        self.model.train()
        grad1 = self.get_gradient_for_input(messages1)
        grad2 = self.get_gradient_for_input(messages2)
        grad1_norm = grad1.norm().item()
        grad2_norm = grad2.norm().item()
        if grad1_norm == 0 or grad2_norm == 0:
            del grad1, grad2
            torch.cuda.empty_cache()
            return 0.0
        if not torch.isfinite(grad1).all() or not torch.isfinite(grad2).all():
            del grad1, grad2
            torch.cuda.empty_cache()
            return 0.0
        inner_product = torch.dot(grad1, grad2).item()
        if not math.isfinite(inner_product):
            inner_product = 0.0
        del grad1, grad2
        torch.cuda.empty_cache()
        return inner_product

    def compute_hidden_similarity(self, messages1, messages2) -> float:
        """只对assistant部分做mean-pool后归一化，计算两个chosen和rejected的last layer hidden state相似性"""
        self.model.eval()
        with torch.no_grad():
            def get_assistant_emb(messages):
                # 拆分prompt和assistant
                prompt_msgs = [m for m in messages if m.get("role") != "assistant"]
                assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
                if len(assistant_msgs) == 0:
                    return None
                # 只支持最后一条assistant消息
                reply = assistant_msgs[-1]["content"]
                # prompt token数
                prompt_ids = self.tokenizer.apply_chat_template(
                    prompt_msgs,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                n_prompt = prompt_ids.shape[-1]
                # 拼接assistant
                full_ids = self.tokenizer.apply_chat_template(
                    prompt_msgs + [{"role": "assistant", "content": reply}],
                    add_generation_prompt=False,
                    return_tensors="pt"
                ).to(self.model.device)
                outs = self.model(full_ids, output_hidden_states=True)
                hid = outs.hidden_states[-1][0]  # [seq_len, hidden_dim]
                if n_prompt >= hid.shape[0]:
                    return None
                emb = hid[n_prompt:].mean(0)
                return F.normalize(emb, dim=0)
            emb1 = get_assistant_emb(messages1)
            emb2 = get_assistant_emb(messages2)
            if emb1 is None or emb2 is None:
                return 0.0
            cos_sim = torch.dot(emb1, emb2).item()
            return cos_sim

    def compute_similarity(self, messages1, messages2) -> float:
        if self.mode == 'entk':
            return self.compute_entk_similarity(messages1, messages2)
        elif self.mode == 'last_layer':
            return self.compute_hidden_similarity(messages1, messages2)
        else:
            raise ValueError(f"相似性计算模式: {self.mode}")

def load_ultrafeedback_data(split="train", max_samples=None):
    """加载ultrafeedback_binarized数据集"""
    print(f"📥 加载数据集: trl-lib/ultrafeedback_binarized, split={split}")
    
    dataset = load_dataset("trl-lib/ultrafeedback_binarized", split=split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"📊 使用样本数: {len(dataset)}")
    else:
        print(f"📊 总样本数: {len(dataset)}")
    
    return dataset

def compute_and_save_similarities(dataset, selector, output_file, max_pairs=None):
    mode = selector.mode
    print(f"🔄 开始计算{mode}相似性并保存到: {output_file}")
    results = []
    total_pairs = len(dataset)
    if max_pairs:
        total_pairs = min(max_pairs, total_pairs)
    with tqdm(total=total_pairs, desc=f"计算{mode}相似性") as pbar:
        for i, example in enumerate(dataset):
            if max_pairs and i >= max_pairs:
                break
            chosen = example.get("chosen", [])
            rejected = example.get("rejected", [])
            # 直接用消息列表
            messages_chosen = chosen if isinstance(chosen, list) else [chosen]
            messages_rejected = rejected if isinstance(rejected, list) else [rejected]
            try:
                similarity = selector.compute_similarity(messages_chosen, messages_rejected)
                result = {
                    "chosen": chosen,
                    "rejected": rejected,
                    "similarity": similarity,
                    "score_chosen": example.get("score_chosen", ""),
                    "score_rejected": example.get("score_rejected", "")
                }
                results.append(result)
                pbar.set_postfix({
                    f"{mode}_sim": f"{similarity:.4f}",
                    "当前数量": len(results)
                })
            except Exception as e:
                print(f"⚠️ 处理样本 {i} 时出错: {e}")
                continue
            pbar.update(1)
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"✅ 计算完成，共处理 {len(results)} 个数据对")
    print(f"💾 结果已保存到: {output_file}")
    return df

def load_and_filter_from_csv(csv_file, keep_percentage=None, similarity_threshold=None):
    """从CSV文件加载数据并进行过滤"""
    print(f"📥 从CSV文件加载数据: {csv_file}")
    
    df = pd.read_csv(csv_file)
    print(f"📊 加载了 {len(df)} 个数据对")
    
    # 按相似性排序（降序）
    df_sorted = df.sort_values('similarity', ascending=False).reset_index(drop=True)
    
    # 根据百分比或阈值过滤数据
    if keep_percentage is not None:
        # 按百分比保留最相似的数据
        keep_count = int(len(df_sorted) * keep_percentage)
        df_filtered = df_sorted.head(keep_count)
        
        print(f"🔍 原始数据: {len(df_sorted)} 对")
        print(f"🔍 保留前{keep_percentage*100:.1f}%: {len(df_filtered)} 对")
        print(f"🔍 删除数据: {len(df_sorted) - len(df_filtered)} 对")
        
        if len(df_filtered) > 0:
            threshold_value = df_filtered['similarity'].iloc[-1]
            print(f"🔍 实际阈值: {threshold_value:.4f}")
    
    elif similarity_threshold is not None:
        # 按固定阈值过滤
        df_filtered = df_sorted[df_sorted['similarity'] >= similarity_threshold]
        
        print(f"🔍 过滤前: {len(df_sorted)} 对")
        print(f"🔍 过滤后: {len(df_filtered)} 对")
        if len(df_sorted) > 0:
            print(f"🔍 过滤比例: {(len(df_sorted) - len(df_filtered)) / len(df_sorted) * 100:.1f}%")
    
    else:
        df_filtered = df_sorted
    
    if len(df_filtered) > 0:
        similarities = df_filtered['similarity'].values
        print(f"📈 保留数据相似性统计:")
        print(f"   最大值: {similarities.max():.4f}")
        print(f"   最小值: {similarities.min():.4f}")
        print(f"   平均值: {similarities.mean():.4f}")
        print(f"   中位数: {np.median(similarities):.4f}")
    
    return df_filtered

def save_training_dataset(df_filtered, output_dir):
    """将过滤后的数据保存为训练数据集格式"""
    print("💾 保存训练数据集...")
    
    # 保存为JSON格式（用于HuggingFace datasets）
    training_data = []
    for _, row in df_filtered.iterrows():
        # 重构为原始数据集格式
        training_sample = {
            "chosen": eval(row["chosen"]) if isinstance(row["chosen"], str) else row["chosen"],
            "rejected": eval(row["rejected"]) if isinstance(row["rejected"], str) else row["rejected"],
            "score_chosen": row.get("score_chosen", ""),
            "score_rejected": row.get("score_rejected", ""),
            "similarity": row["similarity"]
        }
        training_data.append(training_sample)
    
    # 保存为JSON Lines格式（便于加载）
    jsonl_file = output_dir / "filtered_training_data.jsonl"
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for sample in training_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 保存为单个JSON文件
    json_file = output_dir / "filtered_training_data.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    # 保存数据集信息
    dataset_info = {
        "total_samples": len(training_data),
        "format": "DPO preference dataset",
        "fields": ["chosen", "rejected", "similarity"],
        "similarity_range": {
            "min": float(df_filtered['similarity'].min()),
            "max": float(df_filtered['similarity'].max()),
            "mean": float(df_filtered['similarity'].mean())
        },
        "usage": {
            "load_jsonl": "dataset = load_dataset('json', data_files='filtered_training_data.jsonl')",
            "load_json": "dataset = load_dataset('json', data_files='filtered_training_data.json')"
        }
    }
    
    info_file = output_dir / "dataset_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print(f"📁 训练数据已保存:")
    print(f"   📄 JSONL格式: {jsonl_file}")
    print(f"   📄 JSON格式: {json_file}")
    print(f"   📄 数据集信息: {info_file}")
    print(f"📊 训练样本数: {len(training_data)}")
    
    return training_data

def main():
    parser = argparse.ArgumentParser(description="ENTK数据选择工具")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="用于计算ENTK的模型路径"
    )
    
    parser.add_argument(
        "--keep_percentage",
        type=float,
        default=None,
        help="保留最相似的前X%数据，例如0.95表示保留前95%数据"
    )
    
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=None,
        help="ENTK相似性阈值，低于此值的数据对将被过滤"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="最大处理样本数（-1表示处理全部）"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录"
    )
    
    parser.add_argument(
        "--csv_file",
        type=str,
        default=None,
        help="已有的CSV文件路径（如果提供，将直接从此文件加载而不重新计算）"
    )
    
    parser.add_argument(
        "--compute_only",
        action="store_true",
        help="只计算相似性并保存CSV，不进行过滤"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="数据集分割（train/test/validation）"
    )
    
    parser.add_argument(
        "--random_drop_percentage",
        type=float,
        default=None,
        help="随机丢弃数据的比例（如0.2表示随机丢弃20%）"
    )
    
    parser.add_argument(
        "--similarity_mode",
        type=str,
        choices=["entk", "last_layer"],
        default=None,
        help="相似度计算模式，可选 'entk' 或 'last_layer'"
    )
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"entk_filtered_data_{timestamp}"
    
    if args.max_samples == -1:
        args.max_samples = None
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("🔬 数据筛选工具")
    print("=" * 60)
    print(f"🤖 模型路径: {args.model_path}")
    print(f"📈 最大样本数: {args.max_samples if args.max_samples else '全部'}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"📚 数据分割: {args.split}")

    # 参数互斥性检查
    random_drop_valid = args.random_drop_percentage is not None and 0 < args.random_drop_percentage < 1
    similarity_mode_valid = args.similarity_mode is not None
    if random_drop_valid == similarity_mode_valid:
        raise ValueError("Mode Conflict!")

    # 1. 随机丢弃分支
    if random_drop_valid:
        print(f"⚠️ 仅做随机丢弃 {args.random_drop_percentage*100:.1f}% 的数据，不进行模型推理")
        dataset = load_ultrafeedback_data(split=args.split, max_samples=args.max_samples)
        import random
        keep_num = int(len(dataset) * (1 - args.random_drop_percentage))
        keep_indices = sorted(random.sample(range(len(dataset)), keep_num))
        dataset = dataset.select(keep_indices)
        results = []
        for example in dataset:
            result = {
                "chosen": example.get("chosen", []),
                "rejected": example.get("rejected", []),
                "score_chosen": example.get("score_chosen", ""),
                "score_rejected": example.get("score_rejected", "")
            }
            results.append(result)
        df = pd.DataFrame(results)
        csv_file = output_dir / "random_filtered_data.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"💾 随机丢弃后数据保存到: {csv_file}")
        # 保存为jsonl
        jsonl_file = output_dir / "random_filtered_data.jsonl"
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for sample in results:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        # 保存为json
        json_file = output_dir / "random_filtered_data.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"📁 训练数据已保存:")
        print(f"   📄 CSV格式: {csv_file}")
        print(f"   📄 JSONL格式: {jsonl_file}")
        print(f"   📄 JSON格式: {json_file}")
        print(f"📊 训练样本数: {len(results)}")
        print("🎉 随机丢弃数据处理完成!")
        return

    # 2. 基于相似度的分支
    similarity_mode = args.similarity_mode
    print(f"当前相似度计算模式: {similarity_mode}")

    # 2.1 有csv直接加载，否则做推理生成csv
    if args.csv_file and Path(args.csv_file).exists():
        print("📥 从已有CSV文件加载数据...")
        df = pd.read_csv(args.csv_file)
    else:
        print("🔄 计算相似性...")
        selector = Selector(args.model_path, mode=similarity_mode)
        dataset = load_ultrafeedback_data(split=args.split, max_samples=args.max_samples)
        csv_file = output_dir / "entk_similarities.csv"
        df = compute_and_save_similarities(dataset, selector, csv_file, args.max_samples)

    # 3. 分数过滤
    if args.keep_percentage is not None or args.similarity_threshold is not None:
        print("\n🔍 开始过滤数据...")
        df_filtered = load_and_filter_from_csv(csv_file, args.keep_percentage, args.similarity_threshold)
        # 保存过滤后的结果
        filtered_csv = output_dir / "entk_filtered_data.csv"
        df_filtered.to_csv(filtered_csv, index=False, encoding='utf-8')
        print(f"💾 过滤后数据保存到: {filtered_csv}")
        # 保存为训练数据集格式
        training_data = save_training_dataset(df_filtered, output_dir)
    else:
        print("⚠️ 未指定过滤条件，跳过过滤步骤")
    print("🎉 数据选择完成!")

if __name__ == "__main__":
    main()