#!/usr/bin/env python3
"""
AlpacaEval两模型对比评估脚本 - 使用VLLM进行高效推理
"""

import os
# 设置环境变量，避免分布式和网络问题
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 只使用第一个GPU（如果未设置）
os.environ["VLLM_USE_MODELSCOPE"] = "False"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["NCCL_P2P_DISABLE"] = "1"  # 禁用P2P通信
os.environ["NCCL_IB_DISABLE"] = "1"   # 禁用InfiniBand

import json
import argparse
import subprocess
import sys
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams

def generate_model_outputs(checkpoint_path, model_name, max_samples=10):
    """
    使用VLLM生成模型输出
    """
    print(f"🔄 生成{model_name}模型输出，样本数量: {max_samples if max_samples != -1 else '全部'}")
    
    # 加载分词器
    print(f"📥 加载分词器: {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 初始化VLLM模型
    print(f"🚀 初始化VLLM模型: {checkpoint_path}")
    
    # 使用简单的单GPU配置，避免分布式问题
    print(f"🔧 使用单GPU模式，避免分布式连接问题")
    
    llm = LLM(
        model=checkpoint_path,
        dtype="float16",
        trust_remote_code=True,
        max_model_len=2048,
        tensor_parallel_size=1,  # 强制使用单GPU，避免分布式问题
        enforce_eager=True,  # 使用eager模式，更稳定
    )
    print("✅ VLLM模型初始化成功")
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
        skip_special_tokens=True
    )
    
    # 加载AlpacaEval数据集
    print("📥 加载AlpacaEval数据集...")
    dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")
    
    # 处理样本数：-1表示使用全部样本
    if max_samples == -1:
        test_set = dataset
        actual_samples = len(dataset)
        print(f"📊 使用全部样本: {actual_samples}")
    else:
        actual_samples = min(max_samples, len(dataset))
        test_set = dataset.select(range(actual_samples))
        print(f"📊 使用样本数: {actual_samples}/{len(dataset)}")
    
    # 准备所有输入提示
    print("🔄 准备输入提示...")
    prompts = []
    instructions = []
    
    for example in test_set:
        instruction = example['instruction']
        instructions.append(instruction)
        
        # 构建输入
        messages = [
            {"role": "user", "content": instruction}
        ]
        
        try:
            # 应用聊天模板
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            # 如果没有聊天模板，使用简单格式
            text = f"User: {instruction}\nAssistant: "
        
        prompts.append(text)
    
    print(f"🚀 开始批量生成 {len(prompts)} 个回复...")
    
    # 使用VLLM批量生成
    outputs = llm.generate(prompts, sampling_params)
    print("✅ 批量生成成功完成")
    
    # 构建结果
    results = []
    for i, output in enumerate(outputs):
        response = output.outputs[0].text.strip()
        
        result = {
            "instruction": instructions[i],
            "output": response,
            "generator": model_name
        }
        results.append(result)
        
        print(f"✅ 完成样本 {i+1}/{len(outputs)}: 长度 {len(response)}")
    
    print(f"🎉 批量生成完成! 共处理 {len(results)} 个样本")
    return results

def run_alpaca_eval_comparison(model1_path, model2_path, output_dir, max_samples=10):
    """
    使用官方AlpacaEval进行两模型对比评估
    """
    print(f"🚀 开始两模型对比评估")
    print(f"📊 模型1（主评估）: {model1_path}")
    print(f"🔄 模型2（参考模型）: {model2_path}")
    
    # 验证路径
    model1_path = Path(model1_path)
    model2_path = Path(model2_path)
    
    if not model1_path.exists():
        print(f"❌ 模型1路径不存在: {model1_path}")
        return False
    
    if not model2_path.exists():
        print(f"❌ 模型2路径不存在: {model2_path}")
        return False
    
    # 设置输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 设置API密钥环境
    env = os.environ.copy()
    default_api_key = ""
    
    if "OPENAI_API_KEY" not in env:
        env["OPENAI_API_KEY"] = default_api_key
        print(f"✅ 使用默认API密钥")
    else:
        print(f"✅ 使用环境变量中的API密钥")
    
    try:
        # 步骤1: 生成模型1的输出
        print(f"\n🔄 步骤1: 生成模型1输出")
        model1_outputs = generate_model_outputs(str(model1_path), "model1", max_samples)
        
        # 步骤2: 生成模型2的输出（作为参考）
        print(f"\n🔄 步骤2: 生成模型2输出（参考模型）")  
        model2_outputs = generate_model_outputs(str(model2_path), "model2", max_samples)
        
        # 保存输出文件
        model1_file = output_dir / "model1_outputs.json"
        model2_file = output_dir / "model2_reference_outputs.json"
        
        with open(model1_file, 'w', encoding='utf-8') as f:
            json.dump(model1_outputs, f, ensure_ascii=False, indent=2)
        
        with open(model2_file, 'w', encoding='utf-8') as f:
            json.dump(model2_outputs, f, ensure_ascii=False, indent=2)
        
        print(f"💾 模型输出已保存:")
        print(f"   📁 模型1输出: {model1_file}")
        print(f"   📁 模型2参考输出: {model2_file}")
        
        # 步骤3: 使用AlpacaEval进行对比评估
        print(f"\n🔄 步骤3: 运行AlpacaEval对比评估")
        
        # 按照官方文档使用reference_outputs参数
        cmd = [
            "alpaca_eval", "evaluate",
            "--model_outputs", str(model1_file),
            "--reference_outputs", str(model2_file),
            "--annotators_config", "alpaca_eval_gpt4_turbo_fn",
            "--output_path", str(output_dir)
        ]
        
        print(f"🔧 执行命令: {' '.join(cmd)}")
        
        # 运行评估命令
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"✅ 对比评估完成! 结果保存在: {output_dir}")
        print(f"📝 评估输出:\n{result.stdout}")
        
        # 查找并读取结果文件
        results_files = list(output_dir.glob("**/leaderboard.csv"))
        if results_files:
            print(f"📊 结果文件: {results_files[0]}")
            try:
                with open(results_files[0], 'r') as f:
                    content = f.read()
                    print(f"📈 对比评估结果:\n{content}")
            except Exception as e:
                print(f"⚠️ 无法读取结果文件: {e}")
        
        # 创建评估摘要
        summary = {
            "model1_path": str(model1_path),
            "model2_path": str(model2_path),
            "evaluation_time": datetime.now().isoformat(),
            "max_samples": max_samples,
            "output_directory": str(output_dir),
            "model1_outputs_file": str(model1_file),
            "model2_reference_file": str(model2_file),
            "command_executed": ' '.join(cmd),
            "status": "success"
        }
        
        summary_file = output_dir / "comparison_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📋 对比摘要保存在: {summary_file}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 评估失败:")
        print(f"   错误代码: {e.returncode}")
        print(f"   错误输出: {e.stderr}")
        print(f"   标准输出: {e.stdout}")
        return False
    except Exception as e:
        print(f"❌ 意外错误: {e}")
        return False

def parse_args():
    parser = argparse.ArgumentParser(description="AlpacaEval两模型对比评估工具")
    
    parser.add_argument(
        "--model1_path", 
        type=str, 
        default="/research/projects/trans_llm/Yanshu_Li/conflict/RL/RLresults/Qwen2.5-1.5B-random30/checkpoint-4660",
        help="模型1路径（主评估模型）"
    )
    
    parser.add_argument(
        "--model2_path", 
        type=str, 
        default="/research/projects/trans_llm/Yanshu_Li/conflict/RL/RLresults/Qwen2.5-1.5B/checkpoint-15532",
        help="模型2路径（参考模型）"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/research/projects/trans_llm/Yanshu_Li/conflict/RL/alpacaresults/1.5B-random30",
        help="评估结果输出目录"
    )
    
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=-1,
        help="最大评估样本数（-1表示使用全部样本，当前AlpacaEval约805个样本）"
    )
    
    parser.add_argument(
        "--api_key", 
        type=str, 
        default=None,
        help="OpenAI API密钥"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("🔬 AlpacaEval 两模型对比评估工具")
    print("=" * 60)
    
    if args.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_dir = f"alpaca_eval_comparison_{timestamp}"
        
    print(f"🥇 模型1（主评估）: {args.model1_path}")
    print(f"🥈 模型2（参考模型）: {args.model2_path}")
    print(f"📊 评估样本数: {args.max_samples if args.max_samples != -1 else '全部(~805)'}")
    print(f"📁 输出目录: {args.output_dir}")
        
    success = run_alpaca_eval_comparison(
            model1_path=args.model1_path,
            model2_path=args.model2_path,
            output_dir=args.output_dir,
            max_samples=args.max_samples
        )
        


if __name__ == "__main__":
    main() 