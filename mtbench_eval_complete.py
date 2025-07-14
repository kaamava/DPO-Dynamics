#!/usr/bin/env python3
"""
MT-Bench 答案生成脚本
使用 vLLM 加速生成模型答案
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import time
import shutil

# 添加 FastChat 到 Python 路径
sys.path.append('/home/mjin/FastChat')

from vllm import LLM, SamplingParams
from fastchat.llm_judge.common import load_questions
from fastchat.conversation import get_conv_template

def setup_directories():
    """创建必要的目录结构"""
    dirs = [
        "/home/mjin/FastChat/fastchat/llm_judge/data/mt_bench/model_answer"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"创建目录: {dir_path}")

def generate_model_answers(model_path: str, model_name: str):
    """使用 vLLM 为指定模型生成 MT-Bench 答案"""
    print(f"\n开始为模型 {model_name} 生成答案...")
    
    # 加载问题
    questions = load_questions("/home/mjin/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl", None, None)
    
    # 初始化 vLLM 模型
    try:
        llm = LLM(
            model=model_path,
            dtype="float16",
            max_model_len=4096,
            tensor_parallel_size=1,
            enforce_eager=True,
            trust_remote_code=True
        )
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return False
    
    # 获取对话模板
    conv_template = get_conv_template("qwen-7b-chat")
    
    answers = []
    
    for question in questions:
        question_id = question["question_id"]
        category = question["category"]
        question_turns = question["turns"]
        
        # 重置对话模板
        conv_template.messages = []
        
        turns = []
        for turn_idx, turn in enumerate(question_turns):
            conv_template.append_message(conv_template.roles[0], turn)
            conv_template.append_message(conv_template.roles[1], None)
            prompt = conv_template.get_prompt()
            
            # 根据问题类别设置采样参数
            if category in ["math", "reasoning"]:
                sampling_params = SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=1024,
                    stop=["<|endoftext|>", "<|im_end|>"]
                )
            else:
                sampling_params = SamplingParams(
                    temperature=0.7,
                    top_p=0.8,
                    max_tokens=1024,
                    stop=["<|endoftext|>", "<|im_end|>"]
                )
            
            # 生成回答
            outputs = llm.generate([prompt], sampling_params)
            answer = outputs[0].outputs[0].text.strip()
            
            turns.append(answer)
            
            # 更新对话模板
            conv_template.update_last_message(answer)
        
        # 保存答案 - 使用 FastChat 标准格式
        choices = [{"index": 0, "turns": turns}]
        answer_data = {
            "question_id": question_id,
            "answer_id": f"{model_name}_{question_id}",
            "model_id": model_name,
            "choices": choices,
            "tstamp": time.time()
        }
        answers.append(answer_data)
        
        print(f"完成问题 {question_id} ({len(answers)}/{len(questions)})")
    
    # 保存答案到文件 - 确保模型名称不包含路径分隔符
    safe_model_name = model_name.replace("/", "-").replace("\\", "-")
    output_file = f"/home/mjin/FastChat/fastchat/llm_judge/data/mt_bench/model_answer/{safe_model_name}.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for answer in answers:
            f.write(json.dumps(answer, ensure_ascii=False) + '\n')
    
    print(f"模型 {model_name} 的答案已保存到: {output_file}")
    print(f"安全文件名: {safe_model_name}.jsonl")
    return True

def copy_required_files():
    """检查必要文件"""
    # 检查问题文件是否存在
    question_file = "/home/mjin/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl"
    if os.path.exists(question_file):
        print("问题文件已存在于 FastChat 目录中")
    else:
        print("警告：问题文件不存在！")
    
    print("文件检查完成")

def main():
    parser = argparse.ArgumentParser(description="MT-Bench 答案生成")
    parser.add_argument("--models", nargs='+', required=True, 
                       help="模型路径列表")
    parser.add_argument("--model-names", nargs='+', required=True,
                       help="模型名称列表")
    
    args = parser.parse_args()
    
    if len(args.models) != len(args.model_names):
        print("错误: 模型路径和名称的数量必须匹配")
        return
    
    print("=== MT-Bench 答案生成 ===")
    print(f"模型: {args.model_names}")
    
    # 创建目录结构
    setup_directories()
    
    # 复制必要文件
    copy_required_files()
    
    # 生成答案
    for model_path, model_name in zip(args.models, args.model_names):
        success = generate_model_answers(model_path, model_name)
        if not success:
            print(f"为模型 {model_name} 生成答案失败")
            return
    
    print("\n=== 答案生成完成 ===")
    print("结果文件位置:")
    print(f"- 模型答案: /home/mjin/FastChat/fastchat/llm_judge/data/mt_bench/model_answer/")

if __name__ == "__main__":
    main() 