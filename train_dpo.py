import os
import torch
# os.environ["WANDB_DISABLED"] = "true"
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from MyDPOTrainer1 import DPOConfig, DPOTrainer
import argparse
from datetime import datetime
from accelerate import Accelerator

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi‑GPU DPO fine‑tuning with TRL + Accelerate/DeepSpeed"
    )
    parser.add_argument(
        "--model_name", type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model for policy and reference"
    )
    parser.add_argument(
        "--dataset_name", type=str,
        default="trl-lib/ultrafeedback_binarized",
        help="Hugging Face dataset for preference pairs"
    )
    parser.add_argument(
        "--local_dataset_path", type=str,
        default=None,
        help="Path to local dataset file (JSON/JSONL format). If provided, will use local dataset instead of HuggingFace dataset"
    )
    parser.add_argument(
        "--split", type=str,
        default="train",
        help="Dataset split to use"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/common/users/Qwen2.5-0.5B-random95",
        help="Where to save checkpoints and logs"
    )
    parser.add_argument(
        "--deepspeed_config", type=str,
        default="ds_config.json",
        help="Path to DeepSpeed JSON config"
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int,
        default=4,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int,
        default=8,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate", type=float,
        default=1e-6,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_steps", type=int,
        default=-1,
        help="Total training steps (use -1 for epoch-based training)"
    )
    parser.add_argument(
        "--num_train_epochs", type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--beta", type=float,
        default=0.1,
        help="DPO temperature beta"
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="Enable mixed-precision fp16 training"
    )

    parser.add_argument(
        "--loss_type", type=str,
        default="sigmoid",
        choices=["sigmoid", "hinge", "ipo", "bco_pair", "sppo_hard", "nca_pair", "robust", "exo_pair", "ENTK_dpo"],
        help="Type of DPO loss function to use"
    )

    
    return parser.parse_args()


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    accelerator = Accelerator()

    args = parse_args()
    print(f"RANK: {os.environ.get('RANK')}, LOCAL_RANK: {os.environ.get('LOCAL_RANK')}, CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}, torch.cuda.current_device(): {torch.cuda.current_device()}")
    # 验证本地数据集路径
    if args.local_dataset_path:
        if not os.path.exists(args.local_dataset_path):
            raise FileNotFoundError(f"本地数据集文件不存在: {args.local_dataset_path}")
        print(f"✅ 本地数据集文件验证通过: {args.local_dataset_path}")

    # Load models & tokenizer
    policy = AutoModelForCausalLM.from_pretrained(args.model_name)
    reference = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 删除这行：policy = accelerator.prepare(policy)  # ❌

    # Load dataset
    if args.local_dataset_path:
        print(f"📥 加载本地数据集: {args.local_dataset_path}")
        # 支持JSON和JSONL格式
        if args.local_dataset_path.endswith('.jsonl'):
            dataset = load_dataset('json', data_files=args.local_dataset_path, split='train')
        elif args.local_dataset_path.endswith('.json'):
            dataset = load_dataset('json', data_files=args.local_dataset_path, split='train')
        else:
            # 尝试自动检测格式
            dataset = load_dataset('json', data_files=args.local_dataset_path, split='train')
        print(f"✅ 本地数据集加载完成，共 {len(dataset)} 个样本")
        
        
    else:
        print(f"📥 加载HuggingFace数据集: {args.dataset_name}")
        dataset = load_dataset(args.dataset_name, split=args.split)
        print(f"✅ HuggingFace数据集加载完成，共 {len(dataset)} 个样本")
    
  
    # Build DPO configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dpo_config = DPOConfig(
        output_dir=f"{args.output_dir}-{args.loss_type}-{timestamp}",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        beta=args.beta,
        fp16=args.fp16,
        deepspeed=args.deepspeed_config,
        loss_type=args.loss_type,
        # Checkpoint saving - only keep 2 最新的checkpoint
        save_strategy="epoch",
        save_total_limit=2,
    )
    dpo_config.use_liger_loss = False
    print(dpo_config)
    
    # Initialize DPO trainer - 直接传原始模型
    trainer = DPOTrainer(
        model=policy,        # 传原始 policy
        ref_model=reference, # 传原始 reference
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Run training
    trainer.train()


if __name__ == "__main__":
    main()
