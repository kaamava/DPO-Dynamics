## 模型和数据集使用
LLM：LLaMa3系列、Qwen2.5-0.5B,1.5B,3B
用于DPO的数据集:Ultrafeedback
用于评估DPO结果的benchmark:Alpaca-eval, mtbench (需要额外使用https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge中的实现）。
## 流程
1.使用dataselect.py对原Ultrafeedback数据集进行处理和筛选。筛选分为三个选项：随机筛选、基于last layer hidden state进行筛选以及基于ENTK进行筛选。执行第一种选项时，会直接筛掉对应比例的数据并获得新json;执行后两种选项时，会首先通过模型推理得到对应的相似度数值，存入csv文件中，再依据对应数值进行指定比例的筛选。
2.使用上一步得到的新数据集的json文件，通过run.sh开始进行DPO。请先登录自己的wandb账号以获得训练log。在run.sh中，你需要指定用到的卡、用于DPO的模型、用于DPO的数据集来源、输出目录以及一些超参数，epoch一般设为4。DPO完成后，会得到新模型的checkpoint。
3.使用Eval.py来使用Alpaca_eval或者mtbench_eval_complete.py来使用mt-bench来pairwise评估模型的DPO质量，参考模型一般是使用所有数据DPO后的模型，主评估模型是使用筛选后数据DPO后的模型。主要的评估指标为win_rate，即主评估模型胜过参考模型的比例，越高则主评估模型越强。
