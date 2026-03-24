<p align="center"><img src="https://raw.githubusercontent.com/ikun-llm/.github/main/profile/logo.png" width="120" /></p>
<h2 align="center">ikun-tokenizer</h2>
<p align="center"><b>分词器是怎么炼成的</b><br/><sub>Level 1 | 基础篇</sub></p>

---

> 为什么分词器能认识"鸡你太美"但不认识"小黑子"？因为练习时长不够！

## 你将学到

- BPE (Byte Pair Encoding) 分词算法原理
- 为什么 vocab_size=6400？大了小了会怎样？
- 特殊 token 的设计：`<|im_start|>` / `<|im_end|>` 是什么
- 训练一个 ikun 专属分词器的完整流程
- 分词器如何把"鸡你太美baby鸡你太美"拆成 token

## 核心代码

基于 [MiniMind](https://github.com/jingyaogong/minimind) 的 `trainer/train_tokenizer.py`

```bash
# 训练一个自己的分词器
cd trainer && python train_tokenizer.py
```

## 知识点

| 概念 | 说明 |
|------|------|
| BPE | 从字符级开始，逐步合并高频字节对 |
| vocab_size | 词表大小，太小=表达力不足，太大=embedding 占太多参数 |
| Special Tokens | `<\|im_start\|>`=对话开始, `<\|im_end\|>`=对话结束 |
| Pre-tokenizer | ByteLevel 预分词，处理多语言 |

## 系列导航

| Level | Repo | 学什么 |
|-------|------|--------|
| **1** | **ikun-tokenizer** <-- 你在这里 | 分词器原理 |
| 1 | [ikun-pretrain](https://github.com/ikun-llm/ikun-pretrain) | 从零预训练 |
| 1 | [ikun-2.5B](https://github.com/ikun-llm/ikun-2.5B) | SFT + LoRA 微调 |
| 2 | [ikun-DPO](https://github.com/ikun-llm/ikun-DPO) | 偏好对齐 |
| 2 | [ikun-GRPO](https://github.com/ikun-llm/ikun-GRPO) | 强化学习 |
| 2 | [ikun-Reason](https://github.com/ikun-llm/ikun-Reason) | 推理模型 |
| 3 | [ikun-MoE](https://github.com/ikun-llm/ikun-MoE) | 混合专家 |
| 3 | [ikun-Distill](https://github.com/ikun-llm/ikun-Distill) | 知识蒸馏 |
| 3 | [ikun-V](https://github.com/ikun-llm/ikun-V) | 多模态 |
| 4 | [ikun-deploy](https://github.com/ikun-llm/ikun-deploy) | 部署 |
