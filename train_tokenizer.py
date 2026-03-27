"""
ikun-tokenizer: 训练一个 ikun 专属分词器
=============================================

基于 HuggingFace tokenizers 库，使用 ByteLevel BPE 算法
训练一个 vocab_size=6400 的分词器。

使用方法：
    pip install tokenizers
    python train_tokenizer.py
"""

import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


# ============================================================
# 配置参数
# ============================================================

VOCAB_SIZE = 6400               # 词表大小（为什么是 6400？看 docs/02-vocab.md）
MIN_FREQUENCY = 2               # 最低出现频率
CORPUS_DIR = "data"             # 语料目录
OUTPUT_FILE = "ikun_tokenizer.json"  # 输出文件

# 特殊 token（顺序决定了 ID，详见 docs/03-special-tokens.md）
SPECIAL_TOKENS = [
    "<unk>",            # ID=0  未知词
    "<s>",              # ID=1  序列开始
    "</s>",             # ID=2  序列结束
    "<|im_start|>",     # ID=3  对话消息开始（ChatML）
    "<|im_end|>",       # ID=4  对话消息结束（ChatML）
]


# ============================================================
# 准备示例语料（如果没有自己的语料，用这个演示）
# ============================================================

SAMPLE_CORPUS = """
鸡你太美是一首非常经典的歌曲，由蔡徐坤演唱。
练习时长两年半的偶像练习生，喜欢唱、跳、rap、篮球。
你干嘛哎哟，这是一句经典的台词。

鸡你太美baby鸡你太美，music～
全民制作人们，大家好，我是练习生蔡徐坤。
只因你太美，baby只因你太美。

AI 大模型是近年来最热门的技术方向。
Transformer 架构在 2017 年被提出，从此改变了自然语言处理领域。
BERT 和 GPT 是两种基于 Transformer 的预训练模型。
GPT 采用自回归方式，通过预测下一个 token 来训练。

大语言模型的训练流程包括：分词器训练、预训练、SFT微调、RLHF对齐。
分词器是大模型的基础组件，负责将文本转换为 token 序列。
BPE（Byte Pair Encoding）是最常用的子词分词算法。
词表大小（vocab_size）需要根据模型大小来选择。

The quick brown fox jumps over the lazy dog.
Transformer is a neural network architecture based on self-attention mechanism.
Large language models can understand and generate human-like text.
Byte Pair Encoding is a subword tokenization algorithm.

def hello_world():
    print("Hello, World!")
    return True

for i in range(10):
    print(f"练习第{i+1}天")

class IkunModel:
    def __init__(self, vocab_size=6400):
        self.vocab_size = vocab_size
    
    def forward(self, x):
        return self.embedding(x)

深度学习的核心是神经网络，通过多层非线性变换来学习数据的表示。
卷积神经网络（CNN）擅长处理图像数据。
循环神经网络（RNN）擅长处理序列数据，但存在长距离依赖问题。
注意力机制（Attention）解决了 RNN 的长距离依赖问题。

自然语言处理（NLP）是人工智能的一个重要分支。
机器翻译、文本摘要、问答系统都属于 NLP 任务。
预训练语言模型通过在大规模语料上无监督训练来获取语言知识。
微调（Fine-tuning）是在预训练模型的基础上针对特定任务进行训练。

强化学习从人类反馈中学习（RLHF）是让大模型更好地对齐人类偏好的方法。
DPO（Direct Preference Optimization）是 RLHF 的一种替代方案。
GRPO（Group Relative Policy Optimization）是 DeepSeek 提出的强化学习方法。

中文是一种表意文字，每个字都有独立的含义。
英文是一种拼音文字，由字母组合成单词。
分词器需要同时处理中文和英文，以及其他语言。
ByteLevel BPE 通过将所有文字转换为字节来统一处理。

篮球是一项团队运动，需要团队合作和个人技术。
唱歌需要良好的音准和节奏感。
跳舞需要身体的协调性和表现力。
Rap 需要快速的语言组织能力和节奏感。
"""


def prepare_corpus():
    """准备训练语料（如果没有语料文件，创建示例语料）"""
    os.makedirs(CORPUS_DIR, exist_ok=True)
    corpus_file = os.path.join(CORPUS_DIR, "sample_corpus.txt")
    
    if not os.path.exists(corpus_file):
        print(f"未找到语料文件，创建示例语料: {corpus_file}")
        with open(corpus_file, "w", encoding="utf-8") as f:
            # 重复语料以增加训练数据量
            for _ in range(50):
                f.write(SAMPLE_CORPUS)
        print(f"示例语料已创建（{os.path.getsize(corpus_file)} 字节）")
    
    # 收集所有 .txt 文件
    corpus_files = []
    for fname in os.listdir(CORPUS_DIR):
        if fname.endswith(".txt"):
            corpus_files.append(os.path.join(CORPUS_DIR, fname))
    
    print(f"找到 {len(corpus_files)} 个语料文件")
    return corpus_files


def train_tokenizer(corpus_files):
    """训练 BPE 分词器"""
    print("\n" + "=" * 60)
    print("🐔 开始训练 ikun 分词器")
    print("=" * 60)
    
    # Step 1: 创建 BPE 分词器
    print("\n📦 Step 1: 创建 BPE 分词器...")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    
    # Step 2: 设置预分词器和解码器
    print("🔧 Step 2: 配置 ByteLevel 预分词器...")
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    
    # Step 3: 配置训练器
    print(f"⚙️  Step 3: 配置训练器 (vocab_size={VOCAB_SIZE})...")
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    
    # Step 4: 开始训练
    print("🚀 Step 4: 开始训练...")
    tokenizer.train(corpus_files, trainer)
    
    print(f"\n✅ 训练完成！词表大小: {tokenizer.get_vocab_size()}")
    
    return tokenizer


def test_tokenizer(tokenizer):
    """测试分词器效果"""
    print("\n" + "=" * 60)
    print("🧪 测试分词器效果")
    print("=" * 60)
    
    test_sentences = [
        "鸡你太美",
        "练习时长两年半",
        "你干嘛哎哟",
        "Hello World",
        "Transformer is all you need",
        "vocab_size=6400",
        "鸡你太美baby鸡你太美",
        "<|im_start|>user\n你好<|im_end|>",
    ]
    
    for text in test_sentences:
        encoded = tokenizer.encode(text)
        print(f"\n{'─' * 50}")
        print(f"原文：{text}")
        print(f"Token 数：{len(encoded.tokens)}")
        print(f"Tokens：{encoded.tokens}")
        print(f"IDs：{encoded.ids}")
        
        # 测试解码
        decoded = tokenizer.decode(encoded.ids)
        print(f"解码：{decoded}")
        
    print(f"\n{'─' * 50}")


def save_tokenizer(tokenizer, output_file):
    """保存分词器"""
    tokenizer.save(output_file)
    file_size = os.path.getsize(output_file)
    print(f"\n💾 分词器已保存到: {output_file} ({file_size:,} 字节)")


def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║         🐔 ikun-tokenizer 训练脚本                  ║")
    print("║         练习时长两年半的分词器                        ║")
    print("╚══════════════════════════════════════════════════════╝")
    
    # 1. 准备语料
    print("\n📚 准备语料...")
    corpus_files = prepare_corpus()
    
    # 2. 训练分词器
    tokenizer = train_tokenizer(corpus_files)
    
    # 3. 测试效果
    test_tokenizer(tokenizer)
    
    # 4. 保存
    save_tokenizer(tokenizer, OUTPUT_FILE)
    
    print("\n🎉 全部完成！鸡你太美！")
    print("\n下一步：")
    print("  1. 运行 python tokenizer_demo.py 查看更多演示")
    print("  2. 替换 data/ 目录下的语料，重新训练更好的分词器")
    print("  3. 去 ikun-pretrain 学习如何用这个分词器训练模型")


if __name__ == "__main__":
    main()
