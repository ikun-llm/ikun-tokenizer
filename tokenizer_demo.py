"""
ikun-tokenizer: 分词效果演示脚本
=================================

加载训练好的分词器，展示各种分词效果。
需要先运行 train_tokenizer.py 训练分词器。

使用方法：
    python tokenizer_demo.py
"""

import os
from tokenizers import Tokenizer


TOKENIZER_FILE = "ikun_tokenizer.json"


def load_tokenizer():
    """加载训练好的分词器"""
    if not os.path.exists(TOKENIZER_FILE):
        print(f"❌ 未找到分词器文件: {TOKENIZER_FILE}")
        print("   请先运行: python train_tokenizer.py")
        return None
    
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
    print(f"✅ 分词器已加载 (词表大小: {tokenizer.get_vocab_size()})")
    return tokenizer


def demo_basic_tokenization(tokenizer):
    """演示基础分词"""
    print("\n" + "=" * 60)
    print("📝 演示 1：基础分词")
    print("=" * 60)
    
    sentences = [
        "鸡你太美",
        "你干嘛哎哟",
        "练习时长两年半的偶像练习生",
        "唱跳rap篮球",
        "Hello World",
        "I love basketball",
    ]
    
    for text in sentences:
        encoded = tokenizer.encode(text)
        print(f"\n原文：{text}")
        print(f"  → Tokens: {encoded.tokens}")
        print(f"  → IDs:    {encoded.ids}")
        print(f"  → 共 {len(encoded.tokens)} 个 token")


def demo_encode_decode(tokenizer):
    """演示编码和解码"""
    print("\n" + "=" * 60)
    print("🔄 演示 2：编码 → 解码（还原）")
    print("=" * 60)
    
    sentences = [
        "鸡你太美baby鸡你太美",
        "Transformer is all you need",
        "大语言模型的训练需要大量数据",
    ]
    
    for text in sentences:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        match = "✅" if decoded.strip() == text else "⚠️"
        print(f"\n原文：  {text}")
        print(f"编码：  {encoded.ids}")
        print(f"解码：  {decoded}")
        print(f"还原：  {match}")


def demo_token_count_comparison(tokenizer):
    """对比不同文本的 token 数量"""
    print("\n" + "=" * 60)
    print("📊 演示 3：Token 数量对比")
    print("=" * 60)
    
    texts = [
        ("中文短句", "你好"),
        ("中文长句", "练习时长两年半的偶像练习生蔡徐坤喜欢唱跳rap篮球"),
        ("英文短句", "Hello"),
        ("英文长句", "The quick brown fox jumps over the lazy dog"),
        ("混合文本", "ikun-2.5B是一个练习时长两年半的AI模型"),
        ("代码片段", "def forward(self, x): return self.embedding(x)"),
        ("纯数字", "1234567890"),
        ("特殊字符", "!@#$%^&*()"),
    ]
    
    print(f"\n{'类型':<12} {'字符数':>6} {'Token数':>8} {'压缩比':>8}")
    print("─" * 40)
    
    for label, text in texts:
        encoded = tokenizer.encode(text)
        char_count = len(text)
        token_count = len(encoded.tokens)
        ratio = char_count / token_count if token_count > 0 else 0
        print(f"{label:<12} {char_count:>6} {token_count:>8} {ratio:>8.2f}")


def demo_special_tokens(tokenizer):
    """演示特殊 token 的处理"""
    print("\n" + "=" * 60)
    print("⭐ 演示 4：特殊 Token 处理")
    print("=" * 60)
    
    # ChatML 格式的对话
    conversation = """<|im_start|>system
你是 ikun-2.5B，一个练习时长两年半的 AI 练习生。
<|im_end|>
<|im_start|>user
你会什么才艺？
<|im_end|>
<|im_start|>assistant
唱、跳、rap、篮球！
<|im_end|>"""
    
    print(f"\n对话原文：")
    print(conversation)
    
    encoded = tokenizer.encode(conversation)
    print(f"\n分词结果 ({len(encoded.tokens)} 个 token)：")
    
    for i, (token, tid) in enumerate(zip(encoded.tokens, encoded.ids)):
        # 高亮特殊 token
        if token in ["<|im_start|>", "<|im_end|>", "<s>", "</s>", "<unk>"]:
            marker = "⭐"
        else:
            marker = "  "
        print(f"  {marker} [{i:3d}] ID={tid:5d}  token={repr(token)}")


def demo_vocab_info(tokenizer):
    """展示词表信息"""
    print("\n" + "=" * 60)
    print("📖 演示 5：词表信息")
    print("=" * 60)
    
    vocab = tokenizer.get_vocab()
    vocab_size = tokenizer.get_vocab_size()
    
    print(f"\n词表总大小: {vocab_size}")
    
    # 展示部分词表内容
    # 按 ID 排序
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    # 特殊 token
    print("\n特殊 Token:")
    for token, tid in sorted_vocab[:10]:
        print(f"  ID={tid:5d}  {repr(token)}")
    
    # 查找一些有趣的 token
    print("\n有趣的 Token:")
    interesting = ["鸡", "你", "太", "美", "练习", "篮球", "hello", "world"]
    for word in interesting:
        if word in vocab:
            print(f"  ID={vocab[word]:5d}  {repr(word)}")
        else:
            # 可能编码方式不同，尝试编码后查看
            encoded = tokenizer.encode(word)
            print(f"  {repr(word)} → tokens: {encoded.tokens} (IDs: {encoded.ids})")


def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║         🐔 ikun-tokenizer 演示脚本                  ║")
    print("║         看看分词器如何拆解"鸡你太美"                  ║")
    print("╚══════════════════════════════════════════════════════╝")
    
    # 加载分词器
    tokenizer = load_tokenizer()
    if tokenizer is None:
        return
    
    # 运行各个演示
    demo_basic_tokenization(tokenizer)
    demo_encode_decode(tokenizer)
    demo_token_count_comparison(tokenizer)
    demo_special_tokens(tokenizer)
    demo_vocab_info(tokenizer)
    
    print("\n" + "=" * 60)
    print("🎉 演示完成！鸡你太美！")
    print("=" * 60)
    print("\n想了解更多？请阅读 docs/ 目录下的文档：")
    print("  📖 docs/01-bpe.md           BPE 分词算法")
    print("  📖 docs/02-vocab.md         词表大小设计")
    print("  📖 docs/03-special-tokens.md 特殊 Token")
    print("  📖 docs/04-train-tokenizer.md 训练流程")


if __name__ == "__main__":
    main()
