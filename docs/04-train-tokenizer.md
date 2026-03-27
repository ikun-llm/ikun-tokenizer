# 第四章：训练分词器

> 从零开始训练一个 ikun 专属分词器

![训练流程图示](images/04_train_tokenizer.png)

---

## 一句话版本

训练分词器就是：拿一堆文本语料，让 BPE 算法从中学习合并规则，最终得到一个"词表"和对应的"合并规则表"。整个过程不需要 GPU，CPU 就能搞定。

---

## 训练流程总览

```
整个过程只有 5 步：

  ┌─────────────────────────────┐
  │  Step 1: 准备语料           │
  │  （收集训练文本数据）         │
  └──────────────┬──────────────┘
                 ↓
  ┌─────────────────────────────┐
  │  Step 2: 配置分词器         │
  │  （选择算法、设置参数）       │
  └──────────────┬──────────────┘
                 ↓
  ┌─────────────────────────────┐
  │  Step 3: 训练！             │
  │  （BPE 合并，生成词表）      │
  └──────────────┬──────────────┘
                 ↓
  ┌─────────────────────────────┐
  │  Step 4: 添加特殊 Token     │
  │  （im_start、im_end 等）     │
  └──────────────┬──────────────┘
                 ↓
  ┌─────────────────────────────┐
  │  Step 5: 测试 & 保存        │
  │  （验证分词效果，导出文件）    │
  └─────────────────────────────┘

  不需要 GPU！纯 CPU 运算！
  训练时间取决于语料大小，几分钟到几小时不等
```

---

## Step 1: 准备语料

```
语料 = 训练分词器的原始文本数据

语料的质量决定了分词器的质量：
  好的语料 → 分词器能认识常用词，分词合理
  差的语料 → 分词器乱拆，"鸡你太美"可能被拆成"鸡你太"+"美"

ikun 分词器的语料来源：
  1. 中文 wiki 百科
  2. 中文小说、新闻
  3. 英文文本
  4. 代码
  5. 当然还有 ikun 相关语料（确信）

语料格式很简单——纯文本文件：
  文件1.txt:
    "练习时长两年半的偶像练习生..."
    "鸡你太美是一首经典歌曲..."
  
  文件2.txt:
    "Transformer is a neural network architecture..."
```

---

## Step 2: 配置分词器

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# 创建一个空的 BPE 分词器
tokenizer = Tokenizer(BPE(unk_token="<unk>"))

# 设置预分词器（ByteLevel）
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

# 设置解码器
tokenizer.decoder = ByteLevelDecoder()

# 配置训练器
trainer = BpeTrainer(
    vocab_size=6400,              # 词表大小
    min_frequency=2,              # 最低出现频率（至少出现2次才合并）
    special_tokens=[              # 特殊 token
        "<unk>",                  # ID=0 未知词
        "<s>",                    # ID=1 序列开始
        "</s>",                   # ID=2 序列结束
        "<|im_start|>",           # ID=3 对话开始
        "<|im_end|>",             # ID=4 对话结束
    ],
    show_progress=True,           # 显示进度条
)
```

### 关键参数解读

```
vocab_size=6400
  词表最终大小
  BPE 会合并到词表达到这个大小为止
  前面第二章讲过为什么选 6400

min_frequency=2
  字节对至少出现 2 次才会被合并
  = 1：所有出现过的对都可能合并（词表可能包含罕见组合）
  = 2：只合并出现 2 次以上的对（更稳健）
  = 5：更保守，但可能合并不够

special_tokens=[...]
  特殊 token 的顺序决定了它们的 ID
  第一个 = ID 0，第二个 = ID 1，以此类推
  训练时这些 token 会被预留，不参与 BPE 合并

ByteLevel pre_tokenizer
  先把文本转成字节再做 BPE
  这样可以处理任何语言和符号
```

---

## Step 3: 开始训练

```python
# 指定训练语料文件
corpus_files = [
    "data/corpus_zh.txt",    # 中文语料
    "data/corpus_en.txt",    # 英文语料
    "data/corpus_code.txt",  # 代码语料
]

# 开始训练！
tokenizer.train(corpus_files, trainer)

print(f"词表大小: {tokenizer.get_vocab_size()}")
# 词表大小: 6400
```

```
训练过程中发生了什么？

  ┌─────────────────────────────────────────┐
  │ 1. 读取所有语料文件                      │
  │ 2. ByteLevel 预分词                      │
  │ 3. 统计初始字节频率                       │
  │ 4. 开始 BPE 合并循环：                    │
  │    ├─ 统计所有相邻字节对频率              │
  │    ├─ 找到最高频的对                      │
  │    ├─ 合并它，加入词表                    │
  │    ├─ 词表大小 +1                        │
  │    └─ 重复，直到 vocab_size = 6400       │
  │ 5. 训练完成！                            │
  └─────────────────────────────────────────┘

  注意：5 个特殊 token 已经占了 ID 0-4
  所以实际 BPE 合并产生的 token 数 = 6400 - 5 = 6395
  加上 256 个初始字节 token
  实际合并次数 = 6400 - 5 - 256 = 6139 次
```

---

## Step 4: 测试分词效果

```python
# 测试几个句子
test_sentences = [
    "鸡你太美",
    "练习时长两年半",
    "你干嘛哎哟",
    "Hello World",
    "Transformer is all you need",
    "<|im_start|>user\n你好<|im_end|>",
]

for text in test_sentences:
    encoded = tokenizer.encode(text)
    print(f"\n原文：{text}")
    print(f"Token数：{len(encoded.tokens)}")
    print(f"Tokens：{encoded.tokens}")
    print(f"IDs：{encoded.ids}")
```

```
预期输出（示意）：

原文：鸡你太美
Token数：2
Tokens：['鸡你', '太美']
IDs：[1024, 2048]

原文：练习时长两年半
Token数：4
Tokens：['练习', '时长', '两年', '半']
IDs：[512, 768, 1536, 3200]

原文：你干嘛哎哟
Token数：3
Tokens：['你', '干嘛', '哎哟']
IDs：[256, 4096, 5120]

原文：Hello World
Token数：2
Tokens：['Hello', 'ĠWorld']
IDs：[3721, 4567]

原文：<|im_start|>user\n你好<|im_end|>
Token数：4
Tokens：['<|im_start|>', 'user', '\n你好', '<|im_end|>']
IDs：[3, 1567, 890, 4]
```

---

## Step 5: 保存分词器

```python
# 保存为 JSON 文件
tokenizer.save("ikun_tokenizer.json")

# 之后加载使用
loaded_tokenizer = Tokenizer.from_file("ikun_tokenizer.json")

# 验证加载后的分词器
encoded = loaded_tokenizer.encode("鸡你太美")
print(encoded.tokens)  # ['鸡你', '太美']
decoded = loaded_tokenizer.decode(encoded.ids)
print(decoded)  # "鸡你太美"
```

```
保存的 JSON 文件包含什么？

  {
    "model": {
      "type": "BPE",
      "vocab": {           ← 词表：token → ID 的映射
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "Ġ": 32,
        "a": 97,
        ...
        "鸡你": 1024,
        "太美": 2048,
        ...
      },
      "merges": [          ← 合并规则（按顺序）
        "太 美",            ← 第 1 次合并
        "鸡 你",            ← 第 2 次合并
        "鸡你 太美",        ← 第 3 次合并
        ...
      ]
    },
    "pre_tokenizer": { "type": "ByteLevel" },
    "decoder": { "type": "ByteLevel" }
  }

  这就是训练好的分词器的全部！
  一个 vocab（词表）+ 一组 merges（合并规则）
```

---

## 完整训练脚本

项目根目录下的 `train_tokenizer.py` 包含了完整的训练代码，可以直接运行：

```bash
# 安装依赖
pip install tokenizers

# 运行训练
python train_tokenizer.py

# 运行演示
python tokenizer_demo.py
```

---

## 常见问题 FAQ

### Q1：训练分词器需要 GPU 吗？

```
不需要！纯 CPU 运算。

BPE 训练本质上是统计 + 排序 + 合并
不涉及矩阵乘法和反向传播
所以不需要 GPU

普通笔记本就能训练
10MB 的语料大概几秒钟
1GB 的语料大概几分钟
```

### Q2：语料越多越好吗？

```
大体上是的，但要注意：

  1. 语料要有代表性
     如果只用小说语料，分词器就不认识代码
     如果只用英文语料，中文分词就会很差
  
  2. 语料要平衡
     中文:英文:代码 ≈ 7:2:1（示例比例）
     如果某类语料太多，那类的 token 会占满词表
  
  3. 语料太少的问题
     BPE 学不到足够的合并规则
     很多常用词不会被合并
     分词结果会很碎
```

### Q3：训练好的分词器可以修改吗？

```
可以！但要小心。

  ✅ 可以做的事：
    - 添加新的特殊 token
    - 在词表末尾追加新 token

  ❌ 不建议做的事：
    - 删除已有的 token
    - 修改 token 的 ID
    - 修改合并规则的顺序

  为什么？
    如果模型已经用某个分词器训练过
    修改词表会导致 token ID 错乱
    模型就"认不出"之前学过的词了

  比喻：
    你已经按某本字典学了两年半
    突然换了一本字典，页码全变了
    你翻到第 42 页想看"鸡"
    结果变成了"鸭"——完蛋！
```

### Q4：分词器和模型是什么关系？

```
分词器 = 翻译官（文字 ↔ 数字）
模型   = 大脑（处理数字，生成数字）

工作流程：
  用户输入   分词器     模型      分词器     输出
  "鸡你太美" →→→ [1024,2048] →→→ [3200,4096] →→→ "你干嘛哎哟"
            编码(encode)   推理(forward)  解码(decode)

关键：
  训练模型之前，必须先训练好分词器
  分词器确定了模型的"语言"
  模型的 Embedding 层大小 = vocab_size
  换分词器 = 换语言 = 模型需要重新训练
```

---

## 本章小结

| 步骤 | 内容 | 关键点 |
|------|------|--------|
| 准备语料 | 收集训练文本 | 多样性、平衡性 |
| 配置参数 | 设置 vocab_size 等 | 6400 for 小模型 |
| 训练 | BPE 合并循环 | 不需要 GPU |
| 添加特殊 Token | im_start 等 | 不参与合并 |
| 测试保存 | 验证效果，导出 JSON | vocab + merges |

```
恭喜你学完了 ikun-tokenizer 的全部内容！🎉

现在你知道了：
  1. BPE 怎么把文字拆成 token
  2. 为什么 vocab_size=6400
  3. 特殊 token 的作用
  4. 如何从零训练一个分词器

下一步：
  去 ikun-pretrain 学习如何用这个分词器训练模型！
  分词器是基础，模型才是目标！

  练习时长 +0.5 年！距离两年半又近了一步！🏀
```

---

[← 上一章：特殊 Token](03-special-tokens.md) | [回到目录 →](../README.md)
