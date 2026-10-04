# Build Transformer From Zero

从零实现一个 Transformer 语言模型，目前进度：BPE 分词器。

基于 Stanford CS336 Assignment 1 的作业框架与测试用例（见 [LICENSE](./LICENSE)）。

## 目录结构

```
cs336_basics/
├── pretokenization.py   # 语料分块 + 并行预分词
├── train_bpe.py         # BPE 词表训练、合并、词表/合并规则的存储与读取
├── converter.py         # bytes 与可见 unicode 字符的互相转换
├── tokenizer.py         # Tokenizer（encode / decode，进行中）
├── vocab.json           # 训练得到的词表
└── merges.txt           # 训练得到的合并规则
tests/                   # 单元测试，通过 tests/adapters.py 接入实现
```

## 环境

使用 [uv](https://github.com/astral-sh/uv) 管理环境：

```sh
uv run <python_file_path>
```

## 运行测试

```sh
uv run pytest
```

## 数据

```sh
mkdir -p data && cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```
