# datasets.py 依赖关系和使用关系说明

## 问题

请梳理 `src/datasets.py` 文件里各个类、函数的依赖关系和使用关系，帮助理解这段代码。

## 简短答案

可以把 `datasets.py` 理解成三层结构：

```text
工具层
  -> Dataset 层
      -> Collator 层
          -> 训练脚本使用
```

一句话总结：

```text
ItemRecord 负责表示 item
Dataset 负责构造单条训练样本
QwenVLCollator 负责把多条样本合成模型 batch
train_caption.py / train_align.py 负责真正训练
```

## 整体依赖关系

```text
ItemRecord
  <- load_items()
  <- build_item_prompt()

read_table()
  -> load_items()
  -> AlignmentPairDataset.__init__()

load_items()
  -> CaptionDataset.__init__()
  -> AlignmentPairDataset.__init__()

build_item_prompt()
  -> CaptionDataset.__getitem__()
  -> AlignmentPairDataset._encode_item_side()

build_query_prompt()
  -> AlignmentPairDataset._encode_query_side()

load_images()
  -> CaptionDataset.__getitem__()
  -> AlignmentPairDataset._encode_item_side()

CaptionDataset
  -> QwenVLCollator.caption_collate()
  -> train_caption.py

AlignmentPairDataset
  -> QwenVLCollator.align_collate()
  -> train_align.py

QwenVLCollator
  -> processor.apply_chat_template()
  -> processor(...)
  -> 输出模型可用 batch
```

## 各个类和函数的角色

### `ItemRecord`

`ItemRecord` 表示一个 item 的结构化数据。它对应 `items.csv` 里的一行，包括：

```text
item_id
frame_paths
title
asr
ocr
description
caption_text
```

它是底层数据结构，不直接训练模型。

### `read_table()`

`read_table()` 是统一读表入口，负责读取 `.csv` 或 `.parquet` 文件。

它被这些地方使用：

```text
load_items()
AlignmentPairDataset.__init__()
```

这样后续如果从 CSV 切换到 parquet，Dataset 主逻辑不用大改。

### `load_items()`

`load_items()` 会读取 `items.csv`，然后返回：

```python
dict[str, ItemRecord]
```

也就是：

```text
item_id -> ItemRecord
```

它被两个 Dataset 使用：

```text
CaptionDataset
AlignmentPairDataset
```

### `build_item_prompt()`

`build_item_prompt()` 把一个 `ItemRecord` 转成 item 侧 prompt。

```text
ItemRecord -> item 文本 prompt
```

它被这些地方使用：

```text
CaptionDataset.__getitem__()
AlignmentPairDataset._encode_item_side()
```

### `build_query_prompt()`

`build_query_prompt()` 把 query 字符串转成 query 侧 prompt。

```text
query string -> query 文本 prompt
```

它被这个地方使用：

```text
AlignmentPairDataset._encode_query_side()
```

### `load_images()`

`load_images()` 根据 `frame_paths` 加载视频帧图片。

```text
frame_paths -> list[PIL.Image]
```

它被这些地方使用：

```text
CaptionDataset.__getitem__()
AlignmentPairDataset._encode_item_side()
```

## Dataset 层

### `CaptionDataset`

`CaptionDataset` 用于 `Stage I-A` caption 训练。

它的输入是：

```text
items.csv
```

它的输出样本是：

```python
{
    "item_id": ...,
    "prompt": ...,
    "caption": ...,
    "images": ...
}
```

也就是：

```text
item 图片 + item 元数据 prompt -> caption
```

它服务于：

```text
train_caption.py
```

### `AlignmentPairDataset`

`AlignmentPairDataset` 用于 `Stage I-B` collaborative alignment 训练。

它的输入是：

```text
items.csv
query_item_pairs.csv
item_item_pairs.csv
```

它会把两种 pair 统一成：

```python
{
    "pair_type": ...,
    "a": ...,
    "b": ...,
    "weight": ...
}
```

其中：

```text
query-item pair:
  a = query
  b = item

item-item pair:
  a = item
  b = item
```

它服务于：

```text
train_align.py
```

训练时会把 `a` 和 `b` 分别编码成 embedding：

```python
h_a = model.encode(a_batch)
h_b = model.encode(b_batch)
```

然后用 `info_nce_loss(h_a, h_b)` 做对比学习。

## Collator 层

### `QwenVLCollator`

`QwenVLCollator` 是连接 Dataset 和模型的桥。

Dataset 返回的是普通 Python dict；模型需要的是 tensor batch。`QwenVLCollator` 负责把样本变成：

```text
input_ids
attention_mask
image tensor
labels
```

它有两个主要入口：

```python
caption_collate()
align_collate()
```

`caption_collate()` 用于 caption 训练，会带上 `labels`。

`align_collate()` 用于 alignment 训练，会分别构造 `a` 侧 batch 和 `b` 侧 batch。

## 两条核心数据流

### Stage I-A

```text
items.csv
  -> load_items()
  -> CaptionDataset
  -> QwenVLCollator.caption_collate()
  -> train_caption.py
  -> caption_cross_entropy
```

### Stage I-B

```text
items.csv + query_item_pairs.csv + item_item_pairs.csv
  -> AlignmentPairDataset
  -> QwenVLCollator.align_collate()
  -> train_align.py
  -> model.encode(a)
  -> model.encode(b)
  -> info_nce_loss
```
