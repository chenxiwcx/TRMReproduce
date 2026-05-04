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

## 补充问答压缩版

### `_batch_encode()` 的作用是什么？

`_batch_encode()` 负责把一组样本转换成 Qwen2.5-VL 可以直接使用的 tensor batch。

整体流程是：

```text
examples
  -> _message()
  -> processor.apply_chat_template()
  -> processor(...)
  -> batch
```

如果 `with_labels=True`，说明当前用于 caption 训练，会额外生成 `labels`。

如果 `with_labels=False`，说明当前只用于 embedding 抽取，不需要生成训练标签。

### `processor.apply_chat_template()` 做什么？

它把结构化 messages 转成 Qwen2.5-VL 熟悉的对话文本格式。

代码里设置 `tokenize=False`，表示这里只生成字符串，真正的 tokenization 交给后面的 `processor(...)`。

`add_generation_prompt=not with_labels` 的含义是：

```text
with_labels=True:
  caption 答案已经在 assistant message 里，不再额外加生成提示。

with_labels=False:
  只给 user 输入，需要加 assistant 开始生成的位置，方便模型按对话格式编码。
```

### `processor(...)` 的输入参数是什么意思？

```python
batch = self.processor(
    text=texts,
    images=image_inputs,
    padding=True,
    return_tensors="pt",
    max_pixels=self.max_pixels,
)
```

各参数含义：

```text
text:
  一组已经套好 chat template 的文本。

images:
  每条样本对应的图片列表；query 没有图片时传 None。

padding=True:
  把不同长度的文本补齐到同一长度，方便组成 batch tensor。

return_tensors="pt":
  返回 PyTorch tensor。

max_pixels:
  限制图片最大像素数，间接控制视觉 token 数和显存开销。
```

`processor` 可以理解成：

```text
tokenizer + image processor
```

它会同时处理文本和图片，输出如 `input_ids`、`attention_mask`、`pixel_values`、`image_grid_thw` 等字段。

### `labels` 是什么？

`labels` 是语言模型训练时的目标 token id，用来计算 caption loss。

当前代码简化为：

```python
batch["labels"] = batch["input_ids"].clone()
```

这表示所有 token 都参与 next-token prediction loss。

不过生产训练里更合理的做法是：

```text
user prompt 部分 -> labels 设为 -100，不计算 loss
padding 部分 -> labels 设为 -100，不计算 loss
assistant caption 部分 -> 保留 token id，计算 loss
```

因为 `caption_cross_entropy()` 里使用了 `ignore_index=-100`，所以 `labels == -100` 的位置不会参与 loss。

### `pixel_values` 和 `image_grid_thw` 是什么？

`pixel_values` 是图片经过 processor 预处理后的视觉张量，表示图片内容本身。

```text
原始 PIL 图片
  -> resize / normalize / 视觉预处理
  -> pixel_values
  -> 视觉 encoder
```

`image_grid_thw` 描述视觉 token 的时空网格形状：

```text
T = temporal，时间维度
H = height，高度方向 patch 数
W = width，宽度方向 patch 数
```

对于单张图片，`T` 通常可以理解为 1。对于多帧视频，`T` 表示时间维度上的帧/patch 结构。

二者区别：

```text
pixel_values:
  视觉内容本身。

image_grid_thw:
  视觉 token 的时间和空间排列结构。
```

