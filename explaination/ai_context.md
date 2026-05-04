# AI Context

这个文件用于保存本项目与 AI 助手协作时形成的上下文。下次重新打开项目时，可以先让 AI 阅读本文档，再继续讨论代码或论文复现细节。

## 项目目标

本项目是对论文 **Farewell to Item IDs: Unlocking the Scaling Potential of Large Ranking Models via Semantic Tokens** 中 Stage I 的 PyTorch 复现骨架。

核心目标包括：

- `Stage I-A`: 用 `caption loss` 把 Qwen2.5-VL 适配到短视频/推荐领域。
- `Stage I-B`: 用 `collaborative alignment loss` 把 query-item 和 item-item 协同信号注入同一个多模态表征空间。
- `export_emb.py`: 导出 item dense embedding，供后续 Stage II 或召回/排序系统使用。

## 已讨论过的问题

### Python class 里的下划线命名规则

- `method()`: 公开方法，是类对外提供的正常 API。
- `_method()`: 内部 helper，约定外部代码不要直接依赖。
- `__method()`: 触发 name mangling，主要用于避免子类意外覆盖父类内部实现。
- `__method__()`: Python 特殊协议方法，例如 `__init__`、`__len__`、`__getitem__`。

在本项目中：

- `CaptionDataset.__len__()`、`CaptionDataset.__getitem__()`、`AlignmentPairDataset.__len__()`、`AlignmentPairDataset.__getitem__()` 是 PyTorch Dataset 协议方法。
- `AlignmentPairDataset._encode_item_side()`、`AlignmentPairDataset._encode_query_side()`、`QwenVLCollator._message()`、`QwenVLCollator._batch_encode()` 是类内部 helper。
- `QwenVLCollator.caption_collate()`、`QwenVLCollator.align_collate()` 是供 `DataLoader` 使用的公开 collate 方法。

### 是否有本地更新并推送远端

曾检查远端 Git 信息：

```text
origin: https://github.com/chenxiwcx/TRMReproduce.git
branch: main
```

发现 `explaination/dataset.md` 有文档更新，并已提交推送到远端 `main`：

```text
cc31cb9 docs: expand dataset explanation
```

该提交补充了 `_batch_encode()`、`processor.apply_chat_template()`、`labels`、`pixel_values`、`image_grid_thw` 等概念说明。

## 已理解的代码主线

整体数据流：

```text
CSV 数据
  -> Dataset 构造样本
  -> QwenVLCollator 构造 Qwen2.5-VL batch
  -> TRMRepresentationModel 生成 logits 或 embedding
  -> losses.py 计算 caption CE 或 InfoNCE
  -> train_caption.py / train_align.py 训练
  -> export_emb.py 导出 item embedding
```

核心文件职责：

- `src/datasets.py`: 读取 CSV/parquet，构造 `CaptionDataset`、`AlignmentPairDataset`，并通过 `QwenVLCollator` 生成 Qwen2.5-VL 输入。
- `src/model.py`: 加载 Qwen2.5-VL backbone，添加 LoRA，冻结视觉塔，并通过 mean pooling + projection 得到 dense representation `h`。
- `src/losses.py`: 实现 caption cross entropy、InfoNCE、双向 InfoNCE 和 Stage I-B 的组合 loss。
- `src/train_caption.py`: Stage I-A 训练入口，用 caption loss 做领域适配。
- `src/train_align.py`: Stage I-B 训练入口，用 query-item 和 item-item pair 做 collaborative alignment，并可加入 caption anchor。
- `src/export_emb.py`: 加载训练后的 adapter，遍历 item，导出 embedding 表。

## 后续协作建议

下次继续使用 AI 协助本项目时，可以先说：

```text
请先阅读 explaination/ai_context.md 和 README.md，然后继续协助我理解或修改这个项目。
```

如果之后又讨论了重要问题、修改了设计或推送了关键提交，建议继续更新本文件。
