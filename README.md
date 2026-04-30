# TRMReproduce

这个项目是对论文 **Farewell to Item IDs: Unlocking the Scaling Potential of Large Ranking Models via Semantic Tokens** 中 Stage I 的 PyTorch 复现骨架。

重点覆盖两件事：

1. `Stage I-A`：用 `caption loss` 把 Qwen2.5-VL 适配到短视频/推荐领域。
2. `Stage I-B`：用 `collaborative alignment loss` 把 query-item 和 item-item 协同信号注入同一个多模态表征空间。

## 目录

```text
TRMReproduce/
  data/
    items.csv              # item 多模态元数据样例；生产环境可替换为 parquet
    query_item_pairs.csv   # query-item 正反馈 pair 样例
    item_item_pairs.csv    # item-item co-click pair 样例
  src/
    datasets.py            # 数据读取、样本构造、batch collate
    model.py               # Qwen2.5-VL wrapper、backbone 解释、pooling/projection
    losses.py              # caption CE 和 InfoNCE
    train_caption.py       # Stage I-A 训练入口
    train_align.py         # Stage I-B 训练入口
    export_emb.py          # 导出 item dense embedding
  requirements.txt
```

## Qwen2.5-VL-7B Backbone 摘要

以下参数来自 `Qwen/Qwen2.5-VL-7B-Instruct` 的公开配置，代码里也写了注释：

- 语言模型：decoder-only Transformer，`28` 层，hidden size `3584`，attention heads `28`，KV heads `4`，intermediate size `18944`，词表约 `152064`。
- 视觉编码器：ViT-like encoder，`32` 层，hidden size `1280`，attention heads `16`，patch size `14`，temporal patch size `2`，window size `112`。
- 多模态连接：视觉侧输出通过 merger/projector 接到语言 hidden size `3584`，配置中 `out_hidden_size=3584`，`spatial_merge_size=2`。
- 工程风险：视频帧数、分辨率和视觉 token 上限会显著影响显存与吞吐，所以训练脚本提供了 `--max_pixels` 控制图像 token 数。

## 安装

```bash
cd "/Users/wangchenxi/Downloads/Farewell to Item IDs/TRMReproduce"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

这些命令的含义如下：

- `cd ...`：进入当前项目目录。
- `python -m venv .venv`：在项目中创建一个名为 `.venv` 的 Python 虚拟环境，用来隔离本项目依赖。
- `source .venv/bin/activate`：激活虚拟环境，后续安装和运行都会优先使用 `.venv` 里的 Python 环境。
- `pip install -r requirements.txt`：按照 `requirements.txt` 中列出的包名和版本安装项目依赖。

## 代码阅读顺序

建议按“先建立整体概念，再追数据流，最后看训练细节”的顺序阅读。

1. 先读 `README.md`

   先建立项目整体地图：本项目复现论文 Stage I，分成 `Stage I-A` caption 适配和 `Stage I-B` collaborative alignment 两个阶段。

2. 再看 `data/` 下的三个样例数据文件

   - `data/items.csv`：每个短视频 item 的多模态元数据，包括标题、ASR、OCR、描述和 caption。
   - `data/query_item_pairs.csv`：用户 query 与 item 的正反馈关系。
   - `data/item_item_pairs.csv`：item 与 item 之间的 co-click / 共现关系。

   这一步的目标是理解训练数据的原始形态。

3. 然后读 `src/datasets.py`

   重点看 `ItemRecord`、`load_items()`、`build_item_prompt()`、`build_query_prompt()`、`CaptionDataset`、`AlignmentPairDataset` 和 `QwenVLCollator`。这个文件解释了一行 CSV 如何变成模型可以接收的 prompt、图片输入和 batch。

4. 接着读 `src/model.py`

   重点看 `ModelConfig`、`load_processor()`、`build_backbone()` 和 `TRMRepresentationModel.encode()`。这个文件解释如何加载 Qwen2.5-VL、加 LoRA、启用 gradient checkpointing、冻结视觉塔，并从 hidden states 中抽取最终 dense embedding `h`。

5. 再读 `src/losses.py`

   重点看 `caption_cross_entropy()`、`info_nce_loss()` 和 `weighted_sum_loss()`。这里对应 Stage I-A 的 caption loss，以及 Stage I-B 的 query-item / item-item 对比学习损失。

6. 然后读 `src/train_caption.py`

   这是 `Stage I-A` 的训练入口。阅读主线是：`CaptionDataset` -> `QwenVLCollator` -> `TRMRepresentationModel` -> `caption_cross_entropy()` -> 保存 LoRA adapter 和 projection head。

7. 再读 `src/train_align.py`

   这是 `Stage I-B` 的训练入口。阅读主线是：`AlignmentPairDataset` -> 分别编码 pair 两侧得到 `h_a` 和 `h_b` -> `info_nce_loss()` -> 可选 caption anchor -> 保存对齐后的 adapter 和 projection head。

8. 最后读 `src/export_emb.py`

   这个文件展示训练完成后如何加载 adapter，遍历 item，调用 `model.encode()`，并导出每个 item 的 embedding，供后续 Stage II 或召回/排序系统使用。

第一轮阅读时建议抓住这条主线：

```text
CSV 数据
  -> Dataset 构造样本
  -> Collator 构造 Qwen2.5-VL 输入
  -> Model 生成 logits 或 embedding
  -> Loss 计算训练目标
  -> 训练脚本更新 LoRA/projection head
  -> export_emb 导出 item embedding
```

读完第一轮后，可以回头重点研究三个核心位置：`QwenVLCollator`、`TRMRepresentationModel.encode()` 和 `info_nce_loss()`。

## 训练示例

小样例主要用于看清数据格式和代码路径，不适合得到有效模型。

```bash
python -m src.train_caption \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --items_path data/items.csv \
  --output_dir outputs/caption_lora \
  --epochs 1 \
  --batch_size 1
```

```bash
python -m src.train_align \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --items_path data/items.csv \
  --query_item_path data/query_item_pairs.csv \
  --item_item_path data/item_item_pairs.csv \
  --caption_adapter outputs/caption_lora \
  --output_dir outputs/align_lora \
  --epochs 1 \
  --batch_size 2
```

```bash
python -m src.export_emb \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --items_path data/items.csv \
  --adapter_dir outputs/align_lora \
  --output_path outputs/item_embeddings.parquet
```

## 生产建议

- Stage I-A：`1 epoch` 起步，LoRA LR `1e-4`，全参 LR `1e-5`，bf16，gradient checkpointing。
- Stage I-B：`1~2 epoch`，LoRA LR `5e-5~1e-4`，projection head LR `1e-4~3e-4`，temperature `0.07`。
- global batch：建议 `2048~8192 pairs`，in-batch negatives 越多越好。
- 资源估算：`1e8` pair，`32 x H100 80GB` 训练 `1 epoch` 约 `2~4` 天；`2 epoch` 加验证约 `5~9` 天。
