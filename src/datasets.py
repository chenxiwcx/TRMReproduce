"""TRM Stage I 的数据读取、样本构造和 batch collate。

这个文件是训练链路的第一站，核心职责是把磁盘上的表格数据转换成
Qwen2.5-VL 可以直接处理的 batch。可以按下面这条主线理解：

1. `items.csv` 记录 item 的多模态元数据，例如视频帧、标题、ASR、OCR、描述和 caption。
2. `query_item_pairs.csv` / `item_item_pairs.csv` 记录协同信号中的正样本 pair。
3. `CaptionDataset` 为 Stage I-A 生成 item -> caption 的训练样本。
4. `AlignmentPairDataset` 为 Stage I-B 生成 query-item 或 item-item 的正对样本。
5. `QwenVLCollator` 把 Dataset 返回的 Python dict 统一转换成 processor 输出的 tensor batch。

因此，训练脚本不直接关心 CSV、图片路径或 chat template；它只消费这里产出的 batch。
"""

from __future__ import annotations  # 让类型注解在运行时更轻，减少循环引用问题。

from dataclasses import dataclass  # 用于定义清晰的数据配置对象。
from pathlib import Path  # 统一处理本地路径。
from typing import Any, Literal  # 提供更明确的类型标注。

import pandas as pd  # 读取 csv/parquet，并做轻量表连接。
from PIL import Image  # 真实训练时用于加载 item 视频采样帧。
from torch.utils.data import Dataset  # PyTorch 数据集基类。


# Stage I-B 只支持两种正样本关系：
# - query_item：用户搜索 query 与发生正反馈的 item 构成正对。
# - item_item：两个被协同行为连接的 item 构成正对。
PairType = Literal["query_item", "item_item"]  # pair 类型只允许这两种，避免拼写错误。


@dataclass
class ItemRecord:
    """单个 item 的多模态元数据。

    这里保存的是原始素材和文本字段，不是模型最终学习到的 embedding。
    item_id 只负责把行为日志里的 pair 重新关联回 item 元数据。
    """

    item_id: str  # item 的稳定业务 ID；这里只用于索引原始素材，不进入 ranking 模型。
    frame_paths: list[str]  # 视频采样帧路径列表；真实环境建议离线抽帧并存对象存储。
    title: str  # 标题文本。
    asr: str  # 语音识别文本。
    ocr: str  # OCR 文本。
    description: str  # 作者/系统生成描述。
    caption_text: str  # captioning 阶段的目标文本 T。


def read_table(path: str | Path) -> pd.DataFrame:
    """读取 csv 或 parquet；生产环境推荐 parquet，样例为了可读性使用 csv。

    项目里所有表格读取都走这个函数，所以后续从 csv 切到 parquet 时不用改 Dataset 主逻辑。
    """

    path = Path(path)  # 转成 Path，便于判断后缀。
    if path.suffix.lower() == ".parquet":  # 如果是 parquet，就走 pandas parquet reader。
        return pd.read_parquet(path)  # 需要安装 pyarrow 或 fastparquet。
    return pd.read_csv(path)  # 样例数据是 csv，无需额外二进制依赖。


def load_items(items_path: str | Path) -> dict[str, ItemRecord]:
    """把 items 表加载成 item_id -> ItemRecord 的字典。

    Stage I-B 的 pair 表里只保存 item_id，所以需要先把 item 表做成字典，
    后续才能通过 item_id 快速取回标题、ASR、OCR、caption 和视频帧路径。
    """

    df = read_table(items_path)  # 读取 item 元数据表。
    records: dict[str, ItemRecord] = {}  # 初始化结果字典。
    for row in df.to_dict("records"):  # 按行遍历，便于做字段清洗。
        frame_text = str(row.get("frame_paths", ""))  # frame_paths 用竖线分隔。
        frame_paths = [x for x in frame_text.split("|") if x]  # 过滤空路径。
        record = ItemRecord(  # 构建结构化对象，降低后续代码的字段拼写风险。
            item_id=str(row["item_id"]),
            frame_paths=frame_paths,
            title=str(row.get("title", "")),
            asr=str(row.get("asr", "")),
            ocr=str(row.get("ocr", "")),
            description=str(row.get("description", "")),
            caption_text=str(row.get("caption_text", "")),
        )
        records[record.item_id] = record  # 写入字典。
    return records  # 返回全部 item。


def build_item_prompt(item: ItemRecord) -> str:
    """构造 item 侧 prompt；视觉帧由 collator 单独传给 processor。

    prompt 只放文本元数据，图片不会被拼进字符串，而是以 image content 的形式交给
    Qwen2.5-VL processor。这样能保持多模态输入的官方格式。
    """

    return (  # 明确字段名，能帮助 MLLM 区分 title/asr/ocr 等来源。
        "请根据视频帧和以下元数据理解该短视频。\n"
        f"title: {item.title}\n"
        f"asr: {item.asr}\n"
        f"ocr: {item.ocr}\n"
        f"description: {item.description}\n"
    )


def build_query_prompt(query: str) -> str:
    """构造 query 侧 prompt；query 没有视觉输入，所以走纯文本路径。

    Stage I-B 会把 query 和 item 编码到同一个 embedding 空间里，因此 query 也需要
    包装成一段自然语言 prompt，而不是直接裸传一个字符串。
    """

    return f"用户搜索 query: {query}\n请编码该 query 的推荐意图。"  # 保持短 prompt，降低 token 成本。


def load_images(frame_paths: list[str], data_root: str | Path, max_frames: int = 4) -> list[Image.Image]:
    """加载 item 视频采样帧；样例路径不存在时返回空列表。

    真实训练中这里会返回视频抽帧后的 PIL 图片列表；当前样例仓库没有真实图片，
    所以缺失路径会被跳过，模型输入会自然降级成纯文本样本。
    """

    images: list[Image.Image] = []  # 初始化图片列表。
    root = Path(data_root)  # 数据根目录，通常是 TRMReproduce/data。
    for rel_path in frame_paths[:max_frames]:  # 控制帧数，避免视觉 token 爆炸。
        path = root / rel_path  # 样例中 frame_paths 是相对 data/ 的路径。
        if not path.exists():  # 样例项目没有真实图片，这是正常的。
            continue  # 跳过缺失图片，collator 会降级为纯文本。
        image = Image.open(path).convert("RGB")  # 转 RGB，避免灰度/透明通道导致 processor 报错。
        images.append(image)  # 加入本样本图片列表。
    return images  # 返回 0 到 max_frames 张图。


class CaptionDataset(Dataset):
    """Stage I-A 的 captioning 数据集，每个样本是一个 item。

    返回格式面向 caption 训练：

    ```text
    {
      item_id: 仅用于日志/导出,
      prompt: item 多模态元数据 prompt,
      caption: 模型需要学习生成的目标文本,
      images: item 视频帧图片列表
    }
    ```

    训练脚本 `train_caption.py` 会把这些样本交给 `caption_collate()`，
    最终计算 caption cross entropy。
    """

    def __init__(self, items_path: str | Path, data_root: str | Path | None = None) -> None:
        self.items = list(load_items(items_path).values())  # 加载全部 item。
        self.data_root = Path(data_root or Path(items_path).parent)  # 默认图片路径相对 items 表目录。

    def __len__(self) -> int:
        return len(self.items)  # 返回样本数。

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]  # 取出一个 item。
        return {  # 返回 collator 所需字段。
            "item_id": item.item_id,  # 仅用于日志和导出，不作为模型监督。
            "prompt": build_item_prompt(item),  # 输入 prompt。
            "caption": item.caption_text,  # 目标 caption。
            "images": load_images(item.frame_paths, self.data_root),  # 图片列表；可为空。
        }


class AlignmentPairDataset(Dataset):
    """Stage I-B 的协同对齐 pair 数据集。

    它把两张 pair 表统一成一种训练样本：

    ```text
    {
      pair_type: "query_item" 或 "item_item",
      a: anchor 侧输入,
      b: positive 侧输入,
      weight: 行为强度权重，当前版本只记录，尚未用于 loss
    }
    ```

    对于 query-item pair，a 侧是 query，b 侧是 item。
    对于 item-item pair，a 侧和 b 侧都是 item。
    后续 `train_align.py` 会分别编码 a/b，使用同下标元素作为正样本，
    batch 内其它元素自然成为 InfoNCE 的负样本。
    """

    def __init__(
        self,
        items_path: str | Path,
        query_item_path: str | Path,
        item_item_path: str | Path,
        data_root: str | Path | None = None,
        query_item_ratio: float = 0.7,
    ) -> None:
        self.items = load_items(items_path)  # 加载 item 元数据。
        self.data_root = Path(data_root or Path(items_path).parent)  # 图片根目录。
        qi = read_table(query_item_path)  # 读取 query-item 正反馈。
        ii = read_table(item_item_path)  # 读取 item-item co-click 正对。
        self.samples: list[dict[str, Any]] = []  # 统一保存 pair 样本。
        for row in qi.to_dict("records"):  # 遍历 query-item 表。
            self.samples.append(  # 每行构成一个正 pair。
                {
                    "pair_type": "query_item",  # 标明 pair 类型。
                    "query": str(row["query"]),  # anchor 侧 query 文本。
                    "item_b": str(row["item_id"]),  # positive 侧 item。
                    "weight": float(row.get("weight", 1.0)),  # 可用于加权采样，当前代码只记录。
                }
            )
        for row in ii.to_dict("records"):  # 遍历 item-item 表。
            self.samples.append(  # 每行也是一个正 pair。
                {
                    "pair_type": "item_item",  # 标明 pair 类型。
                    "item_a": str(row["item_a"]),  # anchor 侧 item。
                    "item_b": str(row["item_b"]),  # positive 侧 item。
                    "weight": float(row.get("weight", 1.0)),  # 协同强度权重。
                }
            )
        self.query_item_ratio = query_item_ratio  # 保留配置；大规模训练建议用 WeightedRandomSampler 实现比例采样。

    def __len__(self) -> int:
        return len(self.samples)  # 返回 pair 数。

    def _encode_item_side(self, item_id: str) -> dict[str, Any]:
        """把一个 item_id 展开成模型输入侧需要的完整 item 信息。"""

        item = self.items[item_id]  # 找到 item 元数据。
        return {  # 返回 MLLM item 输入。
            "kind": "item",  # kind 用于调试和日志。
            "id": item_id,  # 原始 item id。
            "prompt": build_item_prompt(item),  # 图文 prompt。
            "caption": item.caption_text,  # B 阶段 caption anchor 可复用。
            "images": load_images(item.frame_paths, self.data_root),  # 真实训练时包含视频帧。
        }

    def _encode_query_side(self, query: str) -> dict[str, Any]:
        """把 query 文本包装成与 item 侧形态一致的输入 dict。"""

        return {  # 返回 MLLM query 输入。
            "kind": "query",  # query 侧没有 item_id。
            "id": query,  # 用 query 文本作为调试 ID。
            "prompt": build_query_prompt(query),  # query prompt。
            "caption": "",  # query 不参与 caption loss。
            "images": [],  # query 没有视觉输入。
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.samples[idx]  # 取出 pair 元信息。
        if row["pair_type"] == "query_item":  # query-item 正对。
            side_a = self._encode_query_side(row["query"])  # a 侧为 query。
            side_b = self._encode_item_side(row["item_b"])  # b 侧为 item。
        else:  # item-item 正对。
            side_a = self._encode_item_side(row["item_a"])  # a 侧为 item。
            side_b = self._encode_item_side(row["item_b"])  # b 侧为 item。
        return {  # 返回训练一步需要的 pair。
            "pair_type": row["pair_type"],  # 保留 pair 类型，便于监控比例。
            "a": side_a,  # anchor 输入。
            "b": side_b,  # positive 输入。
            "weight": row["weight"],  # 当前版本不加权 loss，后续可扩展。
        }


class QwenVLCollator:
    """把文本和图片样本转换为 Qwen2.5-VL processor 输入。

    Dataset 只负责返回普通 Python 对象；Collator 才负责把一个 batch 的样本变成
    `input_ids`、`attention_mask`、视觉 tensor 和可选 `labels`。

    两个训练阶段复用同一个 Collator：

    - `caption_collate()`：给 Stage I-A 使用，会拼入 assistant caption 并生成 labels。
    - `align_collate()`：给 Stage I-B 使用，会分别构造 a/b 两侧 batch，用于抽 embedding。
    """

    def __init__(self, processor: Any, max_pixels: int = 512 * 28 * 28) -> None:
        self.processor = processor  # HuggingFace AutoProcessor。
        self.max_pixels = max_pixels  # 控制视觉 token 数；生产环境很关键。

    def _message(self, prompt: str, images: list[Image.Image], answer: str | None = None) -> list[dict[str, Any]]:
        """构造 Qwen2.5-VL chat template 所需的 messages。

        有 answer 时用于 caption 训练，没有 answer 时用于 embedding 抽取。
        """

        content: list[dict[str, Any]] = []  # Qwen chat template 的 content 列表。
        for image in images:  # 每张图片单独作为一个 image content。
            content.append({"type": "image", "image": image})  # processor 会转成视觉 token。
        content.append({"type": "text", "text": prompt})  # 文本 prompt 总是存在。
        messages = [{"role": "user", "content": content}]  # 单轮 user prompt。
        if answer is not None and answer:  # caption 训练需要 assistant answer。
            messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})  # 目标 caption。
        return messages  # 返回 chat messages。

    def _batch_encode(self, examples: list[dict[str, Any]], with_labels: bool) -> dict[str, Any]:
        """把一组样本编码成 HuggingFace processor 的 tensor batch。

        `with_labels=True` 时，batch 会额外包含 `labels`，用于 caption loss。
        `with_labels=False` 时，只生成模型前向需要的输入，用于 `model.encode()`。
        """

        texts: list[str] = []  # 保存 apply_chat_template 后的文本。
        image_inputs: list[list[Image.Image] | None] = []  # 保存每条样本的图片。
        for ex in examples:  # 遍历 batch 中每条样本。
            messages = self._message(ex["prompt"], ex["images"], ex.get("caption") if with_labels else None)  # 构造消息。
            text = self.processor.apply_chat_template(  # 使用官方 chat template，避免手写特殊 token。
                messages,
                tokenize=False,
                add_generation_prompt=not with_labels,  # 表征抽取不需要 assistant answer。
            )
            texts.append(text)  # 保存模板化文本。
            image_inputs.append(ex["images"] if ex["images"] else None)  # 无图时传 None，支持样例降级。
        try:  # 不同 transformers 版本对 max_pixels 的接收位置略有差异。
            batch = self.processor(  # 新版本可能允许在 processor 调用里传 max_pixels。
                text=texts,
                images=image_inputs,
                padding=True,
                return_tensors="pt",
                max_pixels=self.max_pixels,
            )
        except TypeError:  # 如果当前版本不接受 max_pixels，就降级为普通 processor 调用。
            batch = self.processor(  # 降级路径仍可训练，但视觉 token 控制要在图片预处理阶段完成。
                text=texts,
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )
        if with_labels:  # caption loss 需要 labels。
            batch["labels"] = batch["input_ids"].clone()  # 简化版：所有 assistant token 参与 loss。
            # 风险提示：生产版应把 user prompt 的 labels 置为 -100，只保留 assistant caption。
            # 这里保留完整 labels 是为了让项目骨架更短；真正训练建议实现 assistant span mask。
        return batch  # 返回模型可直接接收的 batch。

    def caption_collate(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Stage I-A 的 collate：item prompt + 图片 + caption -> caption 训练 batch。"""

        return self._batch_encode(examples, with_labels=True)  # Stage I-A 使用 caption labels。

    def align_collate(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Stage I-B 的 collate：pair batch -> a/b 两侧编码结果。

        返回的 `a` 和 `b` 会分别送入 `model.encode()` 得到 `h_a` 和 `h_b`。
        返回的 `caption` 是可选 caption anchor，用来在对齐训练时保留 item 的文本生成能力。
        """

        side_a = [ex["a"] for ex in examples]  # 收集 anchor 侧。
        side_b = [ex["b"] for ex in examples]  # 收集 positive 侧。
        b_caption = [ex["b"] for ex in examples if ex["b"]["kind"] == "item"]  # caption anchor 只对 item 做。
        return {  # 返回 Stage I-B 训练 step 所需内容。
            "a": self._batch_encode(side_a, with_labels=False),  # a 侧只抽 embedding。
            "b": self._batch_encode(side_b, with_labels=False),  # b 侧只抽 embedding。
            "caption": self._batch_encode(b_caption, with_labels=True) if b_caption else None,  # 可选 caption 正则。
            "pair_type": [ex["pair_type"] for ex in examples],  # 监控 batch 类型比例。
        }
