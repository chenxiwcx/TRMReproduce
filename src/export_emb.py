"""导出 Stage I 训练后的 item dense embedding h。"""

from __future__ import annotations  # 延迟类型注解解析。

import argparse  # 命令行参数。
from pathlib import Path  # 路径处理。

import pandas as pd  # 保存 embedding 表。
import torch  # PyTorch。
from torch.utils.data import DataLoader  # DataLoader。
from tqdm import tqdm  # 进度条。

from .datasets import CaptionDataset, QwenVLCollator  # 复用 item dataset。
from .model import ModelConfig, TRMRepresentationModel, build_backbone, load_processor  # 模型构造。


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export TRM Stage I item embeddings")  # parser。
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")  # 模型名。
    parser.add_argument("--items_path", default="data/items.csv")  # item 表。
    parser.add_argument("--adapter_dir", default="outputs/align_lora")  # Stage I-B checkpoint。
    parser.add_argument("--output_path", default="outputs/item_embeddings.parquet")  # 输出 embedding 表。
    parser.add_argument("--batch_size", type=int, default=1)  # 导出 batch。
    parser.add_argument("--max_pixels", type=int, default=512 * 28 * 28)  # 视觉 token 上限。
    parser.add_argument("--device", default="cuda")  # 设备。
    return parser.parse_args()  # 返回参数。


def load_adapter_if_exists(model: TRMRepresentationModel, adapter_dir: str) -> None:
    """加载对齐后的 adapter 和 projection head。"""

    path = Path(adapter_dir)  # 转为 Path。
    adapter_path = path / "backbone_adapter"  # LoRA adapter 目录。
    projection_path = path / "projection_head.pt"  # projection head 文件。
    if adapter_path.exists():  # adapter 存在才加载。
        from peft import PeftModel  # 延迟导入 PEFT。

        model.backbone = PeftModel.from_pretrained(model.backbone, adapter_path, is_trainable=False)  # 推理模式加载。
    if projection_path.exists():  # projection 存在才加载。
        model.load_projection(str(projection_path))  # 恢复投影头。


def main() -> None:
    args = parse_args()  # 解析参数。
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")  # 选择设备。
    processor = load_processor(args.model_name)  # 加载 processor。
    dataset = CaptionDataset(args.items_path)  # item 数据集；只用 prompt/images/item_id。
    collator = QwenVLCollator(processor, max_pixels=args.max_pixels)  # collator。
    loader = DataLoader(  # DataLoader。
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda xs: (xs, collator._batch_encode(xs, with_labels=False)),  # 同时保留原始 item_id。
    )
    cfg = ModelConfig(model_name=args.model_name)  # 模型配置。
    model = TRMRepresentationModel(build_backbone(cfg), embedding_dim=cfg.embedding_dim).to(device)  # 构造模型。
    load_adapter_if_exists(model, args.adapter_dir)  # 加载 Stage I-B 权重。
    model.eval()  # 推理模式。
    rows = []  # 保存导出结果。
    with torch.no_grad():  # 导出不需要梯度。
        for raw_examples, batch in tqdm(loader, desc="export embeddings"):  # 遍历 item。
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}  # Tensor 上设备。
            emb = model.encode(batch).detach().cpu().float().numpy()  # 得到 numpy embedding。
            for ex, vec in zip(raw_examples, emb):  # 写出每个 item。
                rows.append(  # 每行包含 item_id 和向量。
                    {
                        "item_id": ex["item_id"],  # item id 用于 Stage II tokenization 对齐。
                        "embedding": vec.tolist(),  # parquet 可保存 list；CSV 会保存为字符串。
                    }
                )
    df = pd.DataFrame(rows)  # 构建 DataFrame。
    output_path = Path(args.output_path)  # 输出路径。
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 创建父目录。
    if output_path.suffix.lower() == ".parquet":  # parquet 输出。
        df.to_parquet(output_path, index=False)  # 需要 pyarrow/fastparquet。
    else:  # 非 parquet 输出。
        df.to_csv(output_path, index=False)  # CSV 降级路径，方便没有 pyarrow 时调试。


if __name__ == "__main__":  # 脚本入口。
    main()  # 执行导出。
