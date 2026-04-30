"""Stage I-A：用 caption loss 做领域多模态适配。"""

from __future__ import annotations  # 延迟类型注解解析。

import argparse  # 解析命令行参数。
from pathlib import Path  # 处理输出目录。

import torch  # PyTorch。
from torch.utils.data import DataLoader  # 小规模单机 DataLoader；生产建议接分布式采样。
from torch.optim import AdamW  # AdamW 优化器。
from tqdm import tqdm  # 训练进度条。

from .datasets import CaptionDataset, QwenVLCollator  # 数据集和 collator。
from .losses import caption_cross_entropy  # caption CE。
from .model import ModelConfig, TRMRepresentationModel, build_backbone, load_processor  # 模型构造。


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRM Stage I-A captioning training")  # 创建 parser。
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")  # HF 模型名或本地路径。
    parser.add_argument("--items_path", default="data/items.csv")  # item 元数据表。
    parser.add_argument("--output_dir", default="outputs/caption_lora")  # 输出目录。
    parser.add_argument("--epochs", type=int, default=1)  # 大规模数据通常 1 epoch 起步。
    parser.add_argument("--batch_size", type=int, default=1)  # 单卡样例用 1；生产靠梯度累积和多卡扩大。
    parser.add_argument("--grad_accum", type=int, default=8)  # 梯度累积步数。
    parser.add_argument("--lr", type=float, default=1e-4)  # LoRA 推荐 1e-4；全参要小一档。
    parser.add_argument("--weight_decay", type=float, default=0.01)  # AdamW 权重衰减。
    parser.add_argument("--max_grad_norm", type=float, default=1.0)  # 梯度裁剪阈值。
    parser.add_argument("--max_pixels", type=int, default=512 * 28 * 28)  # 控制视觉 token 数。
    parser.add_argument("--device", default="cuda")  # 默认使用 GPU。
    return parser.parse_args()  # 返回参数。


def main() -> None:
    args = parse_args()  # 读取命令行参数。
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")  # 无 GPU 时降级 CPU，仅用于调试。
    processor = load_processor(args.model_name)  # 加载 Qwen2.5-VL processor。
    dataset = CaptionDataset(args.items_path)  # 构建 caption 数据集。
    collator = QwenVLCollator(processor, max_pixels=args.max_pixels)  # 构建 collator。
    loader = DataLoader(  # 构建 DataLoader。
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator.caption_collate,
    )
    cfg = ModelConfig(model_name=args.model_name)  # 模型配置。
    backbone = build_backbone(cfg)  # 加载 Qwen2.5-VL + LoRA。
    model = TRMRepresentationModel(backbone, embedding_dim=cfg.embedding_dim).to(device)  # 包装表征模型。
    optimizer = AdamW(  # 优化 LoRA 参数和 projection head。
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    model.train()  # 切换训练模式。
    global_step = 0  # 全局 step 计数。
    optimizer.zero_grad(set_to_none=True)  # 清空梯度。
    for epoch in range(args.epochs):  # 遍历 epoch。
        pbar = tqdm(loader, desc=f"caption epoch {epoch}")  # 进度条。
        for step, batch in enumerate(pbar):  # 遍历 batch。
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}  # Tensor 移到设备。
            labels = batch.pop("labels")  # 取出 labels，避免重复传参。
            outputs = model(batch, labels=None)  # 前向得到 logits。
            loss = caption_cross_entropy(outputs.logits, labels)  # 计算 caption CE。
            (loss / args.grad_accum).backward()  # 梯度累积时缩放 loss。
            if (step + 1) % args.grad_accum == 0:  # 到达累积步数才更新。
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  # 梯度裁剪。
                optimizer.step()  # 参数更新。
                optimizer.zero_grad(set_to_none=True)  # 清空梯度。
                global_step += 1  # 更新全局 step。
            pbar.set_postfix(loss=float(loss.detach().cpu()), step=global_step)  # 显示训练指标。
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)  # 确保输出目录存在。
    model.save_trainable(args.output_dir)  # 保存 LoRA adapter 和 projection head。


if __name__ == "__main__":  # 脚本入口。
    main()  # 执行训练。
