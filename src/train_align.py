"""Stage I-B：用 collaborative alignment loss 注入用户行为协同信号。"""

from __future__ import annotations  # 延迟类型注解解析。

import argparse  # 命令行参数。
from pathlib import Path  # 路径处理。

import torch  # PyTorch。
from torch.optim import AdamW  # AdamW 优化器。
from torch.utils.data import DataLoader  # DataLoader。
from tqdm import tqdm  # 进度条。

from .datasets import AlignmentPairDataset, QwenVLCollator  # pair 数据集和 collator。
from .losses import caption_cross_entropy, info_nce_loss, weighted_sum_loss  # Stage I-B 损失。
from .model import ModelConfig, TRMRepresentationModel, build_backbone, load_processor  # 模型构造。


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TRM Stage I-B collaborative alignment training")  # parser。
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")  # HF 模型名。
    parser.add_argument("--items_path", default="data/items.csv")  # item 表。
    parser.add_argument("--query_item_path", default="data/query_item_pairs.csv")  # query-item pair 表。
    parser.add_argument("--item_item_path", default="data/item_item_pairs.csv")  # item-item pair 表。
    parser.add_argument("--caption_adapter", default="")  # Stage I-A 产物；留空则从原模型开始。
    parser.add_argument("--output_dir", default="outputs/align_lora")  # 输出目录。
    parser.add_argument("--epochs", type=int, default=1)  # 对齐训练 epoch。
    parser.add_argument("--batch_size", type=int, default=2)  # 样例 batch；生产建议 global batch 2048+。
    parser.add_argument("--grad_accum", type=int, default=8)  # 梯度累积。
    parser.add_argument("--lr", type=float, default=5e-5)  # 对齐阶段 LoRA 学习率。
    parser.add_argument("--proj_lr", type=float, default=1e-4)  # projection head 学习率可更高。
    parser.add_argument("--lambda_align", type=float, default=0.1)  # 公式 (4) 中 lambda_align。
    parser.add_argument("--temperature", type=float, default=0.07)  # InfoNCE 温度 tau。
    parser.add_argument("--weight_decay", type=float, default=0.01)  # 权重衰减。
    parser.add_argument("--max_grad_norm", type=float, default=1.0)  # 梯度裁剪。
    parser.add_argument("--max_pixels", type=int, default=512 * 28 * 28)  # 视觉 token 上限。
    parser.add_argument("--device", default="cuda")  # 设备。
    parser.add_argument("--no_caption_anchor", action="store_true")  # 是否关闭 caption 正则。
    return parser.parse_args()  # 返回参数。


def maybe_load_adapter(model: TRMRepresentationModel, adapter_dir: str) -> None:
    """加载 Stage I-A 的 LoRA adapter 和 projection head。"""

    if not adapter_dir:  # 留空表示不加载。
        return  # 直接返回。
    path = Path(adapter_dir)  # 转为 Path。
    adapter_path = path / "backbone_adapter"  # train_caption.py 保存的 adapter 路径。
    projection_path = path / "projection_head.pt"  # train_caption.py 保存的投影头。
    if adapter_path.exists():  # 如果 adapter 存在。
        from peft import PeftModel  # 延迟导入 PEFT。

        model.backbone = PeftModel.from_pretrained(model.backbone, adapter_path, is_trainable=True)  # 加载并保持可训练。
    if projection_path.exists():  # 如果 projection head 存在。
        model.load_projection(str(projection_path))  # 恢复投影头。


def main() -> None:
    args = parse_args()  # 解析参数。
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")  # 没 GPU 时降级。
    processor = load_processor(args.model_name)  # 加载 processor。
    dataset = AlignmentPairDataset(  # 构建 pair 数据集。
        args.items_path,
        args.query_item_path,
        args.item_item_path,
    )
    collator = QwenVLCollator(processor, max_pixels=args.max_pixels)  # 构建 collator。
    loader = DataLoader(  # 构建 DataLoader。
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator.align_collate,
    )
    cfg = ModelConfig(model_name=args.model_name)  # 模型配置。
    backbone = build_backbone(cfg)  # 加载 backbone。
    model = TRMRepresentationModel(backbone, embedding_dim=cfg.embedding_dim).to(device)  # 表征模型。
    maybe_load_adapter(model, args.caption_adapter)  # 可选加载 Stage I-A checkpoint。
    backbone_params = []  # 保存 backbone 可训练参数。
    proj_params = []  # 保存 projection head 参数。
    for name, param in model.named_parameters():  # 遍历参数。
        if not param.requires_grad:  # 跳过冻结参数。
            continue  # 继续下一个。
        if name.startswith("proj."):  # projection head 单独使用 proj_lr。
            proj_params.append(param)  # 加入投影头参数组。
        else:  # 其它参数通常是 LoRA。
            backbone_params.append(param)  # 加入 backbone 参数组。
    optimizer = AdamW(  # 构建多参数组优化器。
        [
            {"params": backbone_params, "lr": args.lr},  # LoRA 学习率。
            {"params": proj_params, "lr": args.proj_lr},  # projection head 学习率。
        ],
        weight_decay=args.weight_decay,
    )
    model.train()  # 切换训练模式。
    global_step = 0  # 全局 step。
    optimizer.zero_grad(set_to_none=True)  # 清空梯度。
    for epoch in range(args.epochs):  # 遍历 epoch。
        pbar = tqdm(loader, desc=f"align epoch {epoch}")  # 进度条。
        for step, batch in enumerate(pbar):  # 遍历 batch。
            a_batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch["a"].items()}  # a 侧上设备。
            b_batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch["b"].items()}  # b 侧上设备。
            h_a = model.encode(a_batch)  # 抽取 anchor 表征。
            h_b = model.encode(b_batch)  # 抽取 positive 表征。
            align_loss = info_nce_loss(h_a, h_b, temperature=args.temperature)  # 计算 InfoNCE。
            cap_loss = None  # 默认没有 caption anchor。
            if (not args.no_caption_anchor) and batch["caption"] is not None:  # 如果启用 caption 正则。
                cap_batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch["caption"].items()}  # 上设备。
                labels = cap_batch.pop("labels")  # 取出 caption labels。
                outputs = model(cap_batch, labels=None)  # 前向得到 logits。
                cap_loss = caption_cross_entropy(outputs.logits, labels)  # 计算 caption CE。
            loss = weighted_sum_loss(cap_loss, align_loss, args.lambda_align)  # 组合 L_rep。
            (loss / args.grad_accum).backward()  # 梯度累积。
            if (step + 1) % args.grad_accum == 0:  # 到达累积步数。
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  # 梯度裁剪。
                optimizer.step()  # 参数更新。
                optimizer.zero_grad(set_to_none=True)  # 清空梯度。
                global_step += 1  # 计数。
            pos_cos = (h_a * h_b).sum(dim=-1).mean().detach().cpu().item()  # 正样本 cosine 均值。
            pbar.set_postfix(  # 展示关键训练指标。
                loss=float(loss.detach().cpu()),
                align=float(align_loss.detach().cpu()),
                cap=float(cap_loss.detach().cpu()) if cap_loss is not None else -1.0,
                pos_cos=pos_cos,
                step=global_step,
            )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)  # 确保输出目录存在。
    model.save_trainable(args.output_dir)  # 保存 Stage I-B adapter 和 projection head。


if __name__ == "__main__":  # 脚本入口。
    main()  # 执行训练。
