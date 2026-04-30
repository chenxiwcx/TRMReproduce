"""TRM Stage I 的损失函数实现。"""

from __future__ import annotations  # 允许在旧 Python 版本里使用更轻量的类型注解。

import torch  # PyTorch 主包，提供 Tensor、自动求导和分布式张量操作。
import torch.nn.functional as F  # 函数式 API，主要用于 cross entropy 和归一化。


def caption_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """计算论文公式 (1) 的 caption loss。

    参数:
        logits: 形状为 [batch, seq_len, vocab_size]，来自 Qwen2.5-VL 语言头。
        labels: 形状为 [batch, seq_len]，caption token id；非 caption 区域应为 -100。

    返回:
        标准 next-token prediction cross entropy。
    """

    vocab_size = logits.size(-1)  # 取词表大小；Qwen2.5-VL-7B 通常约 15 万。
    shifted_logits = logits[:, :-1, :].contiguous()  # 第 k 个位置预测第 k+1 个 token。
    shifted_labels = labels[:, 1:].contiguous()  # label 右移后与 shifted_logits 对齐。
    loss = F.cross_entropy(  # PyTorch 内置 CE 已包含 log_softmax 和 NLL。
        shifted_logits.view(-1, vocab_size),  # 展平成 [batch * seq, vocab]。
        shifted_labels.view(-1),  # 展平成 [batch * seq]。
        ignore_index=-100,  # -100 位置不计入 loss，避免 prompt token 被训练。
    )
    return loss  # 返回一个标量 Tensor，可直接反向传播。


def info_nce_loss(
    h_a: torch.Tensor,
    h_b: torch.Tensor,
    temperature: float = 0.07,
    false_negative_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """计算论文公式 (3) 的单向 InfoNCE collaborative alignment loss。

    参数:
        h_a: anchor 表征，形状 [batch, dim]；可以是 query 或 item。
        h_b: positive 表征，形状 [batch, dim]；与 h_a 的同下标元素构成正对。
        temperature: 温度系数 tau，常用 0.05 到 0.1。
        false_negative_mask: 可选布尔矩阵 [batch, batch]，True 表示该位置不应作为负样本。

    返回:
        单向 a -> b 的对比学习 loss。
    """

    h_a = F.normalize(h_a, dim=-1)  # 先做 L2 归一化，使点积等价于 cosine similarity。
    h_b = F.normalize(h_b, dim=-1)  # b 侧同样归一化，避免 embedding norm 作弊。
    logits = h_a @ h_b.t()  # 得到 [batch, batch] 相似度矩阵。
    logits = logits / temperature  # 温度越小，softmax 分布越尖锐，训练也更不稳定。
    if false_negative_mask is not None:  # 当 batch 内存在其它正相关 item 时，启用掩码。
        logits = logits.masked_fill(false_negative_mask, -1e4)  # 用很小值移除伪负样本。
    labels = torch.arange(h_a.size(0), device=h_a.device)  # 对角线是正样本位置。
    loss = F.cross_entropy(logits, labels)  # 每行做一次多分类 CE。
    return loss  # 返回对齐 loss 标量。


def symmetric_info_nce_loss(
    h_a: torch.Tensor,
    h_b: torch.Tensor,
    temperature: float = 0.07,
    false_negative_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """计算双向 InfoNCE；工程上常比单向更稳，但论文公式只写了单向。"""

    loss_ab = info_nce_loss(h_a, h_b, temperature, false_negative_mask)  # anchor 为 a。
    loss_ba = info_nce_loss(h_b, h_a, temperature, None)  # anchor 为 b；简化起见不复用 mask。
    return 0.5 * (loss_ab + loss_ba)  # 两个方向取平均，避免量级翻倍。


def weighted_sum_loss(
    caption_loss: torch.Tensor | None,
    align_loss: torch.Tensor,
    lambda_align: float,
) -> torch.Tensor:
    """组合 Stage I-B 的 L_rep = L_cap + lambda_align * L_align。"""

    if caption_loss is None:  # 允许只跑 alignment，用于 ablation 或快速调试。
        return lambda_align * align_loss  # 只返回对齐项。
    return caption_loss + lambda_align * align_loss  # 与论文公式 (4) 一致。
