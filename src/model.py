"""Qwen2.5-VL 表征模型封装。

本文件负责三件事：
1. 加载 Qwen2.5-VL backbone。
2. 给 backbone 加 LoRA，降低 Stage I 微调成本。
3. 从最后一层 hidden states 做 mean-pooling，得到论文公式 (2) 的 h。

Qwen2.5-VL-7B-Instruct 的公开配置要点：
- 语言侧是 decoder-only Transformer：28 层，hidden size 3584，28 个 attention heads，4 个 KV heads。
- 语言侧 MLP intermediate size 是 18944，激活函数是 SiLU，位置编码使用 RoPE。
- 视觉侧是 ViT-like encoder：32 层，hidden size 1280，16 个 attention heads。
- 视觉 patch size 为 14，temporal patch size 为 2，因此视频会先被切成时空 patch。
- 视觉侧输出经 multimodal merger/projector 对齐到语言 hidden size 3584。
- 代码里不手写这些模块，而是从 HuggingFace config/model 自动加载，避免结构漂移。
"""

from __future__ import annotations  # 延迟解析类型注解，减少导入成本。

from dataclasses import dataclass  # 用于保存模型配置。
from typing import Any  # processor/model 类型来自 transformers，运行时再检查。

import torch  # PyTorch 主包。
from torch import nn  # 神经网络模块基类。
import torch.nn.functional as F  # 用于 normalize。


@dataclass
class ModelConfig:
    """TRM Stage I 模型配置。"""

    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"  # 默认使用 7B 版本，兼顾效果和成本。
    embedding_dim: int = 1024  # 输出 h 的维度；Stage II RQ-Kmeans 常用 768/1024/1536。
    use_lora: bool = True  # 默认启用 LoRA，生产全参微调成本太高。
    lora_rank: int = 16  # LoRA rank；资源足够可提高到 32/64。
    lora_alpha: int = 32  # LoRA scaling；通常为 rank 的 2 倍。
    lora_dropout: float = 0.05  # 轻微 dropout，缓解小数据过拟合。
    freeze_vision: bool = True  # 默认冻结视觉塔，避免破坏预训练视觉感知。
    torch_dtype: str = "bfloat16"  # H100/A100 推荐 bf16。


def _dtype_from_name(name: str) -> torch.dtype:
    """把字符串 dtype 转成 torch dtype。"""

    if name == "float16":  # fp16 显存更省，但数值稳定性略弱。
        return torch.float16  # 返回 fp16。
    if name == "float32":  # 调试时可用 fp32，但 7B 训练显存压力很大。
        return torch.float32  # 返回 fp32。
    return torch.bfloat16  # 默认 bf16，推荐用于 Ampere/Hopper GPU。


def load_processor(model_name: str) -> Any:
    """加载 Qwen2.5-VL 的 AutoProcessor。"""

    from transformers import AutoProcessor  # 延迟导入，便于没有依赖时阅读源码。

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)  # 使用官方 processor。
    return processor  # 返回 processor，供 collator 使用。


def build_backbone(config: ModelConfig) -> nn.Module:
    """加载 Qwen2.5-VL backbone，并按需添加 LoRA。"""

    from transformers import AutoModelForVision2Seq  # Qwen2.5-VL 在新版 transformers 走该接口。

    dtype = _dtype_from_name(config.torch_dtype)  # 转换 dtype。
    model = AutoModelForVision2Seq.from_pretrained(  # 从 HuggingFace 或本地缓存加载模型。
        config.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if hasattr(model, "gradient_checkpointing_enable"):  # 大模型训练通常需要 gradient checkpointing。
        model.gradient_checkpointing_enable()  # 以额外计算换显存。
    if config.freeze_vision:  # 冻结视觉 encoder 是 Stage I-A 的保守策略。
        for name, param in model.named_parameters():  # 遍历全部参数。
            if "visual" in name or "vision" in name:  # Qwen-VL 视觉塔参数名通常含 visual/vision。
                param.requires_grad = False  # 不更新视觉塔，减少显存和过拟合风险。
    if config.use_lora:  # 如果启用参数高效微调。
        from peft import LoraConfig, get_peft_model  # 延迟导入 PEFT。

        lora_cfg = LoraConfig(  # 构造 LoRA 配置。
            r=config.lora_rank,  # 低秩矩阵 rank。
            lora_alpha=config.lora_alpha,  # LoRA 缩放。
            lora_dropout=config.lora_dropout,  # LoRA dropout。
            bias="none",  # 通常不训练 bias，减少参数。
            task_type="CAUSAL_LM",  # Qwen2.5-VL 语言侧是 causal LM。
            target_modules=[  # 这些名字覆盖 Qwen 系常见 attention/MLP 线性层。
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, lora_cfg)  # 包装成 PEFT 模型。
    return model  # 返回 backbone。


class TRMRepresentationModel(nn.Module):
    """论文 Stage I 的 MLLM 表征模型。"""

    def __init__(self, backbone: nn.Module, embedding_dim: int = 1024) -> None:
        super().__init__()  # 初始化 nn.Module。
        self.backbone = backbone  # Qwen2.5-VL 主干。
        hidden_size = self._infer_hidden_size(backbone)  # 从 config 推断语言 hidden size。
        self.proj = nn.Sequential(  # 小投影头，把 3584 维压到适合量化的维度。
            nn.Linear(hidden_size, hidden_size),  # 第一层保持宽度，学习表征变换。
            nn.GELU(),  # 平滑非线性。
            nn.Linear(hidden_size, embedding_dim),  # 输出最终 h 维度。
        )

    @staticmethod
    def _infer_hidden_size(backbone: nn.Module) -> int:
        """从不同包装层中推断 hidden size。"""

        cfg = getattr(backbone, "config", None)  # 普通 HF 模型有 config。
        if cfg is not None and hasattr(cfg, "hidden_size"):  # 直接模型配置。
            return int(cfg.hidden_size)  # Qwen2.5-VL-7B 为 3584。
        if cfg is not None and hasattr(cfg, "text_config"):  # 部分多模态模型将语言配置放在 text_config。
            return int(cfg.text_config.hidden_size)  # 从 text_config 取。
        base = getattr(backbone, "base_model", None)  # PEFT 包装后可能有 base_model。
        if base is not None:  # 如果存在 base_model。
            return TRMRepresentationModel._infer_hidden_size(base)  # 递归推断。
        raise ValueError("无法从 backbone config 推断 hidden_size，请手动检查模型配置。")  # 明确报错。

    def forward(self, batch: dict[str, torch.Tensor], labels: torch.Tensor | None = None) -> Any:
        """直接调用 backbone，主要用于 caption loss。"""

        if labels is not None:  # 如果显式传 labels。
            batch = dict(batch)  # 复制 batch，避免修改外部对象。
            batch["labels"] = labels  # 注入 labels。
        outputs = self.backbone(  # 调用 Qwen2.5-VL。
            **batch,
            output_hidden_states=True,  # 必须输出 hidden states，后续 pooling 要用。
            return_dict=True,  # 返回 ModelOutput，便于按字段访问。
        )
        return outputs  # 返回 logits、hidden_states 等。

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """抽取论文公式 (2) 的 mean-pooled dense representation h。"""

        outputs = self.forward(batch)  # 前向得到最后一层 hidden states。
        last_hidden = outputs.hidden_states[-1]  # 形状 [batch, seq_len, hidden]。
        attention_mask = batch["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)  # mask 扩展到 hidden 维。
        pooled = (last_hidden * attention_mask).sum(dim=1)  # 对有效 token 求和。
        denom = attention_mask.sum(dim=1).clamp_min(1.0)  # 防止空序列除零。
        pooled = pooled / denom  # mean-pooling，和论文公式 (2) 对齐。
        embedding = self.proj(pooled)  # 投影到 embedding_dim。
        embedding = F.normalize(embedding, dim=-1)  # 单位化，方便 cosine/InfoNCE。
        return embedding  # 返回 [batch, embedding_dim]。

    def save_trainable(self, output_dir: str) -> None:
        """保存 LoRA adapter 和 projection head。"""

        from pathlib import Path  # 局部导入，保持文件顶部依赖简洁。

        path = Path(output_dir)  # 转为 Path。
        path.mkdir(parents=True, exist_ok=True)  # 创建输出目录。
        if hasattr(self.backbone, "save_pretrained"):  # PEFT/HF 模型都支持 save_pretrained。
            self.backbone.save_pretrained(path / "backbone_adapter")  # 保存 LoRA adapter。
        torch.save(self.proj.state_dict(), path / "projection_head.pt")  # 保存投影头。

    def load_projection(self, path: str) -> None:
        """加载 projection head 权重。"""

        state = torch.load(path, map_location="cpu")  # 先加载到 CPU，避免占用 GPU 峰值显存。
        self.proj.load_state_dict(state)  # 恢复投影头参数。
