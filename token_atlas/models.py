"""Real, per-occurrence transformer representations with explicit context windows.

Model weights are loaded locally; text is never submitted to an embedding API.
Optional ML dependencies are imported lazily so saved/demo projects stay usable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .types import Article, TokenResult


MODEL_PRESETS = {
    "Qwen 2.5 · 0.5B（中文 / 多语言）": "Qwen/Qwen2.5-0.5B",
    "Qwen 3 · 0.6B（中文 / 多语言）": "Qwen/Qwen3-0.6B",
    "BERT · 中文（双向编码器）": "google-bert/bert-base-chinese",
}


@dataclass
class ModelHandle:
    model: Any
    tokenizer: Any
    model_id: str
    device: str = "cpu"
    revision: str | None = None


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("真实模型模式需要 PyTorch。请先安装项目的 models 依赖。") from exc
    return torch


def load_model(model_id: str, device: str = "auto", revision: str | None = None) -> ModelHandle:
    """Load a standard text backbone and fast tokenizer, without remote Python code."""
    torch = _torch()
    try:
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("真实模型模式需要 Transformers。请先安装项目的 models 依赖。") from exc
    model_id = model_id.strip()
    if not model_id:
        raise ValueError("请输入 Hugging Face 模型 ID 或本地模型目录。")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        )
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError("设备请选择 auto、cpu、cuda 或 mps。")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("这台电脑没有可用的 CUDA 设备，请选择 auto 或 cpu。")
    if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise ValueError("这台电脑没有可用的 Apple MPS 设备，请选择 auto 或 cpu。")
    try:
        config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=False)
        if getattr(config, "is_encoder_decoder", False):
            raise ValueError("首版支持单一文本编码器或解码器，暂不支持 encoder-decoder 模型。")
        resolved_revision = getattr(config, "_commit_hash", None) or revision
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=resolved_revision, use_fast=True, trust_remote_code=False
        )
        if not tokenizer.is_fast:
            raise ValueError("这个模型缺少 fast tokenizer，无法可靠定位 token 在原文中的位置。")
        model = AutoModel.from_pretrained(
            model_id,
            config=config,
            revision=resolved_revision,
            trust_remote_code=False,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        )
        model.to(device)
        model.eval()
    except ValueError as exc:
        raise ValueError(f"模型配置不受支持：{exc}") from exc
    except Exception as exc:
        raise RuntimeError(
            "模型加载失败。请检查模型 ID、网络连接、磁盘和内存；受限模型需要先在本机登录 "
            f"Hugging Face。可尝试较小模型或 CPU。原始错误：{exc}"
        ) from exc
    return ModelHandle(model, tokenizer, model_id, device, resolved_revision)


def _window_limit(handle: ModelHandle, requested: int) -> int:
    limits = [requested]
    # HF uses a huge sentinel when the tokenizer has no declared context limit.
    for value in (
        getattr(handle.tokenizer, "model_max_length", None),
        getattr(handle.model.config, "max_position_embeddings", None),
        getattr(handle.model.config, "n_positions", None),
    ):
        if isinstance(value, int) and 0 < value < 1_000_000:
            limits.append(value)
    return min(limits)


def _fair_quotas(counts: list[int], total_limit: int) -> list[int]:
    """Deterministically share a point budget; short articles return unused quota."""
    quotas = [0] * len(counts)
    remaining = min(sum(counts), total_limit)
    active = [i for i, n in enumerate(counts) if n]
    while active and remaining:
        share = max(1, remaining // len(active))
        for i in active:
            take = min(share, counts[i] - quotas[i], remaining)
            quotas[i] += take
            remaining -= take
            if not remaining:
                break
        active = [i for i in active if quotas[i] < counts[i]]
    return quotas


def _bounded_windows(tokenizer, text, raw, quota, window_size, capacity, overlap):
    """Build only needed full context windows, verifying prefix tokenization.

    Cutting raw text can change subword tokenization at the cut. A small
    look-ahead generally avoids that; exact ID and offset checks are mandatory.
    Unusual tokenizers fall back to progressively longer text, up to the whole
    article, rather than quietly changing the selected tokens' context.
    """
    total = len(raw["input_ids"])
    step = capacity - overlap
    last_window = max(0, (quota - capacity + step - 1) // step)
    required_end = min(total, last_window * step + capacity)
    lookahead_end = min(total, required_end + 8)
    prefix_chars = (
        len(text) if lookahead_end == total else
        max(end for _, end in raw["offset_mapping"][:lookahead_end])
    )
    prefix_chars = min(len(text), max(1, prefix_chars))
    while True:
        windows = tokenizer(
            text[:prefix_chars], add_special_tokens=True, truncation=True,
            max_length=window_size, stride=overlap,
            return_overflowing_tokens=True, return_offsets_mapping=True,
            return_special_tokens_mask=True, return_attention_mask=True,
            verbose=False,
        )
        valid = len(windows["input_ids"]) > last_window
        if valid:
            for index in range(last_window + 1):
                positions = [
                    i for i, (special, attended) in enumerate(zip(
                        windows["special_tokens_mask"][index], windows["attention_mask"][index]
                    )) if not special and attended
                ]
                start = index * step
                stop = min(total, start + capacity)
                ids = [windows["input_ids"][index][i] for i in positions]
                offsets = [tuple(windows["offset_mapping"][index][i]) for i in positions]
                if (ids != raw["input_ids"][start:stop] or
                        offsets != [tuple(pair) for pair in raw["offset_mapping"][start:stop]]):
                    valid = False
                    break
        if valid:
            return windows
        if prefix_chars == len(text):
            raise ValueError("此 tokenizer 的窗口无法与原始 token 和位置对齐，请换用受支持的模型。")
        prefix_chars = min(len(text), max(prefix_chars + 256, prefix_chars * 2))


def extract_tokens(
    articles: Sequence[Article],
    handle: ModelHandle,
    mode: str = "contextual",
    layer: int = -1,
    window_size: int = 256,
    overlap: int = 32,
    max_tokens_per_article: int = 1500,
    max_total_tokens: int = 6000,
    progress: Callable[[int, int, str], None] | None = None,
) -> TokenResult:
    """Extract input lookup vectors or contextual hidden states, never pooled vectors.

    Caps select a prefix from each article, with a fair global budget. Each
    retained occurrence uses its earliest containing window. Full containing
    windows are evaluated even when the output quota ends within that window.
    Overlap is left context for later new occurrences, and is never emitted twice.
    ``layer`` indexes the HF hidden_states tuple (0 is initial hidden state).
    """
    torch = _torch()
    mode = {"input_embedding": "input", "static": "input"}.get(mode, mode)
    if mode not in {"input", "contextual"}:
        raise ValueError("向量模式必须是 input（输入 embedding）或 contextual（上下文表示）。")
    if not articles or not any(article.text.strip() for article in articles):
        raise ValueError("请先导入至少一篇包含正文的文章。")
    if len({article.id for article in articles}) != len(articles):
        raise ValueError("每篇文章的 ID 必须唯一。")
    if min(window_size, max_tokens_per_article, max_total_tokens) < 1 or overlap < 0:
        raise ValueError("窗口和 token 数量上限必须为正整数，重叠数量不能为负。")
    if not getattr(handle.tokenizer, "is_fast", False):
        raise ValueError("需要 fast tokenizer 才能保留准确的原文位置。")
    if getattr(handle.model.config, "is_encoder_decoder", False):
        raise ValueError("暂不支持 encoder-decoder 模型。请选择单一文本编码器或解码器。")

    tokenizer = handle.tokenizer
    effective_window = _window_limit(handle, window_size)
    special_count = tokenizer.num_special_tokens_to_add(pair=False)
    content_capacity = effective_window - special_count
    if content_capacity < 1:
        raise ValueError("窗口过小，无法在模型特殊 token 之外容纳正文。")
    effective_overlap = min(overlap, content_capacity - 1)
    warnings: list[str] = []
    if effective_window != window_size:
        warnings.append(f"模型上下文限制：窗口从 {window_size} 调整为 {effective_window}（含特殊 token）。")
    if effective_overlap != overlap:
        warnings.append(f"重叠数从 {overlap} 调整为 {effective_overlap}，以小于正文窗口容量。")

    counts = []
    for article in articles:
        encoding = tokenizer(
            article.text, add_special_tokens=False, truncation=False,
            return_attention_mask=False, return_token_type_ids=False, verbose=False,
        )
        counts.append(len(encoding["input_ids"]))
        # Large imports must not retain every article's unbounded token arrays.
        # Counting one article at a time keeps memory proportional to one input.
        del encoding
    quotas = _fair_quotas([min(n, max_tokens_per_article) for n in counts], max_total_tokens)
    if not sum(quotas):
        raise ValueError("这些文章没有可提取的正文 token。")

    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    article_counts = []
    resolved_layer = None
    hidden_state_count = None
    handle.model.eval()
    for article_number, (article, total, quota) in enumerate(zip(articles, counts, quotas)):
        article_counts.append({
            "article_id": article.id, "title": article.title,
            "original_tokens": total, "retained_tokens": quota, "dropped_tokens": total - quota,
        })
        if quota < total:
            warnings.append(
                f"《{article.title}》原有 {total} 个 token，保留开头 {quota} 个；"
                f"其余 {total - quota} 个未绘制（每篇 / 总量上限）。"
            )
        if progress:
            progress(article_number, len(articles), f"正在处理《{article.title}》")
        if not quota:
            continue

        raw = tokenizer(
            article.text, add_special_tokens=False, truncation=False,
            return_offsets_mapping=True, verbose=False,
        )

        # Preserve full-window context, but avoid constructing overflow windows
        # for unselected article tails. Every ID and original offset is checked.
        windows = _bounded_windows(
            tokenizer, article.text, raw, quota, effective_window,
            content_capacity, effective_overlap,
        )
        original_start = 0
        emitted_until = 0
        for window_index, window_ids in enumerate(windows["input_ids"]):
            special_mask = windows["special_tokens_mask"][window_index]
            attention = windows["attention_mask"][window_index]
            content_positions = [
                i for i, (special, attended) in enumerate(zip(special_mask, attention))
                if not special and attended
            ]
            content_ids = [window_ids[i] for i in content_positions]
            expected = raw["input_ids"][original_start:original_start + len(content_positions)]
            if content_ids != expected:
                raise ValueError("此 tokenizer 的窗口无法与原始 token 对齐，请换用受支持的模型。")
            selected = [
                (position, original_start + local)
                for local, position in enumerate(content_positions)
                if emitted_until <= original_start + local < quota
            ]
            if selected:
                try:
                    with torch.inference_mode():
                        if mode == "input":
                            ids = torch.tensor(
                                [window_ids[position] for position, _ in selected],
                                dtype=torch.long, device=handle.device,
                            )
                            state = handle.model.get_input_embeddings()(ids)
                        else:
                            inputs = {}
                            for key in tokenizer.model_input_names:
                                if key in windows:
                                    inputs[key] = torch.tensor(
                                        [windows[key][window_index]], dtype=torch.long, device=handle.device
                                    )
                            forward_options = {"output_hidden_states": True, "return_dict": True}
                            if hasattr(handle.model.config, "use_cache"):
                                forward_options["use_cache"] = False
                            output = handle.model(**inputs, **forward_options)
                            states = output.hidden_states
                            if states is None:
                                raise ValueError("此模型没有返回逐 token hidden states。")
                            hidden_state_count = len(states)
                            if not -len(states) <= layer < len(states):
                                raise ValueError(f"层索引 {layer} 无效；该模型支持 {-len(states)} 到 {len(states) - 1}。")
                            resolved_layer = layer % len(states)
                            state = states[layer][0, [position for position, _ in selected], :]
                        block = state.detach().to(device="cpu", dtype=torch.float32).numpy().copy()
                        # Do not keep all layers of the previous window alive
                        # while evaluating the next window on a small GPU.
                        if mode == "contextual":
                            del output, states
                        del state
                except (ValueError, IndexError):
                    raise
                except Exception as exc:
                    raise RuntimeError(
                        f"《{article.title}》提取失败。可尝试减小窗口、较小模型或 CPU。原始错误：{exc}"
                    ) from exc
                if block.ndim != 2 or block.shape[0] != len(selected) or not np.isfinite(block).all():
                    raise ValueError("模型返回了无效向量；请尝试 CPU 或其他模型。")
                vectors.append(block)
                for position, original_index in selected:
                    start, end = raw["offset_mapping"][original_index]
                    if tuple(windows["offset_mapping"][window_index][position]) != (start, end):
                        raise ValueError("窗口的原文位置无法对齐，请换用受支持的模型。")
                    rows.append({
                        "article_id": article.id, "title": article.title,
                        "token_index": original_index, "token_id": int(window_ids[position]),
                        "token": tokenizer.convert_ids_to_tokens(int(window_ids[position])),
                        "char_start": int(start), "char_end": int(end),
                        "window_index": window_index if mode == "contextual" else 0,
                        "text": article.text[start:end],
                        "excerpt": article.text[max(0, start - 30):min(len(article.text), end + 30)],
                    })
                emitted_until = selected[-1][1] + 1
            if emitted_until >= quota:
                break
            original_start += len(content_positions) - effective_overlap
        if emitted_until != quota:
            raise ValueError(f"《{article.title}》的窗口未覆盖全部选中 token，请换用其他 tokenizer。")
        del raw, windows
    if progress:
        progress(len(articles), len(articles), "逐 token 向量提取完成")
    matrix = np.concatenate(vectors, axis=0).astype(np.float32, copy=False)
    result = TokenResult(matrix, rows, {
        "model_id": handle.model_id,
        "revision": handle.revision or getattr(handle.model.config, "_commit_hash", None),
        "tokenizer": tokenizer.__class__.__name__,
        "tokenizer_name": getattr(tokenizer, "name_or_path", handle.model_id),
        "device": handle.device, "mode": mode,
        "layer": layer if mode == "contextual" else None,
        "resolved_layer": resolved_layer, "hidden_state_count": hidden_state_count,
        "hidden_size": int(matrix.shape[1]), "dtype": "float32",
        "requested_window_size": window_size, "window_size": effective_window,
        "content_window_size": content_capacity, "overlap": effective_overlap,
        "max_tokens_per_article": max_tokens_per_article, "max_total_tokens": max_total_tokens,
        "original_tokens": sum(counts), "retained_tokens": len(rows),
        "dropped_tokens": sum(counts) - len(rows), "article_counts": article_counts,
        "selection_strategy": "deterministic_prefix_with_fair_global_budget",
        "context_strategy": (
            "input_embedding_table_no_context" if mode == "input" else
            "overlapping_windows_first_occurrence; positions_restart_per_window; "
            "full_containing_window_is_context_even_when_output_quota_ends_inside_it"
        ),
        "warnings": warnings,
    })
    result.validate()
    return result
