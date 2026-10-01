"""Interactive token maps and portable exports with stable article colours."""
from __future__ import annotations

import colorsys
import csv
import hashlib
import html
import io
import json
from threading import Lock
from typing import Any, Iterable
import zipfile

import numpy as np
import plotly.graph_objects as go

from .types import TokenResult

_IMAGE_LOCK = Lock()


def article_color(article_id: str) -> str:
    """An article keeps its colour across projections and visibility changes."""
    digest = hashlib.sha256(str(article_id).encode("utf-8")).digest()
    hue = int.from_bytes(digest[:4], "big") / 2**32
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.68, 0.76)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def _title(token: dict[str, Any]) -> str:
    return str(token.get("title", token.get("article_title", token["article_id"])))


def _position(token: dict[str, Any]) -> int:
    return int(token.get("token_index", token.get("position", token.get("index", 0))))


def _window(token: dict[str, Any]) -> Any:
    return token.get("window_index", token.get("window", token.get("window_id", 0)))


def _validate(tokens: list[dict[str, Any]], coordinates: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape != (len(tokens), 2):
        raise ValueError("每个 token 必须有且只有一对二维坐标。")
    if not np.isfinite(coordinates).all():
        raise ValueError("坐标包含非有限值。")
    if any("article_id" not in token for token in tokens):
        raise ValueError("每个 token 必须具有 article_id。")
    return coordinates


def _bounds(coordinates: np.ndarray) -> tuple[list[float], list[float]]:
    if not len(coordinates):
        return [-1.0, 1.0], [-1.0, 1.0]
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    margin = np.maximum((maximum - minimum) * 0.07, 0.05)
    return (minimum[0] - margin[0], maximum[0] + margin[0]), (minimum[1] - margin[1], maximum[1] + margin[1])


def _groups(tokens: list[dict[str, Any]], article_ids: Iterable[str] | None) -> list[tuple[str, str, list[int]]]:
    selected = set(map(str, article_ids)) if article_ids is not None else None
    groups: dict[str, list[int]] = {}
    titles: dict[str, str] = {}
    for index, token in enumerate(tokens):
        article_id = str(token["article_id"])
        groups.setdefault(article_id, []).append(index)
        titles.setdefault(article_id, _title(token))
    duplicate_titles = {title for title in titles.values() if sum(other == title for other in titles.values()) > 1}
    return [
        (article_id, titles[article_id] + (f" · {article_id[-6:]}" if titles[article_id] in duplicate_titles else ""), indices)
        for article_id, indices in groups.items()
        if selected is None or article_id in selected
    ]


def _line_coordinates(tokens: list[dict[str, Any]], coordinates: np.ndarray, indices: list[int]) -> tuple[list[Any], list[Any]]:
    """Connect only consecutive original tokens in the same context window."""
    ordered = sorted(indices, key=lambda index: (_position(tokens[index]), index))
    xs: list[Any] = []
    ys: list[Any] = []
    for previous, current in zip(ordered, ordered[1:]):
        if _position(tokens[current]) == _position(tokens[previous]) + 1 and _window(tokens[current]) == _window(tokens[previous]):
            xs.extend([float(coordinates[previous, 0]), float(coordinates[current, 0]), None])
            ys.extend([float(coordinates[previous, 1]), float(coordinates[current, 1]), None])
    return xs, ys


def build_figure(
    tokens: list[dict[str, Any]],
    coordinates: np.ndarray,
    article_ids: Iterable[str] | None = None,
    title: str = "",
    connect_tokens: bool = False,
) -> go.Figure:
    """Build a Plotly map. Filtering never recomputes coordinates or axis limits."""
    coordinates = _validate(tokens, coordinates)
    figure = go.Figure()
    for article_id, label, indices in _groups(tokens, article_ids):
        color = article_color(article_id)
        if connect_tokens:
            xs, ys = _line_coordinates(tokens, coordinates, indices)
            if xs:
                figure.add_trace(go.Scattergl(
                    x=xs, y=ys, mode="lines", line={"color": color, "width": 0.8},
                    opacity=0.28, legendgroup=article_id, showlegend=False, hoverinfo="skip",
                    name=html.escape(label),
                ))
        hover = [
            [html.escape(str(tokens[index].get("token", ""))).replace("\n", "↵"),
             _position(tokens[index]), html.escape(_title(tokens[index])), html.escape(str(_window(tokens[index]))),
             html.escape(str(tokens[index].get("text", tokens[index].get("token", "")))).replace("\n", "↵").replace("\t", "⇥"),
             html.escape(str(tokens[index].get("excerpt", tokens[index].get("text", tokens[index].get("token", ""))))).replace("\n", "↵").replace("\t", "⇥")]
            for index in indices
        ]
        figure.add_trace(go.Scattergl(
            x=coordinates[indices, 0], y=coordinates[indices, 1], mode="markers",
            name=html.escape(label), legendgroup=article_id, customdata=hover,
            marker={"color": color, "size": 6, "opacity": 0.72},
            hovertemplate="<b>%{customdata[2]}</b><br>Token: %{customdata[0]}<br>对应原文：%{customdata[4]}<br>上下文：%{customdata[5]}<br>原始位置：%{customdata[1]}<br>窗口：%{customdata[3]}<extra></extra>",
        ))
    x_range, y_range = _bounds(coordinates)
    figure.update_layout(
        title={"text": html.escape(title), "x": 0.02}, template="plotly_white",
        xaxis={"title": "投影维度 1", "range": list(x_range), "autorange": False, "zeroline": False, "constrain": "domain"},
        yaxis={"title": "投影维度 2", "range": list(y_range), "autorange": False, "zeroline": False, "constrain": "domain", "scaleanchor": "x", "scaleratio": 1},
        legend={"title": {"text": "文章"}, "groupclick": "togglegroup", "itemsizing": "constant",
                "orientation": "h", "x": 0, "xanchor": "left", "y": -0.14, "yanchor": "top", "font": {"size": 11}},
        margin={"l": 24, "r": 24, "t": 60 if title else 24, "b": 100}, height=520,
        hovermode="closest", uirevision=hashlib.sha256(coordinates.tobytes()).hexdigest()[:16],
        paper_bgcolor="#ffffff", plot_bgcolor="#fbfcff",
    )
    return figure


def figure_html(
    tokens: list[dict[str, Any]], coordinates: np.ndarray,
    article_ids: Iterable[str] | None = None, title: str = "", connect_tokens: bool = False,
) -> str:
    """A self-contained HTML file; Plotly's JSON encoder escapes script tags."""
    figure = build_figure(tokens, coordinates, article_ids, title, connect_tokens)
    return figure.to_html(full_html=True, include_plotlyjs=True, config={"displaylogo": False, "responsive": True})


def export_image(
    tokens: list[dict[str, Any]], coordinates: np.ndarray,
    article_ids: Iterable[str] | None = None, title: str = "", connect_tokens: bool = False,
    format: str = "png",
) -> bytes:
    """Export PNG/SVG through Matplotlib; no browser or Kaleido installation."""
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if format not in {"png", "svg"}:
        raise ValueError("图片格式必须是 png 或 svg。")
    coordinates = _validate(tokens, coordinates)
    groups = _groups(tokens, article_ids)
    font_names = ["PingFang SC", "PingFang HK", "Heiti SC", "Heiti TC", "Songti SC", "Microsoft YaHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "Arial Unicode MS", "DejaVu Sans"]
    with _IMAGE_LOCK, matplotlib.rc_context({"font.family": "sans-serif", "font.sans-serif": font_names, "axes.unicode_minus": False, "text.parse_math": False}):
        height = max(7, min(30, 0.22 * len(groups) + 1.5))
        figure = Figure(figsize=(12, height), dpi=150, facecolor="white")
        FigureCanvasAgg(figure)
        axis = figure.add_subplot(111)
        axis.set_facecolor("#fbfcff")
        for article_id, label, indices in groups:
            color = article_color(article_id)
            if connect_tokens:
                xs, ys = _line_coordinates(tokens, coordinates, indices)
                if xs:
                    axis.plot(xs, ys, color=color, alpha=0.22, linewidth=0.65, zorder=1)
            axis.scatter(coordinates[indices, 0], coordinates[indices, 1], color=color, s=13, alpha=0.72, linewidths=0, label=label, zorder=2)
        x_range, y_range = _bounds(coordinates)
        axis.set_xlim(x_range)
        axis.set_ylim(y_range)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("投影维度 1")
        axis.set_ylabel("投影维度 2")
        axis.set_title(title, loc="left", pad=18)
        axis.grid(alpha=0.12)
        if groups:
            # All selected articles receive a legend entry, even for large exports.
            columns = max(1, (len(groups) + 99) // 100)
            axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8, ncol=columns, title="文章")
        buffer = io.BytesIO()
        figure.savefig(buffer, format=format, bbox_inches="tight", dpi=150)
        return buffer.getvalue()


def coordinates_csv(tokens: list[dict[str, Any]], coordinates: np.ndarray, article_ids: Iterable[str] | None = None) -> bytes:
    """Spreadsheet-friendly UTF-8 CSV, with formula-like text neutralized."""
    coordinates = _validate(tokens, coordinates)
    selected = set(map(str, article_ids)) if article_ids is not None else None
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["article_id", "title", "token_index", "token_id", "token", "char_start", "char_end", "window_index", "x", "y"])
    for token, point in zip(tokens, coordinates):
        if selected is not None and str(token["article_id"]) not in selected:
            continue
        def safe_text(value: Any) -> str:
            result = str(value)
            return "'" + result if result.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")) else result
        writer.writerow([
            safe_text(token["article_id"]), safe_text(_title(token)), _position(token), token.get("token_id", ""),
            safe_text(token.get("token", "")), token.get("char_start", ""), token.get("char_end", ""), _window(token),
            float(point[0]), float(point[1]),
        ])
    return buffer.getvalue().encode("utf-8-sig")


def export_bundle(
    result: TokenResult, coordinates: np.ndarray, projection_metadata: dict[str, Any],
    article_ids: Iterable[str] | None = None, *, title: str = "Token Atlas", connect_tokens: bool = False,
) -> bytes:
    """Export HTML, CSV, lossless JSON records, and model/projection settings."""
    coordinates = _validate(result.tokens, coordinates)
    selected = list(map(str, article_ids)) if article_ids is not None else None
    selected_set = set(selected) if selected is not None else None
    records = [dict(token, x=float(point[0]), y=float(point[1])) for token, point in zip(result.tokens, coordinates)
               if selected_set is None or str(token["article_id"]) in selected_set]
    metadata = {
        "format_version": 1, "model": result.metadata, "projection": projection_metadata,
        "selected_article_ids": selected, "total_token_count": len(result.tokens), "exported_token_count": len(records),
        "view": {"title": title, "connect_tokens": connect_tokens},
        "coordinate_frame": "Fitted on all analysed articles; filters retain full coordinate bounds.",
        "csv_note": "Formula-like text is prefixed with an apostrophe. tokens.json preserves exact strings.",
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("figure.html", figure_html(result.tokens, coordinates, selected, title, connect_tokens))
        archive.writestr("coordinates.csv", coordinates_csv(result.tokens, coordinates, selected))
        archive.writestr("tokens.json", json.dumps(records, ensure_ascii=False, indent=2, default=_json_default))
        archive.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default))
    return buffer.getvalue()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")
