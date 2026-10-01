"""Local-first, beginner-friendly token representation explorer."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from token_atlas.demo import available_demos, load_demo  # noqa: E402
from token_atlas.documents import parse_uploads  # noqa: E402
from token_atlas.plotting import build_figure, export_bundle, export_image, figure_html  # noqa: E402
from token_atlas.projection import ALGORITHMS, project  # noqa: E402

st.set_page_config(page_title="文迹 · Token Atlas", page_icon="✳️", layout="wide")
st.markdown("""<style>
  .block-container {padding-top:4rem; padding-bottom:3rem; max-width:1600px}
  h1 {font-weight:650!important; letter-spacing:-.04em}
  [data-testid="stSidebar"] {border-right:1px solid #dfe5dc}
  [data-testid="stMetricValue"] {font-size:1.65rem}
  .eyebrow {font-size:.73rem;letter-spacing:.18em;color:#59816c;font-weight:700}
  .intro {color:#6a786e;font-size:1.04rem;max-width:800px;line-height:1.8}
  .note {padding:14px 18px;background:#edf2e9;border-radius:10px;font-size:.9rem;color:#49614d}
  [data-testid="stPlotlyChart"] {border:1px solid #e0e7dc;border-radius:12px;overflow:hidden}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=16)
def cached_projection(vectors: np.ndarray, algorithm: str, options: dict):
    return project(vectors, algorithm, **options)


@st.cache_resource(show_spinner=False, max_entries=1)
def cached_model(model_id: str, device: str, revision: str):
    from token_atlas.models import load_model
    return load_model(model_id, device=device, revision=revision or None)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_image(tokens, coordinates, article_ids, title, connect_tokens, image_format):
    return export_image(tokens, coordinates, article_ids=article_ids, title=title,
                        connect_tokens=connect_tokens, format=image_format)


def fingerprint(articles, settings):
    payload = {"articles": [(a.id, a.title, a.text) for a in articles], "settings": settings}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


st.markdown('<div class="eyebrow">TOKEN ATLAS / 文本表征实验室</div>', unsafe_allow_html=True)
st.title("文迹")
st.markdown('<p class="intro">把文章展开成一张 token 地图。切换模型与表示方式，观察文字在向量空间里的形态。</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 开始探索")
    source = st.radio("文章来源", ["体验真实模型示例", "分析我的文章"], label_visibility="collapsed")
    st.divider()
    st.markdown("### ① 表示方式")
    mode_label = st.radio("每个点从哪里来", ["上下文表示", "输入层 embedding"], label_visibility="collapsed")
    mode = "contextual" if mode_label == "上下文表示" else "input"
    st.caption("读取文章经过模型后，每个位置的向量。" if mode == "contextual" else "直接查模型输入向量表。相同 token ID 的向量相同。")
    st.markdown("### ② 二维投影")
    algorithm = st.selectbox("降维算法", ALGORITHMS, index=0, label_visibility="collapsed")
    descriptions = {
        "PCA": "线性基线 · 观察主要变化方向", "SVD": "不减均值的线性投影",
        "UMAP": "观察邻近关系 · 首次运行需要编译", "t-SNE": "观察局部邻近关系",
        "PaCMAP": "兼顾局部与整体结构 · 需安装可选依赖",
    }
    st.caption(descriptions[algorithm])
    with st.expander("投影参数"):
        normalize = st.checkbox("先把每个向量归一化", value=True,
                                  help="L2 归一化减少向量长度的影响；关闭后保留原始幅度。")
        seed = st.number_input("随机种子", min_value=0, max_value=2147483647, value=42)
        perplexity = st.slider("t-SNE perplexity", 2, 100, 30)
        neighbors = st.slider("邻居数（UMAP / PaCMAP）", 2, 100, 15)
        min_dist = st.slider("UMAP min_dist", 0.0, 1.0, 0.1)
    options = dict(seed=int(seed), normalize=normalize, perplexity=perplexity,
                   n_neighbors=neighbors, min_dist=min_dist)
    st.divider()
    st.caption("文章在本机计算。首次选择新模型会联网下载权重。普通文本 embedding API 不提供这里需要的逐 token 内部表示。")

runs = []
if source == "体验真实模型示例":
    demos = available_demos()
    if demos:
        choices = {r["id"]: r for r in demos}
        selected_demos = st.multiselect("选择模型，可并排比较", list(choices),
                                       default=list(choices)[:2],
                                       format_func=lambda x: choices[x]["label"], max_selections=2)
        st.caption("这些点来自真实预训练模型的预计算输出。示例无需下载模型；分析自己的文章请切换左侧来源。")
        runs = [(choices[k]["label"], load_demo(k, mode)) for k in selected_demos]
    else:
        st.info("预计算示例尚未生成。请切换到「分析我的文章」，导入文章并选择模型。")
else:
    with st.expander("① 导入文章", expanded=True):
        uploaded = st.file_uploader("拖入多篇文章或一个 ZIP", type=["txt", "md", "docx", "csv", "zip"], accept_multiple_files=True)
        st.caption("CSV 使用 title、text 两列；保留原文标点和换行。支持中文 UTF-8 / GB18030 文本。")
        with st.form("paste_article", clear_on_submit=True):
            col_a, col_b = st.columns([1, 3])
            with col_a:
                pasted_title = st.text_input("文章标题", placeholder="例如：雨后的街道")
            with col_b:
                pasted_text = st.text_area("或直接粘贴正文", height=100)
            if st.form_submit_button("加入文章列表"):
                if pasted_text.strip():
                    st.session_state.setdefault("pasted", []).append((pasted_title or f"粘贴文章 {len(st.session_state.get('pasted', [])) + 1}", pasted_text))
                else:
                    st.warning("请先输入正文。")
        extra_a, extra_b = st.columns(2)
        with extra_a:
            if st.button("加入随附示例文章"):
                st.session_state["include_examples"] = True
        with extra_b:
            if st.button("清空粘贴与示例文章"):
                st.session_state["pasted"] = []
                st.session_state["include_examples"] = False
        files = [(f.name, f.getvalue()) for f in uploaded]
        if st.session_state.get("include_examples"):
            files += [(f.name, f.read_bytes()) for f in sorted((ROOT / "examples").glob("*.txt"))]
        for i, (title, text) in enumerate(st.session_state.get("pasted", [])):
            files.append((f"{i + 1}_{title}.txt", text.encode("utf-8")))
        articles, errors = parse_uploads(files)
        for err in errors:
            st.warning(err)
        if articles:
            st.dataframe(pd.DataFrame({"标题": [a.title for a in articles], "字符数": [len(a.text) for a in articles]}), hide_index=True, width="stretch")
        st.download_button("下载 CSV 导入模板", 'title,text\n示例文章,"在这里粘贴完整正文。"\n'.encode("utf-8-sig"), file_name="articles-template.csv", mime="text/csv")

    with st.expander("② 选择模型与计算范围", expanded=True):
        from token_atlas.models import MODEL_PRESETS
        # Presets map descriptive labels to a standard Hugging Face model ID.
        labels = list(MODEL_PRESETS)
        chosen_models = st.multiselect("模型（最多两个，依次计算以节省内存）", labels,
                                       default=labels[:1], max_selections=2)
        custom = st.text_input("自定义模型 ID 或本地目录（可选，替代上方选择）", placeholder="组织名/模型名")
        model_ids = [custom.strip()] if custom.strip() else [MODEL_PRESETS[k] for k in chosen_models]
        st.caption("请从小模型开始。首次运行会下载权重；同一模型的后续运行使用本地缓存。模型需支持 Transformers AutoModel，且无需运行自定义远程代码。")
        with st.expander("高级计算设置"):
            c1, c2, c3 = st.columns(3)
            with c1:
                layer = st.number_input("上下文层（-1 为最后层）", value=-1, min_value=-200, max_value=200,
                                        help="0 是模型返回的初始表示，可能包含位置等信息，不等于纯输入查表模式。")
                device = st.selectbox("计算设备", ["auto", "cpu", "mps", "cuda"])
            with c2:
                window = st.select_slider("每个上下文窗口的 token 上限", [64, 128, 256, 512, 1024], value=256)
                overlap = st.number_input("相邻窗口重叠 token 数", value=32, min_value=0, max_value=512)
            with c3:
                article_cap = st.number_input("每篇文章最多保留 token", min_value=10, max_value=20000, value=1500, step=100)
                total_cap = st.number_input("整张图最多保留 token", min_value=10, max_value=20000, value=6000, step=100)
            revision = st.text_input("模型 revision（可选，可填固定 commit）")
            st.caption("超过上限时明确保留文章开头；总上限会在文章间分配。窗口重新建立局部上下文，重叠 token 只绘制一次。原始 token 数与实际保留数会记录在结果中。")
        settings = dict(model_ids=model_ids, mode=mode, layer=int(layer), device=device, revision=revision,
                        window_size=int(window), overlap=int(overlap), max_tokens_per_article=int(article_cap),
                        max_total_tokens=int(total_cap))
        key = fingerprint(articles, settings)
        go = st.button("生成 token 地图", type="primary", disabled=not articles or not model_ids, width="stretch")
        if go:
            from token_atlas.models import extract_tokens
            results = []
            bar = st.progress(0, text="正在准备模型…")
            for i, model_id in enumerate(model_ids):
                try:
                    bar.progress(i / len(model_ids), text=f"加载 {model_id}；首次下载可能需要几分钟…")
                    handle = cached_model(model_id, device, revision)
                    def progress(done, total, message):
                        bar.progress(min(1.0, (i + done / max(1, total)) / len(model_ids)), text=message)
                    result = extract_tokens(
                        articles, handle, mode=mode, layer=int(layer), window_size=int(window), overlap=int(overlap),
                        max_tokens_per_article=int(article_cap), max_total_tokens=int(total_cap), progress=progress,
                    )
                    results.append((model_id, result))
                    del handle
                    if len(model_ids) > 1:
                        import gc
                        cached_model.clear()
                        gc.collect()
                except Exception as exc:
                    st.error(f"{model_id}：{exc}")
            bar.empty()
            st.session_state["computed"] = (key, results, articles)
        previous = st.session_state.get("computed")
        if previous and previous[0] == key:
            runs = previous[1]
        elif previous:
            st.info("文章或模型设置已改变，点击「生成 token 地图」更新结果。")

if runs:
    st.divider()
    article_labels = {}
    for _, result in runs:
        for token in result.tokens:
            article_labels.setdefault(str(token["article_id"]), token.get("title", token.get("article_title", str(token["article_id"]))))
    article_ids = list(article_labels)
    selection_key = "selection_" + hashlib.sha256("|".join(article_ids).encode()).hexdigest()[:12]
    if selection_key not in st.session_state:
        st.session_state[selection_key] = article_ids
    left, right = st.columns([3, 1])
    with left:
        st.markdown("### ③ 选择要显示的文章")
    with right:
        if st.button("显示全部文章", width="stretch"):
            st.session_state[selection_key] = article_ids
    visible = st.multiselect("文章列表（可以只选一篇）", article_ids, key=selection_key,
                             format_func=lambda x: f"{article_labels[x]} · {x[-6:]}")
    connect = st.checkbox("按原文顺序连接相邻 token", value=False)
    st.caption("筛选只改变显示，坐标与颜色保持不变。单击图例隐藏文章，双击可聚焦；下载范围以文章列表为准。")
    if len(runs) > 1:
        st.info("每个模型独立拟合坐标。不同图的方向、尺度和 token 划分不一致，请比较分布与邻近关系。")
    columns = st.columns(len(runs))
    for idx, ((label, result), column) in enumerate(zip(runs, columns)):
        with column:
            st.markdown(f"#### {label}")
            a, b, c = st.columns(3)
            a.metric("token", f"{len(result.tokens):,}")
            b.metric("向量维度", f"{result.vectors.shape[1]:,}")
            c.metric("表示", "上下文" if mode == "contextual" else "输入层")
            try:
                with st.spinner(f"正在计算 {algorithm} 投影…"):
                    projection = cached_projection(result.vectors, algorithm, options)
                title = f"{label} · {mode_label} · {projection.metadata['effective_algorithm']}"
                fig = build_figure(result.tokens, projection.coordinates, article_ids=visible,
                                   title="", connect_tokens=connect)
                st.plotly_chart(fig, width="stretch", key=f"plot_{idx}", config={"displaylogo": False, "scrollZoom": True})
                for warning in projection.metadata.get("warnings", []):
                    st.caption(warning)
                for warning in result.metadata.get("warnings", []):
                    st.warning(warning)
                if not visible:
                    st.info("选择至少一篇文章以显示点和导出。")
                with st.expander("下载图像与数据"):
                    st.caption("导出当前文章列表中的可见文章，保留整体坐标范围。HTML 包含 token 文本。")
                    prefix = f"token-atlas-{idx + 1}-{mode}-{algorithm.lower()}"
                    format_choice = st.selectbox("图片格式", ["PNG", "SVG"], key=f"format_{idx}")
                    if st.button("准备图片", key=f"prepare_image_{idx}", disabled=not visible):
                        image_data = cached_image(result.tokens, projection.coordinates, visible, title, connect, format_choice.lower())
                        st.download_button("保存图片", image_data, file_name=f"{prefix}.{format_choice.lower()}",
                                           mime="image/png" if format_choice == "PNG" else "image/svg+xml", key=f"download_image_{idx}", on_click="ignore")
                    st.download_button("保存交互 HTML", figure_html(result.tokens, projection.coordinates,
                                       article_ids=visible, title=title, connect_tokens=connect).encode("utf-8"),
                                       file_name=f"{prefix}.html", mime="text/html", key=f"html_{idx}", disabled=not visible)
                    st.download_button("下载数据包（坐标、参数、HTML）", export_bundle(result, projection.coordinates,
                                       projection.metadata, article_ids=visible, title=title, connect_tokens=connect), file_name=f"{prefix}.zip",
                                       mime="application/zip", key=f"bundle_{idx}", disabled=not visible)
                with st.expander("查看 token 与运行记录"):
                    rows = [t for t in result.tokens if str(t["article_id"]) in visible]
                    st.dataframe(pd.DataFrame(rows), hide_index=True, height=240)
                    st.json({"model": result.metadata, "projection": projection.metadata}, expanded=False)
            except Exception as exc:
                st.error(f"无法生成图像：{exc}")

with st.expander("如何阅读这张图？"):
    st.markdown("""
- **一个点 = 一个 token 的一次出现。** 中文 token 可能是字、词的一部分，或字节片段。
- **输入层**读取纯词表向量；**上下文表示**读取模型指定层的输出。同一个词在不同位置可能获得不同表示。
- 相同输入向量在非线性算法中合并投影后还原，避免产生虚假的上下文差异；重合点仍可在 token 表中查看。
- 同一模型内的文章共同拟合投影，文章本身分别经过模型，不会互相成为上下文。
- 图上聚集不等于某种文风已被证明。模型表征也包含语义、位置、句法等信息；二维投影会丢失信息。
- 超长文章采用有重叠的局部窗口。窗口边界处的上下文会改变，可在运行记录中查看具体设置。
""")
st.markdown('<div class="note">文迹 · Token Atlas &nbsp; / &nbsp; 让模型内部的文字表示变得可探索。</div>', unsafe_allow_html=True)
