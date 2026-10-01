"""Fit one shared two-dimensional projection for all articles in a run."""
from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize as l2_normalize


ALGORITHMS = ("PCA", "SVD", "UMAP", "t-SNE", "PaCMAP")
_PACMAP_LOCK = Lock()


@dataclass
class Projection:
    coordinates: np.ndarray
    metadata: dict[str, Any]


def project(
    vectors: np.ndarray,
    algorithm: str = "PCA",
    *,
    seed: int = 42,
    normalize: bool = True,
    perplexity: float = 30,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> Projection:
    if algorithm not in ALGORITHMS:
        raise ValueError(f"未知降维算法：{algorithm}")
    x = np.asarray(vectors, dtype=np.float32)
    if x.ndim != 2 or not x.size or not np.isfinite(x).all():
        raise ValueError("向量必须是非空且不含无效数值的二维矩阵。")
    if perplexity <= 0 or n_neighbors < 2 or not 0 <= min_dist <= 1:
        raise ValueError("请检查 perplexity、邻居数和 min_dist 的设置。")
    if normalize:
        x = l2_normalize(x)
    info: dict[str, Any] = {
        "algorithm": algorithm, "seed": seed, "normalize": normalize,
        "input_shape": list(x.shape), "warnings": [],
        "metric": "euclidean", "fit_scope": "all articles together",
    }
    # A constant cloud has no direction to project. Do not fabricate separation.
    if len(x) == 1 or np.all(x == x[0]):
        info["effective_algorithm"] = "常量投影"
        info["unique_vectors"] = 1
        info["warnings"].append("只有一个不同的向量，所有点显示在原点。")
        return Projection(np.zeros((len(x), 2), dtype=np.float32), info)

    inverse = None
    if algorithm in ("UMAP", "t-SNE", "PaCMAP"):
        # Duplicate static token embeddings must not acquire artificial context.
        unique, inverse = np.unique(x, axis=0, return_inverse=True)
        info["unique_vectors"] = len(unique)
        if len(unique) < len(x):
            info["warnings"].append(
                "非线性投影将相同向量合并计算，再还原各次出现；重复次数不作为邻域权重。"
            )
        x = unique

    actual = algorithm
    if len(x) < 4 and algorithm in ("UMAP", "t-SNE", "PaCMAP"):
        actual = "PCA"
        info["warnings"].append("不同向量少于 4 个，自动使用 PCA。")
    elif len(x) < 8 and algorithm == "PaCMAP":
        actual = "PCA"
        info["warnings"].append("PaCMAP 至少需要 8 个不同向量以建立完整的近邻、中距离和远距离配对，自动使用 PCA。")
    info["effective_algorithm"] = actual
    dims = min(2, x.shape[0], x.shape[1])

    if actual == "PCA":
        reducer = PCA(n_components=dims, svd_solver="full")
        z = reducer.fit_transform(x)
        info["explained_variance_ratio"] = reducer.explained_variance_ratio_.tolist()
        info["centered"] = True
    elif actual == "SVD":
        if x.shape[1] == 1:
            z = x.copy()
        else:
            reducer = TruncatedSVD(n_components=dims, random_state=seed)
            z = reducer.fit_transform(x)
        info["centered"] = False
    elif actual == "t-SNE":
        if x.shape[1] > 50 and len(x) > 50:
            x = PCA(n_components=50, random_state=seed).fit_transform(x)
            info["pre_pca_dimensions"] = 50
        effective_perplexity = min(float(perplexity), max(1.0, (len(x) - 1) / 3))
        info["perplexity"] = effective_perplexity
        z = TSNE(
            n_components=2, perplexity=effective_perplexity, random_state=seed,
            init="pca" if x.shape[1] >= 2 else "random", learning_rate="auto",
            max_iter=1000,
        ).fit_transform(x)
    elif actual == "UMAP":
        try:
            from umap import UMAP
        except ImportError as exc:
            raise ValueError("UMAP 尚未安装，请重新运行启动器以补齐依赖。") from exc
        neighbors = min(n_neighbors, len(x) - 1)
        info.update(n_neighbors=neighbors, min_dist=min_dist)
        z = UMAP(
            n_components=2, n_neighbors=neighbors, min_dist=min_dist,
            metric="euclidean", random_state=seed, n_jobs=1,
        ).fit_transform(x)
    else:
        try:
            import pacmap
        except ImportError as exc:
            raise ValueError("PaCMAP 是可选算法。请安装 pacmap，或先选择其他算法。") from exc
        # Default pair ratios are 1 : 0.5 : 2. Keep every category nonempty
        # and below the sample count so PaCMAP cannot silently resize them.
        neighbors = min(n_neighbors, max(2, int((len(x) - 1) / 3.5)))
        # PaCMAP stores its random seed at module scope. Isolate concurrent
        # Streamlit sessions so each run uses the seed recorded in metadata.
        with _PACMAP_LOCK:
            reducer = pacmap.PaCMAP(
                n_components=2, n_neighbors=neighbors, random_state=seed,
                apply_pca=True,
            )
            z = reducer.fit_transform(x, init="pca" if x.shape[1] >= 2 else "random")
        info.update(n_neighbors=int(reducer.n_neighbors), n_mid_near=int(reducer.n_MN), n_far=int(reducer.n_FP))

    if z.shape[1] < 2:
        z = np.pad(z, ((0, 0), (0, 2 - z.shape[1])))
    if inverse is not None:
        z = z[inverse]
    z = np.asarray(z, dtype=np.float32)
    if not np.isfinite(z).all():
        raise ValueError("降维产生了无效数值，请尝试 PCA 或调整参数。")
    return Projection(z, info)
