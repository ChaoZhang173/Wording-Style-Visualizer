from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from token_atlas.projection import project


def test_pca_preserves_known_rank_two_distances():
    x = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0], [2, 1, 0]], dtype=float)
    z = project(x, "PCA", normalize=False).coordinates
    np.testing.assert_allclose(
        np.linalg.norm(x[:, None] - x[None, :], axis=-1),
        np.linalg.norm(z[:, None] - z[None, :], axis=-1), atol=1e-6,
    )


@pytest.mark.parametrize("algorithm", ["PCA", "SVD", "UMAP", "t-SNE", "PaCMAP"])
def test_constant_vectors_do_not_fabricate_clusters(algorithm):
    p = project(np.ones((6, 8)), algorithm)
    assert p.coordinates.shape == (6, 2)
    assert np.count_nonzero(p.coordinates) == 0
    assert p.metadata["effective_algorithm"] == "常量投影"
    assert p.metadata["unique_vectors"] == 1


def test_nonlinear_small_sample_fallback_keeps_duplicate_occurrences():
    x = np.array([[1, 2], [1, 2], [3, 4]], dtype=float)
    p = project(x, "UMAP")
    assert p.metadata["effective_algorithm"] == "PCA"
    np.testing.assert_array_equal(p.coordinates[0], p.coordinates[1])
    assert len(p.coordinates) == 3


@pytest.mark.parametrize("algorithm", ["PCA", "SVD", "t-SNE"])
def test_projection_reproducible_and_finite(algorithm):
    x = np.random.default_rng(4).normal(size=(25, 10))
    a = project(x, algorithm)
    b = project(x, algorithm)
    np.testing.assert_allclose(a.coordinates, b.coordinates)
    assert np.isfinite(a.coordinates).all()


def test_reject_nonfinite_and_invalid_settings():
    with pytest.raises(ValueError):
        project(np.array([[np.nan, 0]]))
    with pytest.raises(ValueError):
        project(np.ones((3, 2)), perplexity=0)


@pytest.mark.parametrize("count", [4, 7])
def test_pacmap_small_samples_fall_back_without_empty_pair_categories(count):
    x = np.random.default_rng(4).normal(size=(count, 12))
    projection = project(x, "PaCMAP")
    assert projection.metadata["effective_algorithm"] == "PCA"
    assert any("8" in warning for warning in projection.metadata["warnings"])
    assert np.isfinite(projection.coordinates).all()


@pytest.mark.parametrize("count", [10, 50])
def test_pacmap_records_actual_nonempty_pair_counts(count):
    pytest.importorskip("pacmap")
    x = np.random.default_rng(4).normal(size=(count, 12))
    projection = project(x, "PaCMAP")
    info = projection.metadata
    assert info["effective_algorithm"] == "PaCMAP"
    assert info["n_neighbors"] >= 2
    assert info["n_mid_near"] >= 1
    assert info["n_far"] >= 1
    assert info["n_neighbors"] + info["n_mid_near"] + info["n_far"] < count
    assert np.isfinite(projection.coordinates).all()


def test_concurrent_pacmap_runs_keep_their_own_seeds():
    pytest.importorskip("pacmap")
    x = np.random.default_rng(4).normal(size=(20, 12))
    def run(seed):
        return project(x, "PaCMAP", seed=seed).coordinates
    seeds = [3, 7]
    expected = [run(seed) for seed in seeds]
    with ThreadPoolExecutor(max_workers=2) as executor:
        observed = list(executor.map(run, seeds))
    for baseline, concurrent in zip(expected, observed):
        np.testing.assert_allclose(baseline, concurrent)
