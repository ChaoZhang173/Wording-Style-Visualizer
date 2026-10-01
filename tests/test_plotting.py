import io
import json
import zipfile

import numpy as np
import pytest

from token_atlas.plotting import build_figure, coordinates_csv, export_bundle, export_image, figure_html
from token_atlas.types import TokenResult


@pytest.fixture
def sample():
    tokens = [
        {"article_id": "a", "title": "First", "token": "A", "token_index": 0, "window_index": 0},
        {"article_id": "a", "title": "First", "token": "B", "token_index": 1, "window_index": 0},
        {"article_id": "a", "title": "First", "token": "D", "token_index": 3, "window_index": 0},
        {"article_id": "a", "title": "First", "token": "E", "token_index": 4, "window_index": 1},
        {"article_id": "b", "title": "Second", "token": "C", "token_index": 0, "window_index": 0},
    ]
    return tokens, np.array([[0, 0], [1, 1], [2, 2], [3, 3], [10, 10]], dtype=float)


def test_filter_preserves_coordinates_colours_and_bounds(sample):
    tokens, xy = sample
    all_articles = build_figure(tokens, xy)
    selected = build_figure(tokens, xy, ["b"])
    assert len(selected.data) == 1
    assert selected.data[0].marker.color == all_articles.data[1].marker.color
    np.testing.assert_array_equal(selected.data[0].x, [10])
    assert selected.layout.xaxis.range == all_articles.layout.xaxis.range
    assert selected.layout.yaxis.range == all_articles.layout.yaxis.range
    assert selected.layout.uirevision == all_articles.layout.uirevision
    assert not build_figure(tokens, xy, []).data


def test_lines_do_not_bridge_sampling_gaps_or_windows(sample):
    tokens, xy = sample
    figure = build_figure(tokens, xy, connect_tokens=True)
    lines = [trace for trace in figure.data if trace.mode == "lines"]
    assert len(lines) == 1
    assert list(lines[0].x) == [0, 1, None]
    assert lines[0].showlegend is False
    assert sum(trace.mode == "markers" for trace in figure.data) == 2


def test_untrusted_titles_and_tokens_are_escaped(sample):
    tokens, xy = sample
    attack = '</script><script>alert("bad")</script>'
    tokens[0] = dict(tokens[0], title=attack, token="<img src=x onerror=alert(1)>")
    figure = build_figure(tokens, xy, title=attack)
    assert "<script>" not in figure.layout.title.text
    assert "&lt;" in figure.data[0].customdata[0][0]
    document = figure_html(tokens, xy, title=attack)
    assert attack not in document
    assert "cdn.plot.ly/plotly-" not in document


def test_hover_includes_escaped_exact_text_and_context(sample):
    tokens, xy = sample
    tokens[0] = dict(tokens[0], text="雨\n", excerpt='<script>alert("雨")</script>\t风')
    figure = build_figure(tokens, xy)
    hover = figure.data[0].customdata[0]
    assert hover[4] == "雨↵"
    assert hover[5] == "&lt;script&gt;alert(&quot;雨&quot;)&lt;/script&gt;⇥风"
    assert "%{customdata[4]}" in figure.data[0].hovertemplate
    assert "%{customdata[5]}" in figure.data[0].hovertemplate


def test_bundle_keeps_exact_strings_in_json_and_neutralizes_csv_formulas(sample):
    tokens, xy = sample
    tokens[0] = dict(tokens[0], token="=1+2", title="+title")
    result = TokenResult(np.ones((5, 3)), tokens, {"model_id": "sample", "layer": -1})
    content = export_bundle(result, xy, {"method": "PCA", "seed": 42}, ["a"], title="Current view", connect_tokens=True)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        assert set(archive.namelist()) == {"figure.html", "coordinates.csv", "tokens.json", "metadata.json"}
        records = json.loads(archive.read("tokens.json"))
        assert len(records) == 4 and records[0]["token"] == "=1+2"
        assert "'=1+2" in archive.read("coordinates.csv").decode("utf-8-sig")
        metadata = json.loads(archive.read("metadata.json"))
        assert metadata["total_token_count"] == 5
        assert metadata["exported_token_count"] == 4
        assert metadata["projection"]["seed"] == 42
        assert metadata["view"] == {"title": "Current view", "connect_tokens": True}


def test_mismatched_or_nonfinite_coordinates_rejected(sample):
    tokens, xy = sample
    with pytest.raises(ValueError):
        build_figure(tokens, xy[:2])
    xy[0, 0] = np.nan
    with pytest.raises(ValueError):
        coordinates_csv(tokens, xy)


@pytest.mark.parametrize("format,signature", [("png", b"\x89PNG"), ("svg", b"<?xml")])
def test_images_export_without_browser(sample, format, signature):
    tokens, xy = sample
    data = export_image(tokens, xy, title="Token Atlas", format=format, connect_tokens=True)
    assert data.startswith(signature)
    assert len(data) > 1000
