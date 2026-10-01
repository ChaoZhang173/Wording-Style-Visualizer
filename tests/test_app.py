"""UI integration: real demo rendering and a stubbed inference submission."""
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from token_atlas.types import TokenResult

ROOT = Path(__file__).resolve().parents[1]


def test_initial_app_and_input_mode_render_without_errors():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    app.sidebar.radio[1].set_value("输入层 embedding").run()
    assert not app.exception
    if (ROOT / "demo_data" / "manifest.json").exists():
        assert len(app.get("plotly_chart")) >= 1


def test_paste_generate_then_edit_invalidates_previous_result(monkeypatch):
    import token_atlas.models as models

    monkeypatch.setattr(models, "load_model", lambda *a, **kw: object())

    def infer(articles, handle, **kwargs):
        a = articles[0]
        tokens = [dict(article_id=a.id, title=a.title, token=c, token_index=i, window_index=0)
                  for i, c in enumerate(a.text)]
        return TokenResult(np.random.default_rng(4).normal(size=(len(tokens), 8)), tokens, {"mode": kwargs["mode"]})

    monkeypatch.setattr(models, "extract_tokens", infer)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.sidebar.radio[0].set_value("分析我的文章").run()
    app.text_input[0].set_value("测试文章")
    app.text_area[0].set_value("雨落下来。街道很安静。")
    next(b for b in app.button if b.label == "加入文章列表").click().run()
    next(b for b in app.button if b.label == "生成 token 地图").click().run()
    assert not app.exception
    assert not app.error
    assert len(app.get("plotly_chart")) == 1
    app.sidebar.radio[1].set_value("输入层 embedding").run()
    assert not app.exception
    assert len(app.get("plotly_chart")) == 0
    assert any("设置已改变" in i.value for i in app.info)
