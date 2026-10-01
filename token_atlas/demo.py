"""Read checked-in, real model recordings without loading model weights."""
import json
from pathlib import Path

import numpy as np

from token_atlas.types import TokenResult

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo_data"


def available_demos() -> list[dict]:
    manifest = DEMO_DIR / "manifest.json"
    return json.loads(manifest.read_text("utf-8")) if manifest.exists() else []


def load_demo(recording_id: str, mode: str) -> TokenResult:
    records = {r["id"]: r for r in available_demos()}
    if recording_id not in records:
        raise ValueError("没有找到这个预计算示例。")
    stem = records[recording_id]["files"][mode]
    info = json.loads((DEMO_DIR / f"{stem}.json").read_text("utf-8"))
    with np.load(DEMO_DIR / f"{stem}.npz", allow_pickle=False) as data:
        result = TokenResult(data["vectors"], info["tokens"], info["metadata"])
    result.validate()
    return result
