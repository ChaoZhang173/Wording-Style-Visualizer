"""Record genuine model outputs for the included original examples.

Run from the repository root: python scripts/build_demo.py
Downloads model weights on first use; no random/synthetic vectors are substituted.
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402

from token_atlas.documents import parse_uploads  # noqa: E402
from token_atlas.models import extract_tokens, load_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Use Chinese examples with Qwen 0.5B and BERT Chinese (larger downloads).")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    dest = ROOT / "demo_data"
    dest.mkdir(exist_ok=True)
    files = [(p.name, p.read_bytes()) for p in sorted((ROOT / "examples").glob("*.txt"))
             if any("\u4e00" <= c <= "\u9fff" for c in p.read_text("utf-8")) == args.full][:3]
    if len(files) < 2:
        raise RuntimeError("Need at least two original examples.")
    articles, errors = parse_uploads(files)
    if errors:
        raise RuntimeError(errors)
    models = [
        ("qwen25", "Qwen 2.5 · 0.5B", "Qwen/Qwen2.5-0.5B"),
        ("bertzh", "BERT · 中文（双向编码器）", "google-bert/bert-base-chinese"),
    ] if args.full else [
        ("bert_tiny", "BERT Tiny · 英文双向编码器", "prajjwal1/bert-tiny"),
        ("tiny_stories", "TinyStories · 英文因果语言模型", "roneneldan/TinyStories-1M"),
    ]
    manifest = []
    for ident, label, model_id in models:
        print(f"Loading {model_id}", flush=True)
        local = ROOT / ".cache" / "demo-models" / model_id.split("/")[-1]
        handle = load_model(str(local) if (local / "source.json").exists() else model_id, device=args.device)
        if (local / "source.json").exists():
            source = json.loads((local / "source.json").read_text())
            handle.model_id = source["model_id"]
            handle.revision = source["revision"]
        record = {"id": ident, "label": label, "model_id": model_id, "files": {}}
        for mode in ("input", "contextual"):
            result = extract_tokens(articles, handle, mode=mode, layer=-1, window_size=256,
                                    overlap=32, max_tokens_per_article=600, max_total_tokens=1800)
            result.metadata["recording"] = "Real pretrained model outputs; bundled original example articles."
            result.metadata["tokenizer_name"] = model_id
            result.metadata["article_texts"] = [{"id": a.id, "title": a.title, "text": a.text} for a in articles]
            stem = f"{ident}_{mode}"
            np.savez_compressed(dest / f"{stem}.npz", vectors=result.vectors)
            (dest / f"{stem}.json").write_text(json.dumps({"tokens": result.tokens, "metadata": result.metadata}, ensure_ascii=False, indent=2), "utf-8")
            record["files"][mode] = stem
            print(f"Recorded {stem}: {result.vectors.shape}", flush=True)
        manifest.append(record)
        (dest / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
        del handle
        gc.collect()
    print("Demo recordings complete.", flush=True)


if __name__ == "__main__":
    main()
