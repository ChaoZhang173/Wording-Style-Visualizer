# Wording Style Visualizer · 文迹

![Input token vectors from two pretrained models, projected with UMAP](docs/preview.jpg)

Explore how articles appear inside different language models. Each point represents one token occurrence, colored by article. Compare raw input embeddings with contextual hidden states, then project them into two dimensions.

[Download project ZIP](https://github.com/ChaoZhang173/Wording-Style-Visualizer/archive/HEAD.zip) · [GitHub repository](https://github.com/ChaoZhang173/Wording-Style-Visualizer) · [中文说明](README.md) · [MIT license](LICENSE) · [CSV example](examples/articles.csv)

## Quick start

1. Install **Python 3.12** from [python.org](https://www.python.org/downloads/). Python 3.10 and 3.11 also work. On Windows, select **Add python.exe to PATH** during installation.
2. [Download the project ZIP](https://github.com/ChaoZhang173/Wording-Style-Visualizer/archive/HEAD.zip) and extract the complete folder.
3. Double-click `start.command` on macOS or `start.bat` on Windows. On Linux, run `bash start.sh` inside the project folder.
4. Wait for setup. The app opens in your browser at <http://localhost:8501>.
5. Try the bundled example, then import your articles, choose a model and representation, and generate a map.

The launcher creates `.venv` inside the project and installs both interface and model dependencies. **Initial setup needs internet and can take several minutes.** Model weights are downloaded on first use and cached for later runs. Ordinary use requires no code changes or commercial embedding API key.

An existing compatible `.venv` takes priority, so the launcher also works when the system's default Python is older.

Keep the launcher window open. Press `Ctrl+C` there to stop the app; closing its browser tab does not stop the server. If macOS prevents opening the downloaded launcher directly, open a terminal in the project folder and run `bash start.sh`.

Run Windows commands from the extracted project folder; keep the `.\` prefix when using PowerShell. Quote paths containing spaces, for example `cd "C:\Users\YourName\Downloads\Wording-Style-Visualizer"`. For model inference on macOS, use native Apple Silicon Python. Intel Macs and x86 Python under Rosetta should use the `--light` demo: the required PyTorch versions no longer provide official Intel Mac packages ([PyTorch announcement](https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690)).

### Try the interface without installing model dependencies

The bundled demonstration contains **real precomputed token vectors** from two small English pretrained models: `prajjwal1/bert-tiny` (a bidirectional encoder) and `roneneldan/TinyStories-1M` (a causal language model). Inputs are two original English example articles. Recordings include model IDs, revisions, layers, and window settings.

These small models support quick interaction demos and validation of both representation pipelines; they do not establish how large models judge writing style. The main Qwen model presets remain available when analyzing your own articles. Saved demo vectors only represent the fixed examples and cannot analyze new text. New articles require the model runtime. This is a locally run app; the repository link is not a deployed online demo.

```bash
bash start.sh --light       # macOS / Linux
.\start.bat --light         # Windows
```

Running the launcher normally afterward installs the model dependencies.

## Import articles

- Paste text directly into the interface.
- Drop multiple TXT, Markdown, or DOCX files; filenames become article titles.
- Upload a ZIP containing those formats. Subfolders are supported, nested ZIP files are not.
- Upload CSV with one article per row. `text` is required; `title` and `author` are optional. Chinese column aliases are supported. See [articles.csv](examples/articles.csv). Quote text containing commas or newlines. Excel's **CSV UTF-8** export is suitable.

UTF-8 is recommended; UTF-16 with a byte-order mark and GB18030 are also supported. DOCX import reads body and table text without formatting. Images, scanned PDFs, and legacy DOC files are not supported.

Limits: 20 MB per file, 100 MB per import, up to 1,000 articles, and 50 MB of uncompressed ZIP content. Import limits are not a promise that all tokens fit in memory. Start with a low per-article token limit before increasing the workload.

## Two representations

**Input embeddings:** tokenize an article with the selected model's tokenizer, then look up each token ID in `model.get_input_embeddings()`. This excludes context. Repeated token IDs have identical vectors and overlap in the map.

**Contextual representations:** run the article through the model and extract each token's hidden state at a selected layer. A repeated token usually receives different vectors in different contexts. Causal models use preceding context; bidirectional encoders can use both sides of the current window. Layer semantics depend on the architecture, and the final layer is not necessarily the most informative for writing style.

A token may be a character, word, word fragment, punctuation mark, or whitespace. Tokenizers differ between models, so articles can produce different token counts. Hover over points to inspect tokens and their source positions.

This requires access to model internals. A typical embedding API returning one vector per input text cannot provide these token-level representations. Presets include Qwen 2.5 0.5B, Qwen 3 0.6B, and Chinese BERT. Select up to two models for side-by-side views, or supply one compatible model ID or local path instead. Custom models must support Transformers `AutoModel` and a fast tokenizer without executing remote custom code. Encoder-decoder architectures are not supported in this version.

Contextual layer `-1` means the last layer; `0` means the initial state returned by the model. That initial state may include positional information or other processing, so layer `0` is not necessarily the same as the separate input-embedding mode.

## Read and compare maps

All articles in one run share a model, layer, and **jointly fitted projection**. Hiding articles changes visibility while preserving coordinates and colors. Regenerating an analysis, adding articles, or changing settings can change the layout.

The article selector determines display and export scope. Legend clicks temporarily hide articles; a double-click focuses an article. These legend-only changes do not change export scope. Optional reading-order lines connect adjacent tokens within the same computation window, avoiding artificial connections across window boundaries.

Different models receive separate projections. Their tokenizers, vector spaces, and axis orientations are not aligned: matching coordinates across panels do not imply matching meaning. Compare broader patterns in the same source texts rather than treating points as directly aligned across models.

| Algorithm | Purpose |
|---|---|
| PCA | Centered linear projection; a useful, stable baseline. |
| SVD | Uncentered truncated SVD; useful for comparison with PCA. |
| t-SNE | Emphasizes local neighbors; cluster distances and sizes are not direct semantic measurements. |
| UMAP | Builds a layout from neighborhood relations; neighbor count and minimum distance affect the result. |
| PaCMAP | Balances local and larger-scale relationships; optional installation. |

Algorithms are alternatives, not a chain of successive two-dimensional reductions. High-dimensional input to t-SNE may first be reduced with PCA. Normalization and random seeds are configurable; hold them constant when comparing runs.

For t-SNE, UMAP, and PaCMAP, exactly identical input vectors are deduplicated before fitting, then their coordinates are restored to all occurrences. This prevents duplicated raw embeddings from acquiring artificial differences. Their repeated frequency therefore does not weight nonlinear neighborhood fitting. PCA and SVD retain row frequencies. Too few distinct vectors trigger a reported fallback to PCA.

Install optional PaCMAP with `bash start.sh --pacmap` or `.\start.bat --pacmap`. Combine `--light --pacmap` to use it with precomputed demos without installing model dependencies. If optional installation fails, other algorithms remain usable; rerun normally or with `--light`.

## Long text and practical limits

Long articles are processed in overlapping windows. Each source token is retained once, using its earliest containing window. Positions restart in each window; vectors from overlapping windows are not averaged. Context includes that entire window rather than the entire article. Special tokens participate in computation but are excluded from article points.

Defaults are 256 tokens per window (including special tokens), 32 overlapping tokens, 1,500 retained tokens per article, and 6,000 across all articles. **Limits retain article prefixes, not a random sample of the whole text.** The total budget is shared among articles. The interface allows up to 20,000 tokens per article and in total. Windows may be shortened to fit the model. Original/retained counts and adjustments appear in run metadata. Check these settings when comparing articles, and avoid treating a retained prefix as a full-article distribution.

Contextual mode runs Transformer layers and typically needs more memory and computation than input embeddings. Both modes load a model. Start with a small model, a few articles, and short windows. CPU execution works for compatible small models; larger models may need substantial memory or a GPU. Device `auto` prefers CUDA, then Apple MPS, then CPU. Models default to `.cache/huggingface` inside the project, or the existing `HF_HOME` location if configured.

The map can reflect vocabulary, semantics, syntax, position, and training effects together. **It is an exploratory representation view, not a writing-style score or proof of authorship.** A fixed seed helps repeatability, but library versions, hardware, and numeric precision can still affect results.

Choose PNG or SVG under the download section, prepare the image, then save it. Interactive HTML is self-contained and can be opened offline. The ZIP bundle contains `coordinates.csv`, `tokens.json`, `metadata.json`, and `figure.html`: projected coordinates, token records, and model/projection settings. It does not include model weights or high-dimensional vectors. Formula-like CSV strings are neutralized; JSON preserves exact strings. The app currently cannot reload a data bundle to restore a computation.

Exports include token text. Run metadata may include titles of all articles used to fit the projection, even when only a subset is exported. Articles are processed on the computer running the app, and the launcher binds to localhost. Model downloads contact the model host.

## Troubleshooting

- **Setup or a model download stalls:** check access to Python package indexes and the model host. Rerun to reuse completed downloads. Use `--light` to explore the interface first.
- **Out of memory:** choose a smaller model, shorten windows, or reduce articles and per-article token limits.
- **Overlapping points:** repeated input embeddings are identical. The app does not add random jitter that would invent differences.
- **Port in use:** stop the earlier instance or use `bash start.sh --port 8502` / `.\start.bat --port 8502`.
- **Chinese text appears as boxes in PNG/SVG:** macOS usually provides PingFang and Windows provides Microsoft YaHei. On Linux, install Noto Sans CJK and restart; for example, use `sudo apt install fonts-noto-cjk` on Ubuntu/Debian. The app does not download fonts. Interactive HTML uses fonts available to the browser.
- **Broken environment:** stop the app, rename the project's `.venv` folder, and rerun the launcher. This does not alter source articles.

## Development

Clone the repository, or use the ZIP download above:

```bash
git clone https://github.com/ChaoZhang173/Wording-Style-Visualizer.git
cd Wording-Style-Visualizer
```

```bash
# macOS / Linux
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-models.txt
python -m streamlit run app.py --server.address=127.0.0.1 --browser.gatherUsageStats=false
```

Windows PowerShell or Command Prompt, without needing to activate the environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-models.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address=127.0.0.1 --browser.gatherUsageStats=false
```

Tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check .
```

Regular tests use fixed inputs and model substitutes without downloading large models. Real-model end-to-end checks require the model dependencies and weights. Dependency ranges are not a cross-platform lockfile; record installed versions for reproducible experiments.

To regenerate genuine recordings in `demo_data/` (requires model dependencies and network downloads):

```bash
# Default: two small English models + original English articles
python scripts/build_demo.py --device cpu

# Larger models: Qwen 2.5 0.5B + Chinese BERT + original Chinese articles
python scripts/build_demo.py --full
```

The script rewrites the demo manifest. `--full` switches to the Chinese model/article pair and needs larger downloads and more memory. Regeneration is unnecessary for browsing recordings already included in the repository.

The project is MIT-licensed. Example articles were written for this project and can be used with it. External model weights and dependencies retain their own licenses. The repository includes bilingual documentation, launchers, examples, and an offline test workflow. Its internal Python package name remains `token_atlas`.

References: [Transformers token outputs](https://huggingface.co/docs/transformers/main_classes/output), [scikit-learn decomposition](https://scikit-learn.org/stable/modules/decomposition.html), [UMAP reproducibility](https://umap-learn.readthedocs.io/en/latest/reproducibility.html), [PaCMAP](https://github.com/YingfanWang/PaCMAP).
