"""Bounded, local article import. Uploaded archives are never extracted to disk."""
from __future__ import annotations

import csv
from contextlib import contextmanager
import hashlib
import io
from pathlib import Path, PurePosixPath
from threading import Lock
import zipfile
from xml.etree import ElementTree

from .types import Article

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_TOTAL_BYTES = 100 * 1024 * 1024
MAX_ARCHIVE_BYTES = 50 * 1024 * 1024
MAX_FILES = 1000
MAX_ARTICLE_CHARS = 2_000_000
SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".docx", ".csv"}
_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_CSV_LOCK = Lock()


@contextmanager
def _article_csv_limit():
    """csv.field_size_limit is process-global, including Streamlit sessions."""
    with _CSV_LOCK:
        previous = csv.field_size_limit()
        csv.field_size_limit(MAX_ARTICLE_CHARS)
        try:
            yield
        finally:
            csv.field_size_limit(previous)


def _decode(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        result = data.decode("utf-16")
    else:
        try:
            result = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            result = data.decode("gb18030")
    if "\x00" in result:
        raise ValueError("文件包含二进制内容，请使用 UTF-8 文本文件。")
    return result


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    if len(members) > MAX_FILES:
        raise ValueError(f"压缩包条目超过 {MAX_FILES} 个。")
    total = sum(item.file_size for item in members)
    if total > MAX_ARCHIVE_BYTES:
        raise ValueError("压缩包解压后超过 50 MB。")
    for item in members:
        path = PurePosixPath(item.filename.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or (path.parts and ":" in path.parts[0]):
            raise ValueError("压缩包包含不安全的文件路径。")
        if item.flag_bits & 0x1:
            raise ValueError("不支持加密压缩包。")
        if item.file_size > MAX_UPLOAD_BYTES:
            raise ValueError(f"压缩包内文件 {item.filename} 超过 20 MB。")
        if item.file_size > 1024 * 1024 and item.file_size / max(item.compress_size, 1) > 250:
            raise ValueError("压缩包压缩比过高，请直接上传原文件。")
    return members


def _docx_text(data: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        _safe_members(archive)
        try:
            xml = archive.read("word/document.xml")
        except KeyError as exc:
            raise ValueError("DOCX 缺少正文。") from exc
    if b"<!DOCTYPE" in xml.upper() or b"<!ENTITY" in xml.upper():
        raise ValueError("不支持带自定义实体的 DOCX。")
    root = ElementTree.fromstring(xml)
    paragraphs = []
    for paragraph in root.iter(f"{_WORD_NS}p"):
        chunks = []
        for node in paragraph.iter():
            if node.tag == f"{_WORD_NS}t":
                chunks.append(node.text or "")
            elif node.tag == f"{_WORD_NS}tab":
                chunks.append("\t")
            elif node.tag in {f"{_WORD_NS}br", f"{_WORD_NS}cr"}:
                chunks.append("\n")
        paragraphs.append("".join(chunks))
    return "\n".join(paragraphs)


def parse_uploads(files: list[tuple[str, bytes]]) -> tuple[list[Article], list[str]]:
    """Read TXT/Markdown/DOCX/CSV and one level of ZIP, retaining valid siblings.

    CSV accepts title/text/author or 标题/正文/作者. Text is never stripped or
    normalized; stable identifiers distinguish even duplicate titles/files.
    DOCX imports the main body including table paragraphs, without formatting.
    """
    candidates: list[tuple[str, str, str]] = []
    errors: list[str] = []
    remaining_bytes = MAX_TOTAL_BYTES
    file_count = 0

    def append(title: str, body: str, source: str) -> None:
        if not body.strip():
            raise ValueError("正文为空。")
        if len(body) > MAX_ARTICLE_CHARS:
            raise ValueError(f"单篇正文超过 {MAX_ARTICLE_CHARS:,} 字符。")
        if len(candidates) >= MAX_FILES:
            raise ValueError(f"一次最多导入 {MAX_FILES} 篇文章。")
        candidates.append((title.strip() or "未命名文章", body, source))

    def read_one(name: str, data: bytes, *, in_archive: bool = False) -> None:
        nonlocal remaining_bytes, file_count
        file_count += 1
        if file_count > MAX_FILES:
            raise ValueError(f"一次最多读取 {MAX_FILES} 个文件。")
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError("单个文件不能超过 20 MB。")
        remaining_bytes -= len(data)
        if remaining_bytes < 0:
            raise ValueError("此次导入内容总量超过 100 MB。")
        suffix = Path(name).suffix.lower()
        if suffix == ".zip":
            if in_archive:
                raise ValueError("请先解开嵌套 ZIP。")
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                members = _safe_members(archive)
                for item in members:
                    parts = PurePosixPath(item.filename).parts
                    if item.is_dir() or "__MACOSX" in parts or Path(item.filename).name.startswith("."):
                        continue
                    child_name = f"{name}/{item.filename}"
                    try:
                        read_one(child_name, archive.read(item), in_archive=True)
                    except (ValueError, OSError, UnicodeError, zipfile.BadZipFile, ElementTree.ParseError, csv.Error, RuntimeError) as exc:
                        errors.append(f"{child_name}：{exc}")
            return
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError("支持 TXT、Markdown、DOCX、CSV 和 ZIP 文件。")
        if suffix == ".docx":
            append(Path(name).stem, _docx_text(data), name)
        elif suffix == ".csv":
            # Keep the process-global CSV limit stable across concurrent imports.
            with _article_csv_limit():
                reader = csv.DictReader(io.StringIO(_decode(data), newline=""))
                if not reader.fieldnames:
                    raise ValueError("CSV 缺少标题行。")
                aliases = {str(key).strip().lower(): key for key in reader.fieldnames}
                text_key = next((aliases[key] for key in ("text", "正文", "内容", "content") if key in aliases), None)
                title_key = next((aliases[key] for key in ("title", "标题", "名称") if key in aliases), None)
                author_key = next((aliases[key] for key in ("author", "作者") if key in aliases), None)
                if text_key is None:
                    raise ValueError("CSV 需要 text 或 正文 列。")
                for row_number, row in enumerate(reader, start=2):
                    if row_number > MAX_FILES + 1:
                        errors.append(f"{name}：CSV 最多读取 {MAX_FILES} 行文章。")
                        break
                    source = f"{name} · 第 {row_number} 行"
                    if author_key and row.get(author_key):
                        source += f" · 作者：{row[author_key]}"
                    try:
                        if None in row:
                            raise ValueError("CSV 列数不匹配；正文中的逗号需要用引号包裹。")
                        title = row.get(title_key, "") if title_key else ""
                        append(title or f"{Path(name).stem} · {row_number - 1}", row.get(text_key) or "", source)
                    except ValueError as exc:
                        errors.append(f"{source}：{exc}")
        else:
            append(Path(name).stem, _decode(data), name)

    for name, data in files:
        try:
            read_one(str(name), bytes(data))
        except (ValueError, OSError, UnicodeError, zipfile.BadZipFile, ElementTree.ParseError, csv.Error, RuntimeError) as exc:
            errors.append(f"{name}：{exc}")

    counts: dict[str, int] = {}
    articles = []
    for title, body, source in candidates:
        digest = hashlib.sha256(f"{source}\0{title}\0{body}".encode("utf-8")).hexdigest()[:16]
        counts[digest] = counts.get(digest, 0) + 1
        article_id = digest if counts[digest] == 1 else f"{digest}-{counts[digest]}"
        articles.append(Article(article_id, title, body, source))
    return articles, errors


def load_examples(path: str | Path | None = None) -> list[Article]:
    """Load bundled example text articles in a repeatable order."""
    folder = Path(path) if path is not None else Path(__file__).resolve().parent.parent / "examples"
    files = [(file.name, file.read_bytes()) for file in sorted(folder.glob("*.txt"))]
    articles, _ = parse_uploads(files)
    return articles
