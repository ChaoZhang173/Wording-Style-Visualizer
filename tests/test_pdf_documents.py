"""Small PDF fixtures exercise actual extraction without external PDF tools."""
from concurrent.futures import ThreadPoolExecutor
import builtins
import io
import zipfile

import pytest
from pypdf import PdfWriter, get_configuration
from pypdf.errors import DependencyError
from pypdf.generic import (
    ArrayObject,
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
    TextStringObject,
)

from token_atlas.documents import parse_uploads


def pdf_bytes(pages, *, password=None, compress=False, broken_page=None, broken_form=False):
    writer = PdfWriter()
    characters = sorted(set("".join(text or "" for text in pages)))
    cmap = DecodedStreamObject()
    mappings = "\n".join(f"<{ord(char):04X}> <{ord(char):04X}>" for char in characters)
    cmap.set_data((
        "/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
        "/CMapName /TestUnicode def /CMapType 2 def\n"
        "1 begincodespacerange <0000> <FFFF> endcodespacerange\n"
        f"{len(characters)} beginbfchar\n{mappings}\nendbfchar\n"
        "endcmap CMapName currentdict /CMap defineresource pop end end"
    ).encode())
    cid_font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/CIDFontType2"),
        NameObject("/BaseFont"): NameObject("/TestFont"),
        NameObject("/DW"): NumberObject(1000),
        NameObject("/CIDSystemInfo"): DictionaryObject({
            NameObject("/Registry"): TextStringObject("Adobe"),
            NameObject("/Ordering"): TextStringObject("Identity"),
            NameObject("/Supplement"): NumberObject(0),
        }),
    })
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type0"),
        NameObject("/BaseFont"): NameObject("/TestFont"),
        NameObject("/Encoding"): NameObject("/Identity-H"),
        NameObject("/DescendantFonts"): ArrayObject([writer._add_object(cid_font)]),
        NameObject("/ToUnicode"): writer._add_object(cmap),
    })
    font_ref = writer._add_object(font)
    for index, text in enumerate(pages):
        page = writer.add_blank_page(width=612, height=792)
        if text is None:
            continue
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref}),
        })
        stream = DecodedStreamObject()
        content = f"BT /F1 12 Tf 50 700 Td <{text.encode('utf-16-be').hex()}> Tj ET"
        if broken_form:
            content += " /Missing Do"
        stream.set_data(content.encode())
        if compress:
            stream = stream.flate_encode()
        page[NameObject("/Contents")] = (
            NameObject("/InvalidContents") if index == broken_page else writer._add_object(stream)
        )
    if password:
        writer.encrypt(password)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def test_pdf_unicode_pages_keep_order_and_filename_title():
    articles, notices = parse_uploads([("两页文章.PDF", pdf_bytes(["你好，世界。", "Second page."]))])
    assert not notices
    assert len(articles) == 1
    assert articles[0].title == "两页文章"
    assert articles[0].text == "你好，世界。\n\nSecond page."
    assert articles[0].source == "两页文章.PDF"


def test_zip_pdf_and_text_keep_good_siblings():
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("folder/paper.pdf", pdf_bytes(["文章正文"]))
        archive.writestr("broken.pdf", b"not a PDF")
        archive.writestr("valid.txt", "另一篇")
    articles, notices = parse_uploads([("batch.zip", output.getvalue())])
    assert [article.title for article in articles] == ["paper", "valid"]
    assert articles[0].source == "batch.zip/folder/paper.pdf"
    assert len(notices) == 1 and "无法读取 PDF" in notices[0]


@pytest.mark.parametrize("pages", [[], [None], [" ", None]])
def test_blank_pdf_explains_ocr_and_keeps_other_files(pages):
    articles, notices = parse_uploads([("empty.pdf", pdf_bytes(pages)), ("good.txt", b"good")])
    assert len(articles) == 1 and articles[0].title == "good"
    assert len(notices) == 1
    assert "OCR" in notices[0] and "可能" in notices[0]


def test_mixed_pdf_reports_empty_page_numbers_without_inventing_text():
    articles, notices = parse_uploads([("mixed.pdf", pdf_bytes(["First", None, "Third", None]))])
    assert articles[0].text == "First\n\n\n\nThird\n\n"
    assert len(notices) == 1 and "第 2、4 页" in notices[0]
    assert "可能是空白页或扫描图片" in notices[0]


def test_encrypted_pdf_has_unlock_instructions_and_keeps_siblings():
    articles, notices = parse_uploads([
        ("locked.pdf", pdf_bytes(["secret"], password="testing")),
        ("good.pdf", pdf_bytes(["public"])),
    ])
    assert [article.title for article in articles] == ["good"]
    assert len(notices) == 1 and "不加密" in notices[0] and "密码" in notices[0]


def test_corrupt_page_does_not_silently_import_partial_document():
    articles, notices = parse_uploads([("bad.pdf", pdf_bytes(["Good", "Broken"], broken_page=1))])
    assert not articles
    assert "第 2 页无法读取" in notices[0] and "本文件未导入" in notices[0]


def test_encryption_without_optional_crypto_dependency_still_explains_unlock(monkeypatch):
    def unavailable_reader(*args, **kwargs):
        raise DependencyError("cryptography is required for AES algorithm")

    monkeypatch.setattr("pypdf.PdfReader", unavailable_reader)
    articles, notices = parse_uploads([("locked.pdf", b"encrypted")])
    assert not articles and "不加密" in notices[0]


def test_missing_or_outdated_pdf_component_has_restart_instructions(monkeypatch):
    original_import = builtins.__import__

    def missing_pdf_import(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("cannot import name 'apply_configuration'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_pdf_import)
    articles, notices = parse_uploads([("paper.pdf", b"PDF"), ("good.txt", b"Good")])
    assert len(articles) == 1 and articles[0].title == "good"
    assert len(notices) == 1
    assert "关闭正在运行的工具" in notices[0] and "启动器" in notices[0]


def test_swallowed_form_error_is_reported_as_partial_extraction():
    articles, notices = parse_uploads([("partial.pdf", pdf_bytes(["Good"], broken_form=True))])
    assert len(articles) == 1 and "Good" in articles[0].text
    assert len(notices) == 1 and "第 1 页" in notices[0] and "可能不完整" in notices[0]


def test_pdf_page_count_limit(monkeypatch):
    monkeypatch.setattr("token_atlas.documents.MAX_PDF_PAGES", 1)
    articles, notices = parse_uploads([("long.pdf", pdf_bytes(["One", "Two"]))])
    assert not articles and "超过 1 页" in notices[0]


def test_pdf_character_limit_includes_page_separators(monkeypatch):
    monkeypatch.setattr("token_atlas.documents.MAX_ARTICLE_CHARS", 5)
    articles, notices = parse_uploads([("long.pdf", pdf_bytes(["AB", "CD"]))])
    assert not articles and "超过 5 字符" in notices[0]


def test_pdf_compressed_stream_is_bounded(monkeypatch):
    monkeypatch.setattr("token_atlas.documents.MAX_PDF_STREAM_BYTES", 100)
    articles, notices = parse_uploads([("large.pdf", pdf_bytes(["X" * 1000], compress=True))])
    assert not articles and "过大" in notices[0]


def test_pdf_total_decoded_content_is_bounded(monkeypatch):
    monkeypatch.setattr("token_atlas.documents.MAX_PDF_CONTENT_BYTES", 70)
    articles, notices = parse_uploads([("large.pdf", pdf_bytes(["First", "Second"]))])
    assert not articles and "解压后的页面内容过大" in notices[0]


def test_pdf_limits_do_not_leak_between_concurrent_uploads():
    initial_configuration = get_configuration()
    good = pdf_bytes(["Concurrent"])
    files = [("good.pdf", good), ("bad.pdf", b"invalid")]
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: parse_uploads(files), range(8)))
    assert all(len(articles) == len(notices) == 1 for articles, notices in results)
    assert get_configuration() is initial_configuration
