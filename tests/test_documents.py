import csv
from concurrent.futures import ThreadPoolExecutor
import io
import zipfile

from token_atlas.documents import parse_uploads


def archive(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as output:
        for name, data in files:
            output.writestr(name, data)
    return buffer.getvalue()


def test_text_preserves_whitespace_and_duplicate_titles():
    source = "  雨来了！\r\n\n风也来了。\t "
    files = [("same.txt", source.encode()), ("same.txt", source.encode())]
    articles, errors = parse_uploads(files)
    assert not errors
    assert articles[0].text == source
    assert articles[0].title == articles[1].title == "same"
    assert articles[0].id != articles[1].id
    assert articles == parse_uploads(files)[0]


def test_csv_chinese_headers_multiline_and_bad_row_keeps_good_rows():
    data = '标题,正文,作者\n甲,"第一行，\n第二行。",张三\n空,,李四\n乙,下一篇,王五\n'
    articles, errors = parse_uploads([("articles.csv", data.encode("utf-8-sig"))])
    assert [article.title for article in articles] == ["甲", "乙"]
    assert articles[0].text == "第一行，\n第二行。"
    assert "张三" in articles[0].source
    assert len(errors) == 1


def test_invalid_sibling_does_not_discard_valid_file():
    articles, errors = parse_uploads([("empty.txt", b" "), ("good.md", "有效。".encode()), ("bad.csv", b"one,two\na,b")])
    assert [article.title for article in articles] == ["good"]
    assert len(errors) == 2


def test_zip_accepts_nested_folders_but_never_nested_archives():
    data = archive([("folder/first.txt", "第一篇"), ("nested.zip", archive([("second.txt", "第二篇")])), ("__MACOSX/meta", b"ignored")])
    articles, errors = parse_uploads([("batch.zip", data)])
    assert [article.title for article in articles] == ["first"]
    assert len(errors) == 1 and "嵌套" in errors[0]


def test_zip_path_traversal_is_rejected():
    data = archive([("../unsafe.txt", b"bad"), ("safe.txt", b"good")])
    articles, errors = parse_uploads([("batch.zip", data)])
    assert not articles
    assert "不安全" in errors[0]


def test_zip_inflated_limit(monkeypatch):
    monkeypatch.setattr("token_atlas.documents.MAX_ARCHIVE_BYTES", 20)
    articles, errors = parse_uploads([("batch.zip", archive([("large.txt", "x" * 21)]))])
    assert not articles and errors


def test_docx_retains_paragraphs_tabs_and_line_breaks():
    xml = '''<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
    <w:p><w:r><w:t xml:space="preserve"> 开头 </w:t><w:tab/><w:t>！</w:t><w:br/><w:t>换行</w:t></w:r></w:p>
    <w:p><w:r><w:t>第二段</w:t></w:r></w:p></w:body></w:document>'''
    articles, errors = parse_uploads([("article.docx", archive([("word/document.xml", xml)]))])
    assert not errors
    assert articles[0].text == " 开头 \t！\n换行\n第二段"


def test_docx_entity_document_rejected():
    xml = '<!DOCTYPE x [<!ENTITY xx "expansion">]><x>&xx;</x>'
    articles, errors = parse_uploads([("article.docx", archive([("word/document.xml", xml)]))])
    assert not articles and "实体" in errors[0]


def test_gb18030_and_utf16_are_readable():
    text = "这是一篇中文文章。"
    articles, errors = parse_uploads([("gb.txt", text.encode("gb18030")), ("utf16.txt", text.encode("utf-16"))])
    assert not errors
    assert all(article.text == text for article in articles)


def test_concurrent_large_csv_imports_restore_process_limit():
    original_limit = csv.field_size_limit()
    text = "长" * 150_000
    data = f"title,text\n标题,{text}\n".encode()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: parse_uploads([("large.csv", data)]), range(12)))
    assert all(not errors and articles[0].text == text for articles, errors in results)
    assert csv.field_size_limit() == original_limit


def test_csv_limit_restored_after_parser_error(monkeypatch):
    original_limit = csv.field_size_limit()
    monkeypatch.setattr("token_atlas.documents.MAX_ARTICLE_CHARS", 10)
    articles, errors = parse_uploads([("too_long.csv", b"title,text\nA,1234567890123456\n")])
    assert not articles and errors
    assert csv.field_size_limit() == original_limit
