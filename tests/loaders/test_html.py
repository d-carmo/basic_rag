from pathlib import Path

from rag.loaders.base import DocType
from rag.loaders.html import HtmlLoader


def test_strips_nav_footer_script(tmp_path: Path) -> None:
    html = """
    <html><body>
      <nav>Navigation</nav>
      <main><p>Important content.</p></main>
      <footer>Footer text</footer>
      <script>var x = 1;</script>
      <style>body { color: red; }</style>
    </body></html>
    """
    f = tmp_path / "page.html"
    f.write_text(html)
    docs = HtmlLoader().load(f)

    assert len(docs) == 1
    assert "Important content" in docs[0].text
    assert "Navigation" not in docs[0].text
    assert "Footer text" not in docs[0].text
    assert "var x" not in docs[0].text
    assert "color: red" not in docs[0].text


def test_load_from_string() -> None:
    html = "<html><body><p>Hello from string.</p></body></html>"
    docs = HtmlLoader().load_from_string(html, source_url="https://example.com")
    assert len(docs) == 1
    assert "Hello from string" in docs[0].text
    assert docs[0].metadata.source_url == "https://example.com"
    assert docs[0].metadata.doc_type == DocType.HTML


def test_strips_html_comments() -> None:
    html = "<html><body><!-- secret -->visible text</body></html>"
    docs = HtmlLoader().load_from_string(html)
    assert "secret" not in docs[0].text
    assert "visible text" in docs[0].text


def test_empty_after_stripping_returns_empty() -> None:
    html = "<html><body><nav>only nav content</nav></body></html>"
    docs = HtmlLoader().load_from_string(html)
    assert docs == []


def test_can_handle() -> None:
    loader = HtmlLoader()
    assert loader.can_handle("index.html")
    assert loader.can_handle("page.htm")
    assert not loader.can_handle("doc.txt")
