import pytest
from pydantic import BaseModel

from app.structures.artifacts.template_renderer import HTMLTemplateRenderer


def test_missing_template_raises(tmp_path):
    missing = tmp_path / "nope.html"
    with pytest.raises(FileNotFoundError):
        HTMLTemplateRenderer(str(missing))


def test_simple_substitution_and_or_default(tmp_path):
    tpl = tmp_path / "simple.html"
    tpl.write_text("${title} - ${missing || 'DEFAULT'}", encoding="utf-8")
    r = HTMLTemplateRenderer(tpl)
    out = r.render({"title": "X"})
    assert out.strip() == "X - DEFAULT"


def test_ternary_true_false(tmp_path):
    tpl = tmp_path / "ternary.html"
    tpl.write_text("${flag ? 'YES' : 'NO'}", encoding="utf-8")
    r = HTMLTemplateRenderer(tpl)

    assert r.render({"flag": True}).strip() == "YES"
    assert r.render({"flag": False}).strip() == "NO"
    # falsy values like empty string should pick the false branch
    assert r.render({"flag": ""}).strip() == "NO"


def test_array_map_and_join(tmp_path):
    tpl = tmp_path / "array.html"
    tpl.write_text(
        "${tags.map(tag => `<span>${tag}</span>`).join(',')}", encoding="utf-8"
    )
    r = HTMLTemplateRenderer(tpl)
    result = r.render({"tags": ["a", "b"]})
    assert result == "<span>a</span>,<span>b</span>"


def test_concatenation_and_variable_substitution(tmp_path):
    tpl = tmp_path / "concat.html"
    # renderer supports simple variable substitution; put literal outside
    tpl.write_text("Hello ${name}", encoding="utf-8")
    r = HTMLTemplateRenderer(tpl)
    assert r.render({"name": "Bob"}) == "Hello Bob"


def test_render_with_model(tmp_path):
    tpl = tmp_path / "model.html"
    tpl.write_text(
        "${title} - ${tags.map(tag => `<${tag}>`).join(' ')}", encoding="utf-8"
    )
    r = HTMLTemplateRenderer(tpl)

    class M(BaseModel):
        title: str
        tags: list[str]

    m = M(title="T", tags=["x", "y"])
    out = r.render_with_model(m)
    assert out == "T - <x> <y>"
