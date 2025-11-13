import re
from pathlib import Path

from app.structures.artifacts.base_artifact import ChainArtifact


def test_save_md_creates_file_and_returns_path(tmp_path):
    base = tmp_path / "nested" / "dir"
    name = "my_art"
    yml = "```yaml\nkey: value\n```"
    jsn = '```json\n{"k": 1}\n```'

    out = ChainArtifact.save_md(base, name, yml, jsn)
    out_path = Path(out)

    assert out_path.exists()
    assert out_path.suffix == ".md"
    assert out_path.parent == base
    content = out_path.read_text(encoding="utf-8")
    assert f"# Artifact: {name}" in content
    assert yml in content
    assert jsn in content
    assert re.search(rf"{re.escape(name)}_\d{{8}}T\d{{6}}Z\.md$", out_path.name)


def test_save_md_with_default_name_and_pathlib_base(tmp_path):
    base = tmp_path
    yml = "```yaml\nfoo: bar\n```"
    jsn = '```json\n{"x": 2}\n```'

    out = ChainArtifact.save_md(base, None, yml, jsn)
    out_path = Path(out)

    assert out_path.exists()
    assert "# Artifact: artifact" in out_path.read_text(encoding="utf-8")
    assert re.search(r"artifact_\d{8}T\d{6}Z\.md$", out_path.name)


def test_chain_artifact_model_field():
    ca = ChainArtifact(chain_artifact_type="example")
    assert ca.chain_artifact_type == "example"
