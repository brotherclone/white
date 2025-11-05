import os
from pathlib import Path

from app.agents.tools import text_tools as tt


class SimpleArtifact:
    def __init__(
        self,
        base_path,
        thread_id,
        artifact_name,
        artifact_id,
        chain_artifact_file_type,
        text_content,
        rainbow_color=None,
    ):
        self.base_path = base_path
        self.thread_id = thread_id
        self.artifact_name = artifact_name
        self.artifact_id = artifact_id
        self.chain_artifact_file_type = chain_artifact_file_type
        self.text_content = text_content

    def get_artifact_path(self, with_file_name=True):
        # mimic expected behavior
        p = Path(self.base_path) / self.thread_id
        if with_file_name:
            return str(p / f"{self.artifact_name}.md")
        return str(p)


def test__to_primitive_basic_types():
    assert tt._to_primitive(1) == 1
    assert tt._to_primitive("x") == "x"
    assert tt._to_primitive([1, 2]) == [1, 2]
    d = {"a": 1}
    assert tt._to_primitive(d) == {"a": 1}


def test_save_artifact_file_to_md_writes_file(tmp_path):
    base = tmp_path / "chain_artifacts"
    art = SimpleArtifact(str(base), "thread1", "name", "id", None, "hello world")
    # call function under test
    tt.save_artifact_file_to_md(art)
    # ensure file exists and contains text
    expected = Path(art.get_artifact_path())
    assert expected.exists()
    content = expected.read_text(encoding="utf-8")
    assert "hello world" in content

    # cleanup
    expected.unlink()
    if expected.parent.exists():
        expected.parent.rmdir()
    if base.exists():
        os.rmdir(str(base))
