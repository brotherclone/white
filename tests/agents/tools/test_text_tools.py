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
