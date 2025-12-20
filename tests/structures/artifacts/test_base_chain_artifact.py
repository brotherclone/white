import os
from abc import ABC

from app.structures.artifacts.base_artifact import ChainArtifact


class DummyArtifact(ChainArtifact, ABC):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def save_file(self):
        return None

    def flatten(self):
        return {}

    def for_prompt(self) -> str:
        return ""


def test_default_generation(tmp_path):
    a = DummyArtifact(base_path=tmp_path)

    assert a.artifact_id is not None
    assert a.file_name is not None

    ext = a.chain_artifact_file_type.value
    assert a.file_name.startswith(f"{a.artifact_id}_")
    assert a.artifact_name in a.file_name
    assert a.file_name.endswith(f".{ext}")

    expected_path = os.path.join(str(tmp_path), a.thread_id, ext)
    assert a.file_path == expected_path

    assert a.get_artifact_path() == os.path.join(a.file_path, a.file_name)
    assert a.get_artifact_path(with_file_name=False) == os.path.join(a.file_path)


def test_custom_artifact_id_and_name(tmp_path):
    c = DummyArtifact(
        base_path=tmp_path,
        artifact_id="myid",
        artifact_name="sigil",
        rainbow_color_mnemonic_character_value="Z",
    )
    assert c.file_name == f"myid_z_sigil.{c.chain_artifact_file_type.value}"
