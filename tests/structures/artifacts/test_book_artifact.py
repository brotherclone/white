from app.structures.artifacts.book_artifact import BookArtifact, ReactionBookArtifact
from app.structures.artifacts.text_chain_artifact_file import TextChainArtifactFile
from app.structures.concepts.rainbow_table_color import the_rainbow_table_colors
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType


class DummyBook:
    pass


def test_init_sets_thread_and_base_path():
    ba = BookArtifact(
        thread_id="thread_1",
        chain_artifact_file_type=ChainArtifactFileType.MARKDOWN,
    )
    assert ba.thread_id == "thread_1"


def test_excerpts_and_pages_assignment():
    f1 = TextChainArtifactFile(
        text_content="text_1",
        file_name="file_1.md",
        artifact_name="doc_1",
        thread_id="thread_1",
        base_path="",
        rainbow_color=the_rainbow_table_colors["R"],
        chain_artifact_file_type=ChainArtifactFileType.MARKDOWN,
    )
    f2 = TextChainArtifactFile(
        text_content="text_2",
        file_name="file_2.md",
        artifact_name="doc_2",
        thread_id="thread_1",
        base_path="",
        rainbow_color=the_rainbow_table_colors["R"],
        chain_artifact_file_type=ChainArtifactFileType.MARKDOWN,
    )
    ba = BookArtifact(
        excerpts=[f1, f2],
        thread_id="thread_1",
        chain_artifact_file_type=ChainArtifactFileType.MARKDOWN,
    )
    assert ba.excerpts == [f1, f2]
    rb = ReactionBookArtifact(
        thread_id="thread_1",
        pages=[f1],
        chain_artifact_file_type=ChainArtifactFileType.MARKDOWN,
    )
    assert rb.pages == [f1]


def test_reaction_subject_assignment():
    subject = BookArtifact(
        thread_id="thread_1",
        chain_artifact_file_type=ChainArtifactFileType.MARKDOWN,
    )
    rb = ReactionBookArtifact(
        subject=subject,
        thread_id="thread_1",
        chain_artifact_file_type=ChainArtifactFileType.MARKDOWN,
    )
    assert rb.subject is subject
