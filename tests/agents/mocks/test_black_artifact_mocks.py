from pathlib import Path
import yaml
import glob

from app.agents.models.evp_artifact import EVPArtifact
from app.agents.models.sigil_artifact import SigilArtifact
from app.agents.models.reaction_book_chain_artifact import ReactionBookChainArtifact
from app.agents.models.book_data import BookData
from app.agents.models.audio_chain_artifact_file import AudioChainArtifactFile
from app.agents.models.text_chain_artifact_file import TextChainArtifactFile
from app.agents.enums.publisher_type import PublisherType
from app.agents.enums.book_condition import BookCondition
from app.agents.enums.sigil_type import SigilType
from app.agents.enums.sigil_state import SigilState
from app.agents.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.concepts.rainbow_table_color import RainbowTableColor
from tests.agents.mocks.mock_utils import (
    normalize_book_data_enums,
    normalize_bookdata_dict_only,
    normalize_text_page_defaults,
)


def test_evp_artifact_mock():
    path = Path(__file__).resolve().parents[3] / "app" / "agents" / "mocks" / "black_evp_artifact_mock.yml"
    with path.open("r") as f:
        data = yaml.safe_load(f)
    evp = EVPArtifact(**data)
    # strict type checks
    assert isinstance(evp, EVPArtifact)
    assert getattr(evp, "thread_id", None) == "12345"
    assert hasattr(evp, "audio_segments")
    assert isinstance(evp.audio_segments, list)
    # audio segment basic fields and types
    seg = evp.audio_segments[0]
    assert isinstance(seg, AudioChainArtifactFile)
    assert isinstance(getattr(seg, "file_name", None), str) and len(seg.file_name) > 0

    # transcript and mosaic/noise should be their expected types
    assert isinstance(evp.transcript, TextChainArtifactFile)
    assert isinstance(evp.audio_mosiac, AudioChainArtifactFile)
    assert isinstance(evp.noise_blended_audio, AudioChainArtifactFile)


def test_sigil_artifact_mock():
    path = Path(__file__).resolve().parents[3] / "app" / "agents" / "mocks" / "black_sigil_artifact_mock.yml"
    with path.open("r") as f:
        data = yaml.safe_load(f)
    sig = SigilArtifact(**data)
    assert isinstance(sig, SigilArtifact)
    assert getattr(sig, "thread_id") == "mock_thread_001"
    assert getattr(sig, "wish").startswith("I will encode")
    # enum-backed fields should be parsed into their enum values
    assert isinstance(sig.sigil_type, SigilType)
    assert isinstance(sig.activation_state, SigilState)
    # glyph components should be a list of strings
    assert isinstance(sig.glyph_components, list)
    assert all(isinstance(c, str) for c in sig.glyph_components)


def test_red_book_artifact_mock():
    path = Path(__file__).resolve().parents[3] / "app" / "agents" / "mocks" / "red_book_artifact_mock.yml"
    with path.open("r") as f:
        data = yaml.safe_load(f)

    # Normalize nested enums in book_data (YAML uses bare words like VANITY/RECONSTRUCTED)
    normalize_book_data_enums(data)
    bd = data.get("book_data", {}) or {}

    # ReactionBookChainArtifact requires original_book_title and original_book_author - supply from book_data
    data.setdefault("original_book_title", bd.get("title", "Unknown"))
    data.setdefault("original_book_author", bd.get("author", "Unknown"))

    book = ReactionBookChainArtifact(**data)
    assert isinstance(book, ReactionBookChainArtifact)
    assert getattr(book, "thread_id", None) == "MOCK-THREAD-ID-12345"
    assert hasattr(book, "book_data")
    assert isinstance(book.book_data, BookData)
    assert getattr(book.book_data, "title", None) == "Mock Book Title"
    assert isinstance(book.pages, list)
    assert isinstance(book.pages[0], TextChainArtifactFile)
    assert getattr(book.pages[0], "text_content", "").startswith("This is a mock")


def test_red_reaction_book_data_mock():
    path = Path(__file__).resolve().parents[3] / "app" / "agents" / "mocks" / "red_reaction_book_data_mock.yml"
    with path.open("r") as f:
        data = yaml.safe_load(f)

    # Normalize enum-like fields before creating BookData
    normalize_bookdata_dict_only(data)

    bd = BookData(**data)
    assert isinstance(bd, BookData)
    assert getattr(bd, "title", None) == "Mock Book Title"
    assert getattr(bd, "author", None) == "John Doe"
    # enum-backed fields should parse into enums
    assert isinstance(bd.publisher_type, PublisherType)
    assert isinstance(bd.condition, BookCondition)


def test_red_reaction_book_page_1_mock():
    path = Path(__file__).resolve().parents[3] / "app" / "agents" / "mocks" / "red_reaction_book_page_1_mock.yml"
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}

    # provide required fields so the model can validate
    data = normalize_text_page_defaults(data)
    data.setdefault("chain_artifact_file_type", ChainArtifactFileType.MARKDOWN)

    page = TextChainArtifactFile(**data)
    assert isinstance(page, TextChainArtifactFile)
    assert getattr(page, "thread_id", None) == "MOCK_THREAD_020"
    assert getattr(page, "text_content", "").startswith("Page one of the red reaction book.")
    assert getattr(page, "artifact_id", None) == "1234567890"
    assert isinstance(page.file_name, str) and len(page.file_name) > 0


def test_red_reaction_book_page_2_mock_handles_empty():
    path = Path(__file__).resolve().parents[3] / "app" / "agents" / "mocks" / "red_reaction_book_page_2_mock.yml"
    with path.open("r") as f:
        data = yaml.safe_load(f)
    # file may be empty; ensure loader returns None or empty and the test tolerates it
    assert data in (None, {}, [])


# tighten enum equality checks for book_data
def test_red_book_enums_are_expected():
    path = Path(__file__).resolve().parents[3] / "app" / "agents" / "mocks" / "red_book_artifact_mock.yml"
    with path.open("r") as f:
        data = yaml.safe_load(f)

    # use our helper to normalize enum-like fields within the nested book_data dict
    normalize_book_data_enums(data)
    bd = data.get("book_data", {}) or {}

    assert bd.get("publisher_type") is not None
    assert bd.get("condition") is not None
    # explicit equality checks using the actual Enum members
    from app.agents.enums.publisher_type import PublisherType as PT
    from app.agents.enums.book_condition import BookCondition as BC
    assert bd["publisher_type"] == PT.VANITY
    assert bd["condition"] == BC.RECONSTRUCTED


def test_black_to_white_document_synthesis_mock_text_loads():
    path = Path(__file__).resolve().parents[3] / "app" / "agents" / "mocks" / "black_to_white_document_synthesis_mock.yml"
    with path.open("r") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, str)
    assert "peanut butter" in data


def test_black_to_white_rebracket_analysis_mock_text_loads():
    path = Path(__file__).resolve().parents[3] / "app" / "agents" / "mocks" / "black_to_white_rebracket_analysis_mock.yml"
    with path.open("r") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, str)
    assert "re-bracket" in data or "rebracket" in data or "re-brack" in data


def test_all_mock_yml_files_parse():
    """Ensure every .yml file in the mocks directory can be parsed by yaml.safe_load."""
    base = Path(__file__).resolve().parents[3] / "app" / "agents" / "mocks"
    files = sorted(glob.glob(str(base / "*.yml")))
    assert files, "No .yml mock files found"
    for p in files:
        # load each and ensure no exception; empty files may yield None
        with open(p, "r") as f:
            data = yaml.safe_load(f)
        # At minimum the loader should return None or a basic container/string
        assert data is None or isinstance(data, (dict, list, str)), f"Unexpected YAML top-level type for {p}: {type(data)}"
