from app.structures.artifacts.character_portrait_artifact import (
    CharacterPortraitArtifact,
)
from app.structures.artifacts.image_artifact_file import ImageChainArtifactFile
from app.structures.artifacts.pulsar_palace_encounter_artifact import (
    PulsarPalaceEncounterArtifact,
)
from app.structures.concepts.pulsar_palace_character import (
    PulsarPalaceCharacter,
    PulsarPalaceCharacterBackground,
    PulsarPalaceCharacterDisposition,
    PulsarPalaceCharacterProfession,
)


def test_character_portrait_for_prompt_includes_name_and_image_path(tmp_path):
    img = ImageChainArtifactFile(
        thread_id="t1",
        base_path=str(tmp_path),
        file_path=str(tmp_path / "portrait.png"),
        width=64,
        height=64,
    )

    cp = CharacterPortraitArtifact(
        character_name="Vesper",
        role="Detective",
        pose="hands in pockets",
        description="A tired sleuth",
        image=img,
    )

    out = cp.for_prompt()
    assert "Vesper" in out
    assert "Detective" in out
    assert "portrait.png" in out


def test_create_portrait_sets_portrait_and_portrait_artifact(monkeypatch, tmp_path):
    # Prepare a dummy composite function that returns a png path
    png_path = str(tmp_path / "composite.png")

    def fake_composite(a, b, out_path):
        # ignore inputs, return our png path
        return png_path

    # Patch both the composite helper and the underlying composite_images function
    monkeypatch.setattr(
        "app.structures.concepts.pulsar_palace_character.composite_character_portrait",
        fake_composite,
    )
    monkeypatch.setattr(
        "app.agents.tools.image_tools.composite_images",
        lambda out_path, layers: png_path,
    )

    # Ensure character portrait creation has a valid base path
    monkeypatch.setenv("AGENT_WORK_PRODUCT_BASE_PATH", str(tmp_path))

    # Fake PIL Image.open to return object with size attribute used by create_portrait
    class DummyImage:
        def __init__(self, size=(128, 256)):
            self.size = size

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("PIL.Image.open", lambda p: DummyImage((128, 256)))

    bg = PulsarPalaceCharacterBackground(rollId=1, time=1950, place="Gotham")
    disp = PulsarPalaceCharacterDisposition(rollId=1, disposition="Crazed")
    prof = PulsarPalaceCharacterProfession(rollId=1, profession="Detective")
    # Give image paths so composite_character_portrait is not passed None values
    bg.image_path = str(tmp_path / "bg.png")
    disp.image_path = str(tmp_path / "disp.png")
    prof.image_path = str(tmp_path / "prof.png")

    char = PulsarPalaceCharacter(
        thread_id="thread-1",
        encounter_id="enc-1",
        background=bg,
        disposition=disp,
        profession=prof,
        on_max=5,
        on_current=3,
        off_max=5,
        off_current=1,
    )

    # Call create_portrait which should set portrait and portrait_artifact
    char.create_portrait()

    assert char.portrait is not None
    # file_path is now computed from base_path/thread_id/file_type
    expected_file_path = str(tmp_path / "thread-1" / "png")
    assert char.portrait.file_path == expected_file_path
    assert char.portrait_artifact is not None
    assert char.portrait_artifact.image is char.portrait
    # for_prompt should include character name and image path
    prompt = char.portrait_artifact.for_prompt()
    assert "Crazed Detective" in prompt or "Detective" in prompt
    # Should include the auto-generated filename (artifact_id_color_name.ext)
    assert ".png" in prompt
    assert char.portrait.file_name in prompt


def test_encounter_to_markdown_includes_portrait_line():
    # Build a minimal character with portrait_artifact present
    img = ImageChainArtifactFile(
        thread_id="t2",
        base_path="/tmp",
        file_path="/tmp/char.png",
        width=32,
        height=32,
    )
    cp = CharacterPortraitArtifact(
        character_name="Morg",
        role="Janitor",
        image=img,
    )

    bg = PulsarPalaceCharacterBackground(rollId=1, time=1800, place="Bath")
    disp = PulsarPalaceCharacterDisposition(rollId=1, disposition="Angry")
    prof = PulsarPalaceCharacterProfession(rollId=1, profession="Janitor")

    char = PulsarPalaceCharacter(
        thread_id="t2",
        encounter_id="e2",
        background=bg,
        disposition=disp,
        profession=prof,
        on_max=2,
        on_current=1,
        off_max=2,
        off_current=0,
    )
    char.portrait_artifact = cp

    encounter = PulsarPalaceEncounterArtifact(
        thread_id="t2",
        characters=[char],
        rooms=[],
        story=["It was dark."],
    )

    md = encounter.to_markdown()
    assert "## Characters" in md
    assert "Portrait" in md
    assert "/tmp/char.png" in md or "Portrait of Morg" in md
