import json
import polars as pl

from pathlib import Path

from app.reference.mcp.rows_bud.orange_corpus import OrangeMythosCorpus


TEST_STORY = {
    "headline": "Strange Lights Over Township",
    "date": "2023-08-15",
    "source": "Local Paper",
    "text": "Residents reported unexplained lights near the old mill.",
    "location": "Small Township",
    "tags": ["unexplained", "youth_crime"],
}


def make_corpus(tmp_path: Path) -> OrangeMythosCorpus:
    return OrangeMythosCorpus(tmp_path / "corpus_dir")


def test_add_and_get_story(tmp_path):
    corpus = make_corpus(tmp_path)
    story_id, score = corpus.add_story(**TEST_STORY)
    assert isinstance(story_id, str)
    assert 0.0 <= score <= 1.0

    fetched = corpus.get_story(story_id)
    assert fetched is not None
    assert fetched["story_id"] == story_id
    assert fetched["headline"] == TEST_STORY["headline"]
    assert fetched["date"] == TEST_STORY["date"]
    assert fetched["tags"] == TEST_STORY["tags"]
    assert isinstance(fetched["mythologization_score"], float)


def test_insert_symbolic_object_updates_fields_and_stats(tmp_path):
    corpus = make_corpus(tmp_path)
    story_id, _ = corpus.add_story(**TEST_STORY)

    corpus.insert_symbolic_object(
        story_id=story_id,
        category="artifact",
        description="A carved stone found at the scene",
        updated_text="Residents found a carved stone; lights seen around it.",
    )

    fetched = corpus.get_story(story_id)
    assert fetched["symbolic_object_category"] == "artifact"
    assert fetched["symbolic_object_desc"] == "A carved stone found at the scene"
    assert fetched["text"].startswith("Residents found a carved stone")
    assert fetched["symbolic_object_inserted_at"] is not None

    stats = corpus.get_stats()
    assert stats["total_stories"] == 1
    assert stats["mythologization_status"]["with_symbolic_objects"] == 1


def test_add_gonzo_rewrite_and_search_filters(tmp_path):
    corpus = make_corpus(tmp_path)
    # story A will remain without gonzo (eligible)
    story_a_id, score_a = corpus.add_story(**TEST_STORY)

    # story B will get gonzo
    story_b = TEST_STORY.copy()
    story_b["headline"] = "Bizarre Sighting in Borough"
    story_b["location"] = "Old Borough"
    story_b_id, score_b = corpus.add_story(**story_b)

    corpus.add_gonzo_rewrite(
        story_id=story_b_id,
        gonzo_text="An otherworldly account narrated in first person.",
        perspective="first_person",
        intensity=5,
    )

    # needs_mythologizing should only return story A
    results = corpus.search(needs_mythologizing=True)
    ids = {r["story_id"] for r in results}
    assert story_a_id in ids
    assert story_b_id not in ids

    # search by location (case-insensitive, substring)
    loc_results = corpus.search(location="borough")
    assert any(r["story_id"] == story_b_id for r in loc_results)


def test_export_json_and_training_parquet(tmp_path):
    corpus = make_corpus(tmp_path)
    # Add two stories
    story1_id, _ = corpus.add_story(**TEST_STORY)
    story2 = TEST_STORY.copy()
    story2["headline"] = "Another Unexplained Event"
    story2["location"] = "Small Borough"
    story2_id, _ = corpus.add_story(**story2)

    # make story2 complete for training
    corpus.insert_symbolic_object(
        story_id=story2_id,
        category="object",
        description="Mysterious talisman",
        updated_text=corpus.get_story(story2_id)["text"] + " Now with talisman.",
    )
    corpus.add_gonzo_rewrite(
        story_id=story2_id,
        gonzo_text="Gonzo retelling of the talisman discovery.",
        perspective="omniscient",
        intensity=3,
    )

    # Export JSON
    json_path = Path(corpus.export_json("test_export.json"))
    assert json_path.exists()
    assert json_path.stat().st_size > 0

    # The JSON should be readable (NDJSON or array) - try to load lines if NDJSON
    content = json_path.read_text()
    try:
        # try as array
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # fallback: NDJSON lines
        parsed = [json.loads(line) for line in content.splitlines() if line.strip()]
    assert isinstance(parsed, (list, tuple))
    assert len(parsed) >= 2

    # Export training parquet: should only include fully mythologized stories (story2)
    parquet_path = Path(corpus.export_for_training("test_training.parquet"))
    assert parquet_path.exists()
    df = pl.read_parquet(parquet_path)
    assert len(df) == 1
    assert df["story_id"].to_list()[0] == story2_id
