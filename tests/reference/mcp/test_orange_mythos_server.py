import json
from types import SimpleNamespace
from pathlib import Path

import polars as pl
import pytest

from app.reference.mcp.rows_bud import orange_mythos_server as server


def call_tool(fn, *args, **kwargs):
    """
    Call a server export/tool function. Unwrap common tool wrappers:
    - .func (FunctionTool-like)
    - .fn
    - .run (non-callable wrapper exposing run)
    - .call
    Then invoke the resulting callable.
    """
    target = fn
    if hasattr(target, "func"):
        target = getattr(target, "func")
    elif hasattr(target, "fn"):
        target = getattr(target, "fn")
    elif hasattr(target, "run") and not callable(target):
        target = getattr(target, "run")
    elif hasattr(target, "call") and not callable(target):
        target = getattr(target, "call")

    if not callable(target):
        raise TypeError(
            f"Tool {fn!r} is not callable and has no known unwrap attribute"
        )

    return target(*args, **kwargs)


class FakeCorpus:
    def __init__(self, exports_dir: Path):
        self._stories = {}
        self._next = 1
        self.exports_dir = exports_dir
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.df = []

    def add_story(self, headline, date, source, text, location, tags):
        story_id = f"s{self._next}"
        self._next += 1
        score = 0.75
        story = {
            "story_id": story_id,
            "headline": headline,
            "date": date,
            "source": source,
            "text": text,
            "location": location,
            "tags": tags,
            "mythologization_score": score,
        }
        self._stories[story_id] = story
        return story_id, score

    def get_story(self, story_id):
        return self._stories.get(story_id)

    def search(
        self,
        tags=None,
        min_score=0.5,
        location=None,
        start_date=None,
        end_date=None,
        needs_mythologizing=True,
    ):
        results = []
        for s in self._stories.values():
            if s.get("mythologization_score", 0) < min_score:
                continue
            if location and location.lower() not in s.get("location", "").lower():
                continue
            if needs_mythologizing and s.get("gonzo_text"):
                continue
            results.append(s)
        return results

    def export_json(self, filename):
        path = self.exports_dir / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(list(self._stories.values()), f)
        return str(path)

    def export_for_training(self, filename):
        path = self.exports_dir / filename
        rows = []
        for s in self._stories.values():
            if s.get("symbolic_object_desc") and s.get("gonzo_text"):
                rows.append(
                    {
                        "story_id": s["story_id"],
                        "headline": s["headline"],
                        "text": s["text"],
                        "gonzo_text": s["gonzo_text"],
                    }
                )
        if not rows:
            df = pl.DataFrame(
                {"story_id": [], "headline": [], "text": [], "gonzo_text": []}
            )
        else:
            df = pl.DataFrame(rows)
        df.write_parquet(str(path))
        return str(path)

    def get_stats(self):
        total = len(self._stories)
        with_so = sum(
            1 for s in self._stories.values() if s.get("symbolic_object_desc")
        )
        with_gonzo = sum(1 for s in self._stories.values() if s.get("gonzo_text"))
        return {
            "total_stories": total,
            "mythologization_status": {
                "with_symbolic_objects": with_so,
                "with_gonzo": with_gonzo,
            },
        }

    def get_high_value_stories(self, limit=10):
        uns = [s for s in self._stories.values() if not s.get("gonzo_text")]
        sorted_uns = sorted(
            uns, key=lambda x: x.get("mythologization_score", 0), reverse=True
        )
        return sorted_uns[:limit]

    def analyze_temporal_patterns(self):
        years = {}
        for s in self._stories.values():
            y = s.get("date", "")[:4]
            years[y] = years.get(y, 0) + 1
        return {
            "yearly_distribution": years,
            "peak_year": max(years.items(), key=lambda x: x[1])[0] if years else None,
        }


class FakeAnthropic:
    class _Messages:
        def create(self, model, max_tokens, messages, temperature=None):
            content = ""
            for m in messages:
                content = m.get("content", "") if isinstance(m, dict) else str(m)
                break
            if "Insert this symbolic object" in content or "SYMBOLIC OBJECT" in content:
                text = "Updated article text with a strange artifact inserted."
            elif (
                "Rewrite this Sussex County story" in content
                or "GONZO PARAMETERS" in content
            ):
                text = "Gonzo rewrite: I dove into the pine barrens and everything unraveled."
            else:
                text = "AI response text."
            return SimpleNamespace(content=[SimpleNamespace(text=text)])

    def __init__(self):
        self.messages = FakeAnthropic._Messages()


@pytest.fixture
def fake_env(tmp_path):
    fake = FakeCorpus(tmp_path / "exports")
    return fake


def test_add_story_to_corpus_success(monkeypatch, tmp_path):
    fake = FakeCorpus(tmp_path / "exports")
    monkeypatch.setattr(server, "corpus", fake)
    res = call_tool(
        server.add_story_to_corpus,
        headline="Test",
        date="2023-01-01",
        source="Paper",
        text="Something odd happened.",
        location="Town",
        tags=["weird"],
    )
    assert res["status"] == "added"
    assert "story_id" in res
    assert isinstance(res["score"], float)
