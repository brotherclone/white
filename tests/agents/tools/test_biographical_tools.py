from app.agents.tools import biographical_tools as bt


def make_sample_biographical_data():
    return {
        "years": {
            "1993": {
                "world_events": {
                    "major": ["Event A", "Event B"],
                    "cultural": ["Culture X"],
                    "technology": ["Tech Y"],
                },
                "personal_context": {
                    "choice_points": ["chose_band", "moved_city"],
                    "influences": ["mentor", "scene"],
                    "emotional_landscape": "nostalgic, reflective",
                },
            }
        },
        "quantum_analysis_prompts": {
            "global_what_ifs": [
                "If the world had turned differently...",
                "Imagine an alternate geopolitics",
            ],
            "personal_what_ifs": [
                "What if you hadn't left school?",
                "What if you'd stayed in town?",
            ],
        },
        "song_inspiration_templates": {
            "experimental": {
                "concept": "A {year} inspired sound collage",
                "musical_approach": "lo-fi textures",
                "lyrical_approach": "fragmented memory",
            }
        },
    }


def test_load_biographical_data_missing_file():
    data = bt.load_biographical_data("/this/path/does/not/exist.yml")
    assert isinstance(data, dict)
    assert set(data.keys()) >= {
        "years",
        "quantum_analysis_prompts",
        "song_inspiration_templates",
    }
    assert (
        data["years"] == {}
        and data["quantum_analysis_prompts"] == {}
        and data["song_inspiration_templates"] == {}
    )


def test_get_year_analysis_happy_path():
    sample = make_sample_biographical_data()
    result = bt.get_year_analysis(1993, biographical_data=sample)
    assert isinstance(result, dict)
    assert result.get("year") == 1993
    assert "year_data" in result and result["year_data"] == sample["years"]["1993"]
    assert "what_if_scenarios" in result
    what_if_scenarios = result["what_if_scenarios"]
    assert (
        "global_what_ifs" in what_if_scenarios
        and "personal_what_ifs" in what_if_scenarios
    )
    assert (
        any(
            "chose_band" in s or "moved_city" in s
            for s in what_if_scenarios["personal_what_ifs"]
        )
        or len(what_if_scenarios["personal_what_ifs"]) >= 1
    )
    assert "cascade_analysis" in result and "quantum_metrics" in result
    ca = result["cascade_analysis"]
    assert "rebracketing_intensity" in ca and 0.0 <= ca["rebracketing_intensity"] <= 1.0
    qm = result["quantum_metrics"]
    assert qm["choice_point_density"] == 2
    assert "taped_over_coefficient" in qm
    si = result["song_inspiration"]
    assert any(
        str(1993) in s.get("title", "") or "Frequency 1993" in s.get("title", "")
        for s in si
    )


def test_generate_what_if_scenarios_limits_and_types():
    sample = make_sample_biographical_data()
    year_data = sample["years"]["1993"]
    prompts = sample["quantum_analysis_prompts"]
    what_if_scenarios = bt.generate_what_if_scenarios(year_data, prompts)
    assert isinstance(what_if_scenarios, dict)
    assert set(what_if_scenarios.keys()) == {"global_what_ifs", "personal_what_ifs"}
    assert len(what_if_scenarios["global_what_ifs"]) <= 4
    assert len(what_if_scenarios["personal_what_ifs"]) <= 4
    for lst in (
        what_if_scenarios["global_what_ifs"],
        what_if_scenarios["personal_what_ifs"],
    ):
        for item in lst:
            assert isinstance(item, str) and item.strip()


def test_calculate_quantum_metrics_edge_case_zero_choice_points():
    year_data = {"personal_context": {"choice_points": [], "influences": []}}
    metrics = bt.calculate_quantum_metrics(year_data)
    assert metrics["choice_point_density"] == 0
    assert metrics["influence_complexity"] == 0
    assert metrics["narrative_malleability"] == 0
    assert metrics["temporal_significance"] == "low"
    assert metrics["taped_over_coefficient"] == 0


def test_explore_alternate_timeline_with_monkeypatch(monkeypatch):
    sample = make_sample_biographical_data()

    def fake_loader(path=None):
        return sample

    monkeypatch.setattr(bt, "load_biographical_data", fake_loader)
    out = bt.explore_alternate_timeline(1993, 0)
    assert isinstance(out, dict)
    assert out.get("year") == 1993
    assert (
        out.get("choice_point")
        == sample["years"]["1993"]["personal_context"]["choice_points"][0]
    )
    assert "song_concept" in out and "title" in out["song_concept"]
    out2 = bt.explore_alternate_timeline(1993, 99)
    assert isinstance(out2, dict)
    assert "error" in out2 and "available_choice_points" in out2
