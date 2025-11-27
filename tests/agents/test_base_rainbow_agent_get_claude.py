from app.structures.agents.agent_settings import AgentSettings
from app.structures.agents.base_rainbow_agent import BaseRainbowAgent


class DummyClaude:
    def __init__(self, **kwargs):
        self._constructed_with = kwargs


class ConcreteAgent(BaseRainbowAgent):
    def create_graph(self):
        return None

    def generate_alternate_song_spec(self, agent_state):
        return None

    def export_chain_artifacts(self, agent_state):
        return None


def test_get_claude_uses_settings(monkeypatch):
    # monkeypatch the ChatAnthropic used in the module to our DummyClaude
    import app.structures.agents.base_rainbow_agent as bra

    monkeypatch.setattr(bra, "ChatAnthropic", DummyClaude)

    settings = AgentSettings()
    # override a few values to ensure they're passed
    settings = settings.model_copy(
        update={
            "anthropic_sub_model_name": "submodel",
            "anthropic_api_key": "sekrit",
            "temperature": 0.2,
            "max_retries": 1,
            "timeout": 10,
            "stop": ["\n"],
        }
    )

    agent = ConcreteAgent(settings=settings)
    claude = agent._get_claude()
    assert isinstance(claude, DummyClaude)
    # ensure constructor got the expected keys
    cw = claude._constructed_with
    assert cw["model_name"] == settings.anthropic_sub_model_name
    assert cw["api_key"] == settings.anthropic_api_key
    assert cw["temperature"] == settings.temperature


def test_manifest_fallback_data_getitem_and_get():
    from app.structures.manifests.manifest import Manifest
    from app.structures.music.core.duration import Duration

    m = Manifest(
        bpm=100,
        manifest_id="m5",
        tempo="4/4",
        key="C major",
        rainbow_color="R",
        title="T",
        release_date="2020-01-02",
        album_sequence=1,
        main_audio_file="a.wav",
        TRT=Duration(minutes=0, seconds=1.0),
        vocals=False,
        lyrics=False,
        sounds_like=[],
        structure=[],
        mood=[],
        genres=[],
        lrc_file=None,
        concept="",
        audio_tracks=[],
    )

    # attach a fallback _data dict to simulate older behaviour
    m._data = {"extra_key": "extra_val"}
    assert m["extra_key"] == "extra_val"
    assert ("extra_key" in m) is True
    assert m.get("extra_key") == "extra_val"
    assert m.get("missing", "d") == "d"
