import os
from copy import deepcopy
from pathlib import Path

import yaml
from hypothesis import given
from hypothesis import strategies as st

from app.structures.manifests.song_proposal import SongProposalIteration

MOCK_DIR = Path(os.getenv("AGENT_MOCK_DATA_PATH", "tests/mocks"))


@given(
    moods=st.lists(st.text(min_size=1, max_size=12), min_size=1, max_size=4),
    genres=st.lists(st.text(min_size=1, max_size=12), min_size=1, max_size=4),
)
def test_counter_proposals_varying_lists(moods, genres):
    files = [
        MOCK_DIR / "black_counter_proposal_after_evp_mock.yml",
        MOCK_DIR / "black_counter_proposal_after_sigil_mock.yml",
        MOCK_DIR / "black_counter_proposal_mock.yml",
        MOCK_DIR / "red_counter_proposal_mock.yml",
    ]
    for p in files:
        with open(p, "r") as f:
            base = yaml.safe_load(f)
        candidate = deepcopy(base)
        candidate["mood"] = moods
        candidate["genres"] = genres
        sp = SongProposalIteration(**candidate)
        assert sp.mood == moods
        assert sp.genres == genres


def test_black_counter_proposal_mocks():
    p = MOCK_DIR / "black_counter_proposal_after_evp_mock.yml"
    with open(p, "r") as f:
        data = yaml.safe_load(f)
        counter_proposal = SongProposalIteration(**data)
        assert counter_proposal.iteration_id == "mock_001"
        assert counter_proposal.bpm == 121
        assert counter_proposal.tempo == "4/4"
        assert counter_proposal.key == "G major"
        assert counter_proposal.title == "EVP influenced proposal"
        assert counter_proposal.mood[0] == "Even"
        assert counter_proposal.genres[0] == "Mock Rock"
        assert (
            counter_proposal.concept
            == "I heard, clearly, the words 'there is no window'. I heard, clearly, the words 'there is no window'. I heard, clearly, the words 'there is no window'. I heard, clearly, the words 'there is no window'."
        )
    p = MOCK_DIR / "black_counter_proposal_after_sigil_mock.yml"
    with open(p, "r") as f:
        data = yaml.safe_load(f)
        counter_proposal = SongProposalIteration(**data)
        assert counter_proposal.iteration_id == "mock_002"
        assert counter_proposal.bpm == 100
        assert counter_proposal.tempo == "6/4"
        assert counter_proposal.key == "E minor"
        assert counter_proposal.title == "Sigil influenced proposal"
        assert counter_proposal.mood[0] == "Even"
        assert counter_proposal.genres[0] == "Mock Rock"
        assert (
            counter_proposal.concept
            == "Charged up and ready to forget.  Charged up and ready to forget. Charged up and ready to forget. Charged up and ready to forget. Charged up and ready to forget. Charged up and ready to forget."
        )
    p = MOCK_DIR / "black_counter_proposal_mock.yml"
    with open(p, "r") as f:
        data = yaml.safe_load(f)
        counter_proposal = SongProposalIteration(**data)
        assert counter_proposal.iteration_id == "mock_003"
        assert counter_proposal.bpm == 120
        assert counter_proposal.tempo == "18/21"
        assert counter_proposal.key == "G major"
        assert counter_proposal.title == "Black Counter Proposal"
        assert counter_proposal.mood[0] == "Even"
        assert counter_proposal.genres[0] == "Mock Rock"
        assert (
            counter_proposal.concept
            == "This is a mock counter proposal for a musical piece, it gets passed back to the White Agent to refine and finalize the composition."
        )


def test_red_counter_proposal_mocks():
    p = MOCK_DIR / "red_counter_proposal_mock.yml"
    with open(p, "r") as f:
        data = yaml.safe_load(f)
        counter_proposal = SongProposalIteration(**data)
        assert counter_proposal.iteration_id == "mock_008"
        assert counter_proposal.bpm == 110
        assert counter_proposal.tempo == "4/4"
        assert counter_proposal.key == "G# minor"
        assert counter_proposal.title == "Red Counter Proposal"
        assert counter_proposal.mood[0] == "Even"
        assert counter_proposal.genres[0] == "Mock Rock"
        assert (
            counter_proposal.concept
            == "This is a mock counter proposal for a musical piece, it gets passed back to the White Agent to refine and finalize the composition."
        )
