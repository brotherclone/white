import os
from pathlib import Path

import yaml

from app.structures.manifests.song_proposal import SongProposalIteration

MOCK_DIR = Path(os.getenv("AGENT_MOCK_DATA_PATH", "tests/mocks"))


def test_initial_archetypal_proposal_mocks():
    p = MOCK_DIR / "white_initial_proposal_archetypal_mock.yml"
    with open(p, "r") as f:
        data = yaml.safe_load(f)
        proposal = SongProposalIteration(**data)
    assert isinstance(proposal, SongProposalIteration)
    assert proposal.title == "White Initial Proposal Archetypal Mock"


def test_initial_categorical_proposal_mocks():
    p = MOCK_DIR / "white_initial_proposal_categorical_mock.yml"
    with open(p, "r") as f:
        data = yaml.safe_load(f)
        proposal = SongProposalIteration(**data)
    assert isinstance(proposal, SongProposalIteration)
    assert proposal.title == "White Initial Proposal Categorical Mock"


def test_initial_comparative_proposal_mocks():
    p = MOCK_DIR / "white_initial_proposal_comparative_mock.yml"
    with open(p, "r") as f:
        data = yaml.safe_load(f)
        proposal = SongProposalIteration(**data)
    assert isinstance(proposal, SongProposalIteration)
    assert proposal.title == "White Initial Proposal Comparative Mock"


def test_initial_phenomenological_proposal_mocks():
    p = MOCK_DIR / "white_initial_proposal_phenomenological_mock.yml"
    with open(p, "r") as f:
        data = yaml.safe_load(f)
        proposal = SongProposalIteration(**data)
    assert isinstance(proposal, SongProposalIteration)
    assert proposal.title == "White Initial Proposal Phenomenological Mock"


def test_initial_procedural_proposal_mocks():
    p = MOCK_DIR / "white_initial_proposal_procedural_mock.yml"
    with open(p, "r") as f:
        data = yaml.safe_load(f)
        proposal = SongProposalIteration(**data)
    assert isinstance(proposal, SongProposalIteration)
    assert proposal.title == "White Initial Proposal Procedural Mock"


def test_initial_relational_proposal_mocks():
    p = MOCK_DIR / "white_initial_proposal_relational_mock.yml"
    with open(p, "r") as f:
        data = yaml.safe_load(f)
        proposal = SongProposalIteration(**data)
    assert isinstance(proposal, SongProposalIteration)
    assert proposal.title == "White Initial Proposal Relational Mock"


def test_initial_technical_proposal_mocks():
    p = MOCK_DIR / "white_initial_proposal_technical_mock.yml"
    with open(p, "r") as f:
        data = yaml.safe_load(f)
        proposal = SongProposalIteration(**data)
    assert isinstance(proposal, SongProposalIteration)
    assert proposal.title == "White Initial Proposal Technical Mock"
