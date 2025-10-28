import yaml

from app.structures.manifests.song_proposal import SongProposalIteration


# def test_document_synthesis_artifact_mocks():
#     raise Exception ("Not implemented yet")
#
# def test_rebracketing_analysis_artifact_mocks():
#     raise Exception ("Not implemented yet")

def test_initial_archetypal_proposal_mocks():
    with open("/Volumes/LucidNonsense/White/app/agents/mocks/white_initial_proposal_archetypal_mock.yml","r") as f:
        data = yaml.safe_load(f)
        proposal = SongProposalIteration(**data)
    assert isinstance(proposal, SongProposalIteration)
    assert proposal.title == "White Initial Proposal Archetypal Mock"

def test_initial_categorical_proposal_mocks():
    with open("/Volumes/LucidNonsense/White/app/agents/mocks/white_initial_proposal_categorical_mock.yml","r") as f:
        data = yaml.safe_load(f)
        proposal = SongProposalIteration(**data)
    assert isinstance(proposal, SongProposalIteration)
    assert proposal.title == "White Initial Proposal Categorical Mock"

def test_initial_comparative_proposal_mocks():
    with open("/Volumes/LucidNonsense/White/app/agents/mocks/white_initial_proposal_comparative_mock.yml","r") as f:
        data = yaml.safe_load(f)
        proposal = SongProposalIteration(**data)
    assert isinstance(proposal, SongProposalIteration)
    assert proposal.title == "White Initial Proposal Comparative Mock"

def test_initial_phenomenological_proposal_mocks():
    with open("/Volumes/LucidNonsense/White/app/agents/mocks/white_initial_proposal_phenomenological_mock.yml","r") as f:
        data = yaml.safe_load(f)
        proposal = SongProposalIteration(**data)
    assert isinstance(proposal, SongProposalIteration)
    assert proposal.title == "White Initial Proposal Phenomenological Mock"

def test_initial_procedural_proposal_mocks():
    with open("/Volumes/LucidNonsense/White/app/agents/mocks/white_initial_proposal_procedural_mock.yml","r") as f:
        data = yaml.safe_load(f)
        proposal = SongProposalIteration(**data)
    assert isinstance(proposal, SongProposalIteration)
    assert proposal.title == "White Initial Proposal Procedural Mock"

def test_initial_relational_proposal_mocks():
    with open("/Volumes/LucidNonsense/White/app/agents/mocks/white_initial_proposal_relational_mock.yml","r") as f:
        data = yaml.safe_load(f)
        proposal = SongProposalIteration(**data)
    assert isinstance(proposal, SongProposalIteration)
    assert proposal.title == "White Initial Proposal Relational Mock"

def test_initial_technical_proposal_mocks():
    with open("/Volumes/LucidNonsense/White/app/agents/mocks/white_initial_proposal_technical_mock.yml","r") as f:
        data = yaml.safe_load(f)
        proposal = SongProposalIteration(**data)
    assert isinstance(proposal, SongProposalIteration)
    assert proposal.title == "White Initial Proposal Technical Mock"