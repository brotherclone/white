import yaml

from app.structures.manifests.song_proposal import SongProposalIteration

def test_black_counter_proposal_mocks():
    with open("/Volumes/LucidNonsense/White/app/agents/mocks/black_counter_proposal_after_evp_mock.yml", "r") as f:
        data = yaml.safe_load(f)
        counter_proposal = SongProposalIteration(**data)
        assert counter_proposal.iteration_id == "MOCK012"
        assert counter_proposal.bpm == 121
        assert counter_proposal.tempo == "4/4"
        assert counter_proposal.key == "G major"
        assert counter_proposal.title == "EVP influenced proposal"
        assert counter_proposal.mood[0] == "Even"
        assert counter_proposal.genres[0] == "Mock Rock"
        assert counter_proposal.concept == "I heard, clearly, the words 'there is no window'."
    with open("/Volumes/LucidNonsense/White/app/agents/mocks/black_counter_proposal_after_sigil_mock.yml", "r") as f:
        data = yaml.safe_load(f)
        counter_proposal = SongProposalIteration(**data)
        assert counter_proposal.iteration_id == "MOCK011"
        assert counter_proposal.bpm == 100
        assert counter_proposal.tempo == "6/4"
        assert counter_proposal.key == "E minor"
        assert counter_proposal.title == "Sigil influenced proposal"
        assert counter_proposal.mood[0] == "Even"
        assert counter_proposal.genres[0] == "Mock Rock"
        assert counter_proposal.concept == "Charged up and ready to forget."
    with open("/Volumes/LucidNonsense/White/app/agents/mocks/black_counter_proposal_mock.yml", "r") as f:
        data = yaml.safe_load(f)
        counter_proposal = SongProposalIteration(**data)
        assert counter_proposal.iteration_id == "MOCK010"
        assert counter_proposal.bpm == 0
        assert counter_proposal.tempo == "18/21"
        assert counter_proposal.key == "G major"
        assert counter_proposal.title == "Black Counter Proposal"
        assert counter_proposal.mood[0] == "Even"
        assert counter_proposal.genres[0] == "Mock Rock"
        assert counter_proposal.concept == "This is a mock counter proposal for a musical piece, it gets passed back to the White Agent to refine and finalize the composition."

def test_red_counter_proposal_mocks():
    with open("/Volumes/LucidNonsense/White/app/agents/mocks/red_counter_proposal_mock.yml", "r") as f:
        data = yaml.safe_load(f)
        counter_proposal = SongProposalIteration(**data)
        assert counter_proposal.iteration_id == "MOCK015"
        assert counter_proposal.bpm == 110
        assert counter_proposal.tempo == "4/4"
        assert counter_proposal.key == "G# minor"
        assert counter_proposal.title == "Red Counter Proposal"
        assert counter_proposal.mood[0] == "Even"
        assert counter_proposal.genres[0] == "Mock Rock"
        assert counter_proposal.concept == "This is a mock counter proposal for a musical piece, it gets passed back to the White Agent to refine and finalize the composition."