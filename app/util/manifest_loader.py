import yaml
import os

from dotenv import load_dotenv

from app.structures.manifests.song_proposal import SongProposal, SongProposalIteration
from app.structures.manifests.manifest import Manifest
from app.structures.concepts.rainbow_table_color import RainbowTableColor

load_dotenv()

def load_manifest(file_path: str) -> Manifest:
    """
    Load a YAML manifest file and convert it to a Manifest object.

    Args:
        file_path: Path to the YAML manifest file

    Returns:
        A Manifest object
    """
    # Load YAML file
    with open(file_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    return Manifest(**yaml_data)

def get_my_reference_proposals(color_character: str ) -> SongProposal:
    # Loop through the staged raw material and return manifests that match the agent's color
    proposal = SongProposal(iterations=[])
    for root, _, files in os.walk(os.getenv('MANIFEST_PATH')):
        for file in files:
            if file.endswith('.yml'):
                try:
                    manifest = load_manifest(os.path.join(root, file))
                    if manifest.rainbow_color.mnemonic_character_value == color_character:
                        proposal_from_manifest = SongProposalIteration(
                            iteration_id=manifest.manifest_id,
                            bpm=manifest.bpm,
                            tempo=manifest.tempo,
                            key=manifest.key,
                            rainbow_color=manifest.rainbow_color,
                            title=manifest.title,
                            mood=manifest.mood,
                            genres=manifest.genres,
                            concept=manifest.concept
                        )
                        proposal.iterations.append(proposal_from_manifest)
                except Exception as e:
                    print(f"Error loading manifest {file}: {e}")
    return proposal


if __name__ == '__main__':
    # Example usage
    manifest = load_manifest('../../staged_raw_material/01_01/01_01.yml')
    print(manifest)
    my_manifest = get_my_reference_proposals('Z')
    print(my_manifest)