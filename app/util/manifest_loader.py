import yaml
from app.structures.manifests.manifest import Manifest

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

if __name__ == '__main__':
    # Example usage
    manifest = load_manifest('../../staged_raw_material/01_01/01_01.yml')
    print(manifest)