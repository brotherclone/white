from training.hf_dataset_prep import DATASET_ORG, DATASET_NAME, create_dataset_card


def push_to_hub(datasets: dict, version: str, private: bool = True):
    """Push each dataset as a separate config to HuggingFace Hub."""
    from huggingface_hub import HfApi

    repo_id = f"{DATASET_ORG}/{DATASET_NAME}"
    api = HfApi()

    print(f"\n{'='*60}")
    print("PUSHING TO HUGGINGFACE HUB")
    print(f"{'='*60}")
    print(f"  Repo: {repo_id}")
    print(f"  Version: {version}")
    print(f"  Private: {private}")

    # Push each table as a separate config
    for name, ds in datasets.items():
        print(f"  Pushing config '{name}'...")
        ds.push_to_hub(
            repo_id,
            config_name=name,
            private=private,
            commit_message=f"v{version}: update {name}",
        )

    # Upload the dataset card as README.md
    has_preview = "preview" in datasets
    card_content = create_dataset_card(version, has_preview=has_preview)
    api.upload_file(
        path_or_fileobj=card_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Update dataset card for v{version}",
    )
    print("  Uploaded dataset card (README.md)")

    # Create a git tag for the version
    try:
        api.create_tag(
            repo_id=repo_id,
            tag=f"v{version}",
            repo_type="dataset",
        )
        print(f"  Created tag: v{version}")
    except Exception as e:
        print(f"  Tag creation skipped: {e}")

    print(f"\n  Dataset available at: https://huggingface.co/datasets/{repo_id}")
