"""
Example usage of HTML Chain Artifacts

This script demonstrates how to create and save HTML chain artifacts
using the three available templates: Card Catalog, Character Sheet, and Quantum Tape.
"""

import os
from dotenv import load_dotenv

from app.structures.artifacts.html_artifact_file import (
    CardCatalogArtifact,
    CharacterSheetArtifact,
    QuantumTapeArtifact,
)

load_dotenv()


def example_card_catalog():
    """Example: Create a Card Catalog artifact for a forbidden book."""
    print("Creating Card Catalog artifact...")

    base_path = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH", "./chain_artifacts")
    catalog = CardCatalogArtifact(
        thread_id="example_thread_001",
        base_path=base_path,
        image_path=f"{base_path}/img",
        danger_level=4,
        acquisition_date="DEC 15 2024",
        title="The Geometry of Forgotten Spaces",
        subtitle="A Treatise on Non-Euclidean Architecture",
        author="Dr. Helena Ashworth",
        author_credentials="PhD Theoretical Physics, Cambridge",
        year="1973",
        publisher="Miskatonic University Press",
        publisher_type="Academic Press",
        edition="First Edition (Suppressed)",
        pages=342,
        isbn="978-0-123456-78-9",
        language="English",
        translated_from="Ancient Greek fragments",
        condition="Fair - Water damage on pages 127-134",
        abstract=(
            "This controversial work explores the mathematical foundations of "
            "impossible architecture and spaces that exist outside conventional "
            "dimensional constraints. Dr. Ashworth's research combines theoretical "
            "physics with architectural analysis, proposing that certain structures "
            "can create 'folded space' where interior dimensions exceed exterior "
            "measurements. The book was suppressed following unexplained incidents "
            "at three different university libraries."
        ),
        notable_quote=(
            "The door that opens inward may lead outward, and the corridor that "
            "turns left may arrive on the right. In these spaces, Euclid sleeps "
            "and geometry dreams."
        ),
        suppression_history=(
            "Withdrawn from circulation in 1974 after three librarians reported "
            "experiencing severe spatial disorientation. Remaining copies confiscated "
            "by undisclosed government agency."
        ),
        tags=[
            "Non-Euclidean Geometry",
            "Architecture",
            "Suppressed Research",
            "Spatial Anomalies",
            "Theoretical Physics",
        ],
        acquisition_notes="Estate sale, former colleague of Dr. Ashworth",
        catalog_number="FC-1973-047",
    )

    catalog.save_file()
    print(f"Saved to: {catalog.get_artifact_path()}")
    print(f"\nFor prompt:\n{catalog.for_prompt()}\n")


def example_character_sheet():
    """Example: Create a Character Sheet for a Pulsar Palace character."""
    print("Creating Character Sheet artifact...")

    base_path = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH", "./chain_artifacts")
    sheet = CharacterSheetArtifact(
        thread_id="example_thread_001",
        base_path=base_path,
        image_path=f"{base_path}/img",
        portrait_image_url="../png/portrait_example.png",
        disposition="Curious",
        profession="Detective",
        background_place="London",
        background_time="1973",
        arrival_circumstances="Stepped through a glitching payphone booth",
        on_current=12,
        on_max=15,
        off_current=8,
        off_max=12,
        frequency_attunement=67,
        current_location="Pulsar Palace - Main Hall",
        inventory=["Notepad", "Magnifying Glass", "Broken Watch", "Strange Key"],
        reality_anchor="FLUCTUATING",
    )

    sheet.save_file()
    print(f"Saved to: {sheet.get_artifact_path()}")
    print(f"\nFor prompt:\n{sheet.for_prompt()}\n")


def example_quantum_tape():
    """Example: Create a Quantum Tape artifact for Blue Album."""
    print("Creating Quantum Tape artifact...")

    base_path = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH", "./chain_artifacts")
    tape = QuantumTapeArtifact(
        thread_id="example_thread_001",
        base_path=base_path,
        image_path=f"{base_path}/img",
        year_documented="2024",
        original_date="1980",
        original_title="John Lennon is shot outside the Dakota",
        tapeover_date="1980",
        tapeover_title="I hear my mother screaming from the kitchen",
        subject_name="Gabriel Walsh",
        age_during="5",
        location="Wantage, NJ",
        catalog_number="QT-1980-001",
    )

    tape.save_file()
    print(f"Saved to: {tape.get_artifact_path()}")
    print(f"\nFor prompt:\n{tape.for_prompt()}\n")


def main():
    """Run all examples."""
    print("=" * 60)
    print("HTML Chain Artifact Examples")
    print("=" * 60)
    print()

    example_card_catalog()
    print("-" * 60)
    print()

    example_character_sheet()
    print("-" * 60)
    print()

    example_quantum_tape()
    print("-" * 60)
    print()

    print("All artifacts created successfully!")
    print("\nCheck the output directory for HTML files.")


if __name__ == "__main__":
    main()
