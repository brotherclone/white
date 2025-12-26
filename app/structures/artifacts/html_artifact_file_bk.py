# """
# HTML Chain Artifacts
#
# These artifacts render HTML templates for various chain artifact types.
# Each artifact uses a specific HTML template and data model.
# """
#
# # ToDo: Split up into multiple files
# # ToDo: implement artifact export with HTML
#
# from pathlib import Path
# from typing import List, Optional
# from pydantic import Field
#
# from app.structures.artifacts.base_artifact import ChainArtifact
# from app.structures.artifacts.template_renderer import (
#     HTMLTemplateRenderer,
#     get_template_path,
# )
# from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
# from app.structures.enums.chain_artifact_type import ChainArtifactType
#
#
#
#
#
# class CharacterSheetArtifact(ChainArtifact):
#     """
#     Pulsar Palace Character Sheet artifact.
#
#     Template Variables:
#         - portrait_image_url: URL/path to character portrait
#         - disposition: Character disposition
#         - profession: Character profession
#         - background_place: Origin place
#         - background_time: Origin time period
#         - arrival_circumstances: How they arrived
#         - on_current: Current ON stat
#         - on_max: Maximum ON stat
#         - on_percentage: ON percentage (calculated)
#         - off_current: Current OFF stat
#         - off_max: Maximum OFF stat
#         - off_percentage: OFF percentage (calculated)
#         - frequency_attunement: Frequency attunement 0-100
#         - current_location: Current location
#         - inventory_slots: HTML for inventory slots
#         - reality_anchor: Reality anchor description
#     """
#
#     chain_artifact_type: ChainArtifactType = Field(
#         default=ChainArtifactType.CHARACTER_SHEET, description="Type: Character Sheet"
#     )
#     chain_artifact_file_type: ChainArtifactFileType = Field(
#         default=ChainArtifactFileType.HTML, description="File type: HTML"
#     )
#     artifact_name: str = Field(default="character_sheet", description="Artifact name")
#     rainbow_color_mnemonic_character_value: str = Field(
#         default="Y", description="Yellow for character sheets"
#     )
#
#     # Character Sheet specific fields
#     portrait_image_url: str = Field(default="", description="Portrait image URL")
#     disposition: str = Field(description="Character disposition")
#     profession: str = Field(description="Character profession")
#     background_place: str = Field(description="Origin place")
#     background_time: str = Field(description="Origin time period")
#     arrival_circumstances: str = Field(
#         default="Unknown", description="How they arrived"
#     )
#     on_current: int = Field(description="Current ON stat")
#     on_max: int = Field(description="Maximum ON stat")
#     off_current: int = Field(description="Current OFF stat")
#     off_max: int = Field(description="Maximum OFF stat")
#     frequency_attunement: int = Field(
#         default=50, description="Frequency attunement 0-100", ge=0, le=100
#     )
#     current_location: str = Field(default="Unknown", description="Current location")
#     inventory: List[str] = Field(default_factory=list, description="Inventory items")
#     reality_anchor: str = Field(default="STABLE", description="Reality anchor status")
#
#     def save_file(self):
#         """Render and save the HTML file."""
#         template_path = get_template_path("character_sheet")
#         renderer = HTMLTemplateRenderer(template_path)
#
#         # Calculate percentages
#         data = self.model_dump()
#         data["on_percentage"] = (
#             int((self.on_current / self.on_max) * 100) if self.on_max > 0 else 0
#         )
#         data["off_percentage"] = (
#             int((self.off_current / self.off_max) * 100) if self.off_max > 0 else 0
#         )
#
#         # Generate inventory slots HTML
#         inventory_html = []
#         for i in range(9):  # 3x3 grid
#             if i < len(self.inventory):
#                 inventory_html.append(
#                     f'<div class="inventory-slot">{self.inventory[i]}</div>'
#                 )
#             else:
#                 inventory_html.append('<div class="inventory-slot empty">EMPTY</div>')
#         data["inventory_slots"] = "\n        ".join(inventory_html)
#
#         html_content = renderer.render(data)
#
#         file_path = Path(self.file_path)
#         file_path.mkdir(parents=True, exist_ok=True)
#
#         output_file = file_path / self.file_name
#         with open(output_file, "w", encoding="utf-8") as f:
#             f.write(html_content)
#
#     def flatten(self):
#         """Flatten the artifact for easier processing."""
#         return self.model_dump()
#
#     def for_prompt(self) -> str:
#         """Plain text representation for prompts."""
#         return f"""Character Sheet: {self.disposition} {self.profession}
# From: {self.background_place} ({self.background_time})
# ON: {self.on_current}/{self.on_max}
# OFF: {self.off_current}/{self.off_max}
# Location: {self.current_location}
# Inventory: {', '.join(self.inventory) if self.inventory else 'Empty'}"""
#
#
#
