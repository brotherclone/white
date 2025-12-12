import os
import polars as pl
import yaml

from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from app.extractors.manifest_extractor.concept_extractor import ConceptExtractor
from app.structures.concepts.rainbow_table_color import (
    RainbowTableColor,
    RainbowColorTemporalMode,
    RainbowTableTransmigrationalMode,
    RainbowColorObjectionalMode,
)
from app.structures.manifests.manifest import Manifest
from app.structures.manifests.manifest_song_structure import ManifestSongStructure
from app.structures.manifests.manifest_sounds_like import ManifestSoundsLike
from app.structures.manifests.manifest_track import ManifestTrack
from app.structures.music.core.key_signature import KeySignature
from app.structures.music.core.time_signature import TimeSignature
from app.util.string_utils import format_date
from app.util.lrc_utils import extract_lyrics_from_lrc

load_dotenv()


class BuildBaseManifestDB(BaseModel):

    manifest_path: Optional[Path] = Field(
        default=None, description="Path to manifests directory"
    )
    manifest_paths: Optional[List[str]] = Field(
        default_factory=list, description="List of manifest paths"
    )
    concept_extractor: ConceptExtractor | None = None

    def __init__(self, **data):
        super().__init__(**data)
        if not os.getenv("MANIFEST_PATH"):
            raise ValueError("MANIFEST_PATH environment variable not set")
        self.manifest_path = Path(os.getenv("MANIFEST_PATH"))
        self.manifest_paths = self.get_manifests()
        # Don't overwrite concept_extractor if it was provided as a parameter

    def flatten_dict(
        self, d: Dict[str, Any], parent: str = "", sep: str = "_"
    ) -> Dict[str, Any]:
        """
        Recursively flatten nested dicts. Lists are left as-is (or handle specially if needed).
        """
        items: Dict[str, Any] = {}
        for k, v in d.items():
            key = f"{parent}{sep}{k}" if parent else k
            if isinstance(v, dict):
                items.update(self.flatten_dict(v, key, sep=sep))
            else:
                items[key] = v
        return items

    def get_manifests(self, max_depth: int = 2) -> list[str]:
        """
        Return absolute paths of .yml/.yaml files found within
        `self.manifest_path` up to `max_depth` levels deep
        (base directory is depth 0).
        """
        base = self.manifest_path
        if not base or not base.exists() or not base.is_dir():
            return []

        base = base.resolve()
        manifests: list[str] = []

        for root, dirs, files in os.walk(base):
            root_path = Path(root)
            try:
                rel = root_path.relative_to(base)
                depth = len(rel.parts)
            except ValueError:
                depth = 0
            if depth >= max_depth:
                dirs[:] = []
            for file_name in files:
                if Path(file_name).suffix.lower() in (".yml", ".yaml"):
                    manifests.append(str((root_path / file_name).resolve()))
        return manifests

    def process_manifests(self):
        all_records = []
        for manifest_path in self.manifest_paths:
            try:
                with open(manifest_path, "r") as f:
                    manifest_data = yaml.safe_load(f)
                    try:
                        manifest = Manifest(**manifest_data)
                        tm = (
                            manifest.rainbow_color.transmigrational_mode
                            if isinstance(manifest.rainbow_color, RainbowTableColor)
                            else None
                        )
                        sl = getattr(manifest, "sounds_like", None)
                        md = getattr(manifest, "mood", None)
                        gn = getattr(manifest, "genres", None)
                        if hasattr(manifest, "concept") and manifest.concept:

                            self.concept_extractor.load_model(
                                concept_text=manifest.concept
                            )
                            rb = (
                                self.concept_extractor.classify_concept_by_rebracketing_type()
                            )
                        lrc_lyrics = None
                        if hasattr(manifest, "lrc_file") and manifest.lrc_file:
                            lrc_path = Path(manifest_path).parent / manifest.lrc_file
                            lrc_lyrics = extract_lyrics_from_lrc(str(lrc_path))
                        record = {
                            "id": manifest.manifest_id,
                            "bpm": manifest.bpm,
                            "tempo_numerator": (
                                manifest.tempo.numerator
                                if isinstance(manifest.tempo, TimeSignature)
                                else None
                            ),
                            "tempo_denominator": (
                                manifest.tempo.denominator
                                if isinstance(manifest.tempo, TimeSignature)
                                else None
                            ),
                            "key_signature_note": (
                                manifest.key.note.pitch_name
                                if isinstance(manifest.key, KeySignature)
                                else None
                            ),
                            "key_signature_mode": (
                                manifest.key.mode.name.value
                                if isinstance(manifest.key, KeySignature)
                                else None
                            ),
                            "rainbow_color": (
                                manifest.rainbow_color.color_name
                                if isinstance(manifest.rainbow_color, RainbowTableColor)
                                else None
                            ),
                            "rainbow_color_temporal_mode": (
                                manifest.rainbow_color.temporal_mode.value
                                if isinstance(manifest.rainbow_color, RainbowTableColor)
                                and isinstance(
                                    manifest.rainbow_color.temporal_mode,
                                    RainbowColorTemporalMode,
                                )
                                else None
                            ),
                            "rainbow_color_objectional_mode": (
                                manifest.rainbow_color.objectional_mode.value
                                if isinstance(manifest.rainbow_color, RainbowTableColor)
                                and isinstance(
                                    manifest.rainbow_color.objectional_mode,
                                    RainbowColorObjectionalMode,
                                )
                                else None
                            ),
                            "rainbow_color_ontological_mode": (
                                ", ".join(
                                    [
                                        mode.value
                                        for mode in manifest.rainbow_color.ontological_mode
                                    ]
                                )
                                if isinstance(manifest.rainbow_color, RainbowTableColor)
                                and manifest.rainbow_color.ontological_mode
                                else None
                            ),
                            "transmigrational_mode": (
                                f"{tm.current_mode.value} ➡️ {tm.transitory_mode.value} ➡️ {tm.transcendental_mode.value}"
                                if tm is not None
                                and isinstance(tm, RainbowTableTransmigrationalMode)
                                else None
                            ),
                            "title": manifest.title,
                            "release_date": format_date(
                                getattr(manifest, "release_date", None)
                            ),
                            "album_sequence": manifest.album_sequence,
                            "total_running_time": manifest.TRT,
                            "vocals": manifest.vocals,
                            "lyrics": manifest.lyrics,
                            "lrc_lyrics": lrc_lyrics,
                            "sounds_like": (
                                ", ".join(
                                    f"{s.name}, discogs_id: {getattr(s, 'discogs_id', None)}"
                                    for s in sl
                                    if s is not None
                                )
                                if isinstance(sl, list) and sl
                                else (
                                    f"{sl.name}, discogs_id: {getattr(sl, 'discogs_id', None)}"
                                    if isinstance(sl, ManifestSoundsLike)
                                    else None
                                )
                            ),
                            "mood": (
                                ", ".join(md) if isinstance(md, list) and md else md
                            ),
                            "genres": (
                                ", ".join(gn) if isinstance(gn, list) and gn else gn
                            ),
                            "lrc_file": manifest.lrc_file,
                            "concept": manifest.concept,
                            "rebracketing_type": rb,
                        }
                        try:
                            song_structure = []
                            for section in (
                                manifest.structure
                                if isinstance(manifest.structure, List)
                                else []
                            ):
                                if isinstance(section, ManifestSongStructure):
                                    structure_record = f"{section.section_name} : {section.description}, {section.start_time}-{section.end_time}"
                                    song_structure.append(structure_record)
                            record["song_structure"] = " | ".join(song_structure)
                        except AttributeError:
                            print("No song structure found in manifest")
                        try:
                            for track in (
                                manifest.audio_tracks
                                if isinstance(manifest.audio_tracks, List)
                                else []
                            ):
                                if isinstance(track, ManifestTrack):
                                    track_record = {
                                        **record,
                                        "track_id": track.id,
                                        "description": track.description,
                                        "audio_file": track.audio_file,  # ToDo: full path
                                        "midi_file": track.midi_file,  # ToDo: full path
                                        "group": track.group,
                                        "midi_group_file": (
                                            track.midi_group_file
                                            if track.midi_group_file
                                            else None
                                        ),
                                        "player": track.player.value,
                                    }
                                    all_records.append(track_record)
                        except AttributeError:
                            print("No audio tracks found in manifest")
                    except ValueError as e:
                        print(
                            f"Failed to read manifest: {manifest_path} with error: {e}"
                        )
            except TypeError as e:
                print(f"Failed to read manifest: {manifest_path} with error: {e}")
        df = pl.DataFrame(all_records, infer_schema_length=None)
        df.write_parquet(
            f"{os.getenv('BASE_MANIFEST_DB_PATH')}/base_manifest_db.parquet"
        )


if __name__ == "__main__":
    BuildBaseManifestDB().process_manifests()
