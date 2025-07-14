import os
import random
import uuid
import yaml
import asyncio

from random import uniform
from app.enums.plan_state import PlanState
from app.objects.plan_feedback import RainbowPlanFeedback
from app.objects.rainbow_color import RainbowColor
from app.objects.rainbow_song_meta import RainbowSongStructureModel
from app.objects.song_plan import RainbowSongPlan
from app.objects.sounds_like import RainbowSoundsLike
from app.resources.plans.negative_genre_reference import NEGATIVE_GENRES
from app.resources.plans.negative_mood_reference import NEGATIVE_MOODS
from app.resources.plans.positive_genre_reference import POSITIVE_GENRES
from app.resources.plans.positive_mood_reference import POSITIVE_MOODS
from app.resources.plans.tempo_reference import MINIMUM_BPM, MAXIMUM_BPM, TEMPO_CHANGE_TABLE
from app.utils.db_util import create_artist, get_artist_by_discogs_id, update_artist, get_artist_by_local_id
from app.utils.discog_util import search_discogs_artist
from app.utils.string_util import get_random_musical_key, convert_to_rainbow_color
from app.objects.db_models.artist_schema import ArtistSchema, RainbowArtist

POSITIVE_REFERENCE_PLAN_NAMES = ["close", "closer", "closest"]
NEGATIVE_REFERENCE_PLAN_NAMES = ["far", "further", "furthest"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_TO_STAGED_RAW_MATERIALS = os.path.abspath(os.path.join(SCRIPT_DIR, "../..", "staged_raw_material"))
PATH_TO_REFERENCE_PLANS = os.path.join(SCRIPT_DIR, "../..", "plans/reference/unreviewed")


def swap_random_items(count: int, source_list: list, new_items_list: list) -> list:
    """
    Randomly removes 'count' items from source_list and replaces them with 'count'
    random items from new_items_list. Returns the modified list sorted alphabetically.

    Args:
        count: Number of items to swap
        source_list: list to modify
        new_items_list: list to draw new items from

    Returns:
        Modified source_list with swapped items, sorted alphabetically
    """
    if count < 0:
        raise ValueError("Count must be non-negative")

    if count > len(source_list):
        raise ValueError(f"Cannot remove {count} items from a list with only {len(source_list)} items")

    if count > len(new_items_list):
        raise ValueError(f"Cannot add {count} items from a list with only {len(new_items_list)} items")
    items_to_keep = random.sample(source_list, len(source_list) - count)
    items_to_add = random.sample(new_items_list, count)
    result = items_to_keep + items_to_add
    return sorted(result)


def lookup_result_from_roll(roll: float, table: list) -> any:
    """
    Maps a percentage roll (0-100) to a result based on defined ranges.

    Args:
        roll: A float between 0.0 and 100.0
        table: list of tuples in format [(max_value, result), ...]

    Returns:
        The corresponding result for the roll
    """
    for threshold, result in table:
        if roll <= threshold:
            return result
    return table[-1][1]


async def stub_out_reference_plans(current_manifest_id: str,
                             manifest_bpm: int,
                             manifest_tempo: str,
                             manifest_key: str,
                             manifest_structure: list[RainbowSongStructureModel],
                             manifest_sounds_like: list[RainbowArtist],
                             manifest_genres: list[str],
                             manifest_mood: list[str],
                             manifest_color: RainbowColor):
    """
    Stubs out reference plans for a given manifest ID with positive and negative examples.
    :param current_manifest_id:
    :param manifest_bpm:
    :param manifest_tempo:
    :param manifest_key:
    :param manifest_structure:
    :param manifest_sounds_like:
    :param manifest_genres:
    :param manifest_mood:
    :param manifest_color:
    :return:
    """
    print(f"Stubbing out reference plans for manifest ID: {current_manifest_id}")
    for index, positive_name in enumerate(POSITIVE_REFERENCE_PLAN_NAMES):
        positive_plan_id = str(uuid.uuid4())
        positive_plan_file_name = f"{current_manifest_id}_{positive_name}.yml"
        positive_plan = RainbowSongPlan()
        positive_plan.plan_id = positive_plan_id
        positive_plan.plan_state = PlanState.generated
        positive_plan.associated_resource = current_manifest_id
        positive_plan.bpm = manifest_bpm
        positive_plan.tempo = manifest_tempo
        positive_plan.key = manifest_key
        positive_plan.moods = manifest_mood
        positive_plan.moods_feedback = RainbowPlanFeedback(
            plan_id=positive_plan_id,
            field_name="moods",
            rating=None,
            comment=None,
        )
        positive_plan.sounds_like = await manifest_artist_to_soundslike(manifest_sounds_like)
        positive_plan.sounds_like_feedback = RainbowPlanFeedback(
            plan_id=positive_plan_id,
            field_name="sounds_like",
            rating=None,
            comment=None,
        )
        positive_plan.rainbow_color = manifest_color
        positive_plan.rainbow_color_feedback = RainbowPlanFeedback(
            plan_id=positive_plan_id,
            field_name="rainbow_color",
            rating=None,
            comment=None,
        )
        positive_plan.plan = None
        positive_plan.plan_feedback = RainbowPlanFeedback(
            plan_id=positive_plan_id,
            field_name="plan",
            rating=None,
            comment=None,
        )
        positive_plan.implementation_notes = RainbowPlanFeedback(
            plan_id=positive_plan_id,
            field_name="implementation_notes",
            rating=None,
            comment=None,
        )
        positive_plan.genres = manifest_genres
        positive_plan.genres_feedback = RainbowPlanFeedback(
            plan_id=positive_plan_id,
            field_name="genres",
            rating=None,
            comment=None,
        )
        positive_plan = degrade_reference_plans(positive_plan, index + 1, positive=True)
        plan_yaml = positive_plan.to_yaml()
        plan_file_path = os.path.join(PATH_TO_REFERENCE_PLANS, positive_plan_file_name)
        with open(plan_file_path, 'w') as fy:
            fy.write(plan_yaml)
        print(f"Created positive reference plan: {positive_plan_file_name}")

    for index, negative_name in enumerate(NEGATIVE_REFERENCE_PLAN_NAMES):
        negative_plan_id = str(uuid.uuid4())
        negative_plan_file_name = f"{current_manifest_id}_{negative_name}.yml"
        negative_plan = RainbowSongPlan()
        negative_plan.plan_id = negative_plan_id
        negative_plan.plan_state = PlanState.generated
        negative_plan.associated_resource = current_manifest_id
        negative_plan.bpm = manifest_bpm
        negative_plan.tempo = manifest_tempo
        negative_plan.key = manifest_key
        negative_plan.moods = manifest_mood
        negative_plan.moods_feedback = RainbowPlanFeedback(
            plan_id=negative_plan_id,
            field_name="moods",
            rating=None,
            comment=None,
        )
        negative_plan.sounds_like = await manifest_artist_to_soundslike(manifest_sounds_like)
        negative_plan.sounds_like_feedback = RainbowPlanFeedback(
            plan_id=negative_plan_id,
            field_name="sounds_like",
            rating=None,
            comment=None,
        )
        negative_plan.rainbow_color = manifest_color
        negative_plan.rainbow_color_feedback = RainbowPlanFeedback(
            plan_id=negative_plan_id,
            field_name="rainbow_color",
            rating=None,
            comment=None,
        )
        negative_plan.plan = None
        negative_plan.plan_feedback = RainbowPlanFeedback(
            plan_id=negative_plan_id,
            field_name="plan",
            rating=None,
            comment=None,
        )
        negative_plan.implementation_notes = RainbowPlanFeedback(
            plan_id=negative_plan_id,
            field_name="implementation_notes",
            rating=None,
            comment=None,
        )
        negative_plan.genres = manifest_genres
        negative_plan.genres_feedback = RainbowPlanFeedback(
            plan_id=negative_plan_id,
            field_name="genres",
            rating=None,
            comment=None,
        )
        negative_plan = degrade_reference_plans(negative_plan, index + 1, positive=False)
        plan_yaml = negative_plan.to_yaml()
        plan_file_path = os.path.join(PATH_TO_REFERENCE_PLANS, negative_plan_file_name)
        with open(plan_file_path, 'w') as fy:
            fy.write(plan_yaml)
        print(f"Created negative reference plan: {negative_plan_file_name}")


async def manifest_artist_to_soundslike(manifest_sounds_like: list[RainbowArtist]) -> list[RainbowSoundsLike]:
    """
    Converts a list of RainbowArtist objects from a manifest into a list of RainbowSoundsLike objects.
    :param manifest_sounds_like:
    :return:
    """
    sounds_likes = []
    for manifest_artist in manifest_sounds_like:
        if manifest_artist.name:
            sounds_like_artist = RainbowArtist(
                name=manifest_artist.name,
                id=manifest_artist.id,
                discogs_id=manifest_artist.discogs_id
            )
            current_enriched_artist = await enrich_sounds_like(sounds_like_artist)
            if current_enriched_artist:
                current_sounds_like = RainbowSoundsLike(
                    artist_a=current_enriched_artist,
                    artist_b=None,
                    descriptor_a="similar to",
                    descriptor_b=None,
                    location="unknown"
                )
                sounds_likes.append(current_sounds_like)
    return sounds_likes

def degrade_reference_plans(plan: RainbowSongPlan, degrade: float, positive: bool) -> RainbowSongPlan:
    """
    Degrades a RainbowSongPlan based on the degradation factor.
    :param plan:
    :param degrade:
    :param positive:
    :return:
    """
    new_plan: RainbowSongPlan = plan
    new_plan.bpm = randomly_modify_bpm(plan.bpm, degrade)
    new_plan.key = randomly_modify_key(degrade, plan.key)
    new_plan.tempo = randomly_modify_tempo(degrade, plan.tempo)
    new_plan.moods = randomly_modify_mood(plan.moods, degrade, positive)
    new_plan.genres = randomly_modify_genres(plan.genres, degrade, positive)
    new_plan.sounds_like = randomly_modify_sounds_like(plan.sounds_like, degrade, positive)
    new_plan.structure = randomly_modify_structure(plan.structure, degrade, positive)
    return new_plan

def randomly_modify_key(degradation: float, current_key: str) -> str:
    key_roll = uniform(0.0, 100.0)
    if key_roll < degradation * 10.0:
       return get_random_musical_key()
    return current_key


def randomly_modify_bpm(current_bpm: int, degradation: float) -> int:
    r = degradation * 3.33
    ri = int(r)
    modifier = uniform(ri, ri * -1)
    modified_bpm = current_bpm + modifier
    if modified_bpm <= MINIMUM_BPM:
        modified_bpm = MINIMUM_BPM
    if modified_bpm >= MAXIMUM_BPM:
        modified_bpm = MAXIMUM_BPM
    return int(modified_bpm)


def randomly_modify_tempo(degradation: float, current_tempo: str) -> str:
    enact_roll = uniform(0.0, 100.0)
    if enact_roll <= degradation:
        tempo_roll = uniform(0.0, 100.0)
        new_tempo = lookup_result_from_roll(tempo_roll, TEMPO_CHANGE_TABLE)
        if new_tempo is not None:
            return new_tempo
    return current_tempo


def randomly_modify_structure(current_structure: list[RainbowSongStructureModel], degradation: float, positive: bool)-> list[RainbowSongStructureModel]:
    # ToDo: Slice and dice the song structure based on degradation, using combine and split functions
    pass


def randomly_modify_sounds_like(current_sounds_like: list[RainbowSoundsLike], degradation: float, positive: bool)-> list[RainbowSoundsLike]:
    # ToDo: Randomly modify the sounds like artists based on degradation.
    pass


def randomly_modify_genres(current_genres: list[str], degradation: float, positive: bool) -> list[str]:
    i = int(degradation * 0.5)
    if i > 0:
        if positive:
            new_genres = swap_random_items(i, current_genres, POSITIVE_GENRES)
        else:
            new_genres = swap_random_items(i, current_genres, NEGATIVE_GENRES)
        return new_genres
    else:
        return current_genres


def randomly_modify_mood(current_moods: list[str], degradation: float, positive: bool) -> list[str]:
    i = int(degradation * 0.5)
    if i > 0:
        if positive:
            new_moods = swap_random_items(i, current_moods, POSITIVE_MOODS)
        else:
            new_moods = swap_random_items(i, current_moods, NEGATIVE_MOODS)
        return new_moods
    return current_moods


def update_reference_manifest_with_plans():
    """
    Updates the original manifest files with references to the generated plan files.
    Scans through staged raw material directories, reads each manifest,
    adds references to corresponding reference plans, and writes the updated manifest back.
    """
    if not os.path.isdir(PATH_TO_STAGED_RAW_MATERIALS):
        raise FileNotFoundError(f"Directory {PATH_TO_STAGED_RAW_MATERIALS} does not exist")
    for current_subdir in os.listdir(PATH_TO_STAGED_RAW_MATERIALS):
        current_subdir_path = os.path.join(PATH_TO_STAGED_RAW_MATERIALS, current_subdir)
        if not os.path.isdir(current_subdir_path):
            continue
        for current_file in os.listdir(current_subdir_path):
            if not current_file.endswith(".yml"):
                continue
            manifest_path = os.path.join(current_subdir_path, current_file)
            try:
                with open(manifest_path, 'r') as yf:
                    current_manifest_data = yaml.safe_load(yf)
                current_manifest_id = current_manifest_data.get('manifest_id', '')
                if not current_manifest_id:
                    print(f"Skipping {current_file}: No manifest_id found")
                    continue
                positive_plans = [f"{current_manifest_id}_{name}.yml" for name in POSITIVE_REFERENCE_PLAN_NAMES]
                negative_plans = [f"{current_manifest_id}_{name}.yml" for name in NEGATIVE_REFERENCE_PLAN_NAMES]
                current_manifest_data['reference_plans'] = {
                    'positive': positive_plans,
                    'negative': negative_plans
                }
                with open(manifest_path, 'w') as yf:
                    yaml.dump(current_manifest_data, yf, default_flow_style=False, sort_keys=False)
                print(f"Updated reference plans in {current_file}")
            except Exception as e:
                print(f"Error updating manifest {current_file}: {e}")



async def enrich_sounds_like(sounds_like_artist: RainbowArtist) -> RainbowArtist | None:
        """
        Enriches a RainbowArtist object by searching for it in Discogs.
        :param sounds_like_artist:
        :return:
        """
        working_artist = ArtistSchema(name=sounds_like_artist.name)

        # Check for None or 0 (placeholder) values
        has_no_discogs_id = sounds_like_artist.discogs_id is None or sounds_like_artist.discogs_id == 0
        has_no_local_id = sounds_like_artist.id is None or sounds_like_artist.id <= 0

        if has_no_discogs_id and has_no_local_id:
            # Case 1: No IDs at all, search by name
            search_result = await search_discogs_artist(sounds_like_artist.name)
            if search_result is not None:
                discogs_id = str(search_result.id)
                working_artist.discogs_id = discogs_id
                full_artist_data = await get_artist_by_discogs_id(discogs_id)
                if full_artist_data:
                    working_artist.profile = full_artist_data.profile
                existing = await get_artist_by_discogs_id(discogs_id)
                if existing:
                    print(f"Artist already exists with discogs ID {discogs_id}")
                    return db_arist_to_rainbow_artist(existing)
                try:
                    result = await create_artist(working_artist)
                    if result and not isinstance(result, bool):
                        return db_arist_to_rainbow_artist(result)
                    else:
                        print(f"Failed to create artist {sounds_like_artist.name}")
                        return None
                except Exception as err:
                    print(f"Error creating artist {sounds_like_artist.name}: {err}")
                    return None
        elif has_no_discogs_id and not has_no_local_id:
            # Case 2: Has local ID but no discogs ID
            try:
                local_artist = await get_artist_by_local_id(sounds_like_artist.id)
                if not local_artist:
                    print(f"No artist found with local ID {sounds_like_artist.id}")
                    return None
                search_result = await search_discogs_artist(sounds_like_artist.name)
                if search_result:
                    local_artist.discogs_id = str(search_result.id)
                    updated_artist = await update_artist(local_artist)
                    if updated_artist:
                        print(f"Updated artist {sounds_like_artist.name} with discogs ID {local_artist.discogs_id}")
                        return db_arist_to_rainbow_artist(updated_artist)
                    return db_arist_to_rainbow_artist(local_artist)
                else:
                    print(f"No Discogs artist found for {sounds_like_artist.name}")
                    return db_arist_to_rainbow_artist(local_artist)
            except Exception as err:
                print(f"Error processing artist with local ID {sounds_like_artist.id}: {err}")
                return None
        elif not has_no_discogs_id:
            # Case 3: Has a valid discogs ID
            try:
                discogs_id = str(sounds_like_artist.discogs_id)
                local_artist = await get_artist_by_discogs_id(discogs_id)
                if local_artist:
                    return db_arist_to_rainbow_artist(local_artist)
                working_artist.discogs_id = discogs_id
                working_artist.profile = full_artist_data.profile
                result = await create_artist(working_artist)
                if result and not isinstance(result, bool):
                   return db_arist_to_rainbow_artist(result)
                return None
            except Exception as err:
                print(f"Error searching for artist with discogs ID {sounds_like_artist.discogs_id}: {err}")
                return None
        return None


def split_song_structure(current_structure_node: RainbowSongStructureModel)-> list[RainbowSongStructureModel]:
    return []


def combine_song_structure(current_structure_nodes: list[RainbowSongStructureModel]) -> RainbowSongStructureModel:
    pass

def db_arist_to_rainbow_artist(db_artist: ArtistSchema) -> RainbowArtist:
    """
    Converts a database artist schema to a RainbowArtist object.
    :param db_artist: ArtistSchema object from the database
    :return: RainbowArtist object
    """
    return RainbowArtist(
        name=db_artist.name,
        id=db_artist.id,
        discogs_id=db_artist.discogs_id,
        profile=db_artist.profile,
    )

if __name__ == "__main__":
    async def main():
        if not os.path.isdir(PATH_TO_STAGED_RAW_MATERIALS):
            raise FileNotFoundError(
                f"The directory {PATH_TO_STAGED_RAW_MATERIALS} does not exist. Please ensure the raw materials are staged correctly.")
        else:
            for subdir in os.listdir(PATH_TO_STAGED_RAW_MATERIALS):
                subdir_path = os.path.join(PATH_TO_STAGED_RAW_MATERIALS, subdir)
                if os.path.isdir(subdir_path):
                    for file in os.listdir(subdir_path):
                        if file.endswith(".yml"):
                            yaml_file_path = os.path.join(subdir_path, file)
                            try:
                                with open(yaml_file_path, 'r') as f:
                                    manifest_data = yaml.safe_load(f)
                                bpm = manifest_data.get('bpm', 120)
                                tempo = manifest_data.get('tempo', '4/4')
                                key = manifest_data.get('key', 'C major')
                                raw_structure = manifest_data.get('structure', [])
                                structure = []
                                if raw_structure:
                                    for section in raw_structure:
                                        if isinstance(section, dict):
                                            structure.append(RainbowSongStructureModel(
                                                section_name=section.get('section_name', 'Unknown'),
                                                section_description=section.get('section_description', ''),
                                                sequence=section.get('sequence', 0)
                                            ))
                                        else:
                                            structure.append(section)
                                else:
                                    structure = [
                                        RainbowSongStructureModel(
                                            section_name='Intro',
                                            section_description='Introduction section',
                                            sequence=1
                                        ),
                                        RainbowSongStructureModel(
                                            section_name='Verse',
                                            section_description='Verse',
                                            sequence=2
                                        ),
                                        RainbowSongStructureModel(
                                            section_name='Chorus',
                                            section_description='Chorus',
                                            sequence=3
                                        )
                                    ]
                                sounds_like = []
                                raw_sounds_like = manifest_data.get('sounds_like', [])
                                if raw_sounds_like:
                                    for sound_like in raw_sounds_like:
                                        if isinstance(sound_like, dict):
                                            artist = RainbowArtist(
                                                name=sound_like.get('name', ''),
                                                id=sound_like.get('id', 0),
                                                discogs_id=sound_like.get('discogs_id', 0)
                                            )
                                            enriched_artist = await enrich_sounds_like(artist)
                                            if enriched_artist:
                                                sounds_like.append(enriched_artist)
                                        elif isinstance(sound_like, RainbowArtist):
                                            enriched_artist = await enrich_sounds_like(sound_like)
                                            if enriched_artist:
                                                sounds_like.append(enriched_artist)
                                genres = manifest_data.get('genres', [
                                    'Pop', 'Rock', 'Electronic'
                                ])
                                mood = manifest_data.get('mood', [
                                    'Happy', 'Energetic', 'Uplifting'
                                ])
                                manifest_id = manifest_data.get('manifest_id', 'default_manifest')
                                color_value = manifest_data.get('rainbow_color', 'Z')
                                color = convert_to_rainbow_color(color_value)
                                await stub_out_reference_plans(
                                    current_manifest_id=manifest_id,
                                    manifest_bpm=bpm,
                                    manifest_tempo=tempo,
                                    manifest_key=key,
                                    manifest_structure=structure,
                                    manifest_sounds_like=sounds_like,
                                    manifest_genres=genres,
                                    manifest_mood=mood,
                                    manifest_color=color
                                )
                            except Exception as e:
                                print(f"Error loading YAML file {yaml_file_path}: {e}")
                                continue
                else:
                    print(f"Skipping non-directory item: {subdir_path}")

    asyncio.run(main())