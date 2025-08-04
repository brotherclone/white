import os
import random
import uuid
import yaml
import asyncio


from random import uniform
from app.enums.plan_state import PlanState
from app.enums.sounds_like_treatment import PLAN_CHANGE_TABLE, SoundsLikeTreatment
from app.objects.plan_feedback import RainbowPlanFeedback
from app.objects.rainbow_color import RainbowColor
from app.objects.rainbow_song_meta import RainbowSongStructureModel
from app.objects.song_plan import RainbowSongPlan
from app.objects.sounds_like import RainbowSoundsLike
from app.resources.plans.negative_artist_reference import NEGATIVE_ARTISTS
from app.resources.plans.negative_genre_reference import NEGATIVE_GENRES
from app.resources.plans.negative_mood_reference import NEGATIVE_MOODS
from app.resources.plans.negative_plan_structure_reference import NEGATIVE_PLAN_PARTS
from app.resources.plans.negative_sounds_like_descriptor_reference import NEGATIVE_DESCRIPTORS
from app.resources.plans.negative_sounds_like_locations import NEGATIVE_SOUNDS_LIKE_LOCATIONS
from app.resources.plans.positive_artist_reference import POSITIVE_ARTISTS
from app.resources.plans.positive_genre_reference import POSITIVE_GENRES
from app.resources.plans.positive_mood_reference import POSITIVE_MOODS
from app.resources.plans.positive_plan_structure_reference import POSITIVE_PLAN_PARTS
from app.resources.plans.positive_sounds_like_descriptor_reference import POSITIVE_DESCRIPTORS
from app.resources.plans.positive_sounds_like_locations import POSITIVE_SOUNDS_LIKE_LOCATIONS
from app.resources.plans.tempo_reference import MINIMUM_BPM, MAXIMUM_BPM, TEMPO_CHANGE_TABLE
from app.utils.db_util import update_artist, db_arist_to_rainbow_artist, get_or_create_artist
from app.utils.discog_util import search_discogs_artist, get_discogs_artist
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
    if result and hasattr(result[0], 'sequence'):
        return sorted(result, key=lambda x: x.sequence)
    else:
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
                                manifest_concept: str,
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
     :param manifest_concept:
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
        positive_plan_bpm_feedback = RainbowPlanFeedback(
            plan_id=positive_plan_id,
            field_name="bpm",
            rating=None,
            comment=None,
        )
        positive_plan.tempo = manifest_tempo
        positive_plan.tempo_feedback = RainbowPlanFeedback(
            plan_id=positive_plan_id,
            field_name="tempo",
            rating=None,
            comment=None,
        )
        positive_plan.key = manifest_key
        positive_plan.key_feedback = RainbowPlanFeedback(
            plan_id=positive_plan_id,
            field_name="key",
            rating=None,
            comment=None,
        )
        positive_plan.moods = manifest_mood
        positive_plan.moods_feedback = RainbowPlanFeedback(
            plan_id=positive_plan_id,
            field_name="moods",
            rating=None,
            comment=None,
        )
        positive_plan.sounds_like = await manifest_artists_to_soundslike(manifest_sounds_like)
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
        positive_plan.concept = manifest_concept
        positive_plan.concept_feedback = RainbowPlanFeedback(
            plan_id=positive_plan_id,
            field_name="concept",
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
        positive_plan.structure = manifest_structure
        positive_plan.structure_feedback = RainbowPlanFeedback(
            plan_id=positive_plan_id,
            field_name="structure",
            rating=None,
            comment=None,
        )
        positive_plan = await degrade_reference_plans(positive_plan, index + 1, positive=True)
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
        negative_plan.bpm_feedback = RainbowPlanFeedback(
            plan_id=negative_plan_id,
            field_name="bpm",
            rating=None,
            comment=None,
        )
        negative_plan.tempo = manifest_tempo
        negative_plan.tempo_feedback = RainbowPlanFeedback(
            plan_id=negative_plan_id,
            field_name="tempo",
            rating=None,
            comment=None,
        )
        negative_plan.key = manifest_key
        negative_plan.key_feedback = RainbowPlanFeedback(
            plan_id=negative_plan_id,
            field_name="key",
            rating=None,
            comment=None,
        )
        negative_plan.moods = manifest_mood
        negative_plan.moods_feedback = RainbowPlanFeedback(
            plan_id=negative_plan_id,
            field_name="moods",
            rating=None,
            comment=None,
        )
        negative_plan.sounds_like = await manifest_artists_to_soundslike(manifest_sounds_like)
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
        negative_plan.concept = manifest_concept
        negative_plan.concept_feedback = RainbowPlanFeedback(
            plan_id=negative_plan_id,
            field_name="concept",
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
        negative_plan.structure = manifest_structure
        negative_plan.structure_feedback = RainbowPlanFeedback(
            plan_id=negative_plan_id,
            field_name="structure",
            rating=None,
            comment=None,
        )
        negative_plan = await degrade_reference_plans(negative_plan, index + 1, positive=False)
        plan_yaml = negative_plan.to_yaml()
        plan_file_path = os.path.join(PATH_TO_REFERENCE_PLANS, negative_plan_file_name)
        with open(plan_file_path, 'w') as fy:
            fy.write(plan_yaml)
        print(f"Created negative reference plan: {negative_plan_file_name}")



async def manifest_artists_to_soundslike(manifest_sounds_like: list[RainbowArtist]) -> list[RainbowSoundsLike]:
    """
    Converts a list of RainbowArtist objects from a manifest into a list of RainbowSoundsLike objects.
    :param manifest_sounds_like:
    :return:
    """
    sounds_likes = []
    if not manifest_sounds_like or not isinstance(manifest_sounds_like, list):
        print("No sounds like artists found in manifest or invalid format")
        return sounds_likes
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
                    descriptor_a=None,
                    descriptor_b=None,
                    location=None
                )
                sounds_likes.append(current_sounds_like)
    return sounds_likes

async def degrade_reference_plans(plan: RainbowSongPlan, degrade: float, positive: bool) -> RainbowSongPlan:
    """
    Degrades a given RainbowSongPlan by randomly modifying its attributes based on the degradation factor.
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
    new_plan.sounds_like = await randomly_modify_sounds_like(plan.sounds_like, degrade, positive)
    new_plan.structure = randomly_modify_structure(plan.structure, degrade, positive)
    new_plan.concept = plan.concept if plan.concept else "No concept provided"
    return new_plan

def randomly_modify_key(degradation: float, current_key: str) -> str:
    """
    Randomly modifies the musical key based on a degradation factor.
    :param degradation:
    :param current_key:
    :return:
    """
    key_roll = uniform(0.0, 100.0)
    if key_roll < degradation * 10.0:
       return get_random_musical_key()
    return current_key


def randomly_modify_bpm(current_bpm: int, degradation: float) -> int:
    """
    Randomly modifies the BPM based on a degradation factor.
    :param current_bpm:
    :param degradation:
    :return:
    """
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
    """
    Randomly modifies the tempo of a song based on a degradation factor.
    :param degradation:
    :param current_tempo:
    :return:
    """
    enact_roll = uniform(0.0, 100.0)
    if enact_roll <= degradation:
        tempo_roll = uniform(0.0, 100.0)
        new_tempo = lookup_result_from_roll(tempo_roll, TEMPO_CHANGE_TABLE)
        if new_tempo is not None:
            return new_tempo
    return current_tempo


def randomly_modify_structure(current_structure: list[RainbowSongStructureModel], degradation: float, positive: bool)-> list[RainbowSongStructureModel]:
    """
    Randomly modifies the song structure based on a degradation factor.
    :param current_structure:
    :param degradation:
    :param positive:
    :return:
    """
    i = int(degradation * 0.5)
    return_structure: list[RainbowSongStructureModel] = current_structure
    if i > 0:
        plan_parts = POSITIVE_PLAN_PARTS if positive else NEGATIVE_PLAN_PARTS
        new_structure = swap_random_items(i, current_structure, plan_parts)
        split_roll = uniform(0.0, 100.0)
        if split_roll <= degradation * 10.0:
            new_structure = split_song_structure(new_structure)
        combined_roll = uniform(0.0, 100.0)
        if combined_roll <= degradation * 10.0:
            new_structure = combine_song_structure(new_structure)
        return_structure = new_structure
    for index, section in enumerate(return_structure):
        if not section.section_name:
            section.section_name = f"Section {index + 1}"
        if not section.section_description:
            section.section_description = f"Description for {section.section_name}"
        if section.sequence is None or section.sequence <= 0:
            section.sequence = index + 1
    return return_structure

async def randomly_modify_sounds_like(current_sounds_like: list[RainbowSoundsLike], degradation: float, positive: bool) -> list[RainbowSoundsLike]:
    """
    Randomly modifies the sounds like attributes of a song based on a degradation factor.
    :param current_sounds_like:
    :param degradation:
    :param positive:
    :return:
    """
    new_rainbow_sounds_like: list[RainbowSoundsLike] = []
    for sl in current_sounds_like:
        roll = uniform(0.0, 100.0)
        treatment = lookup_result_from_roll(roll, PLAN_CHANGE_TABLE)
        if treatment == SoundsLikeTreatment.remove_b:
            new_sl = RainbowSoundsLike(
                artist_a=sl.artist_a,
                artist_b=None,
                descriptor_a=sl.descriptor_a,
                descriptor_b=sl.descriptor_b,
                location=sl.location
            )
            new_rainbow_sounds_like.append(new_sl)
            break
        elif treatment == SoundsLikeTreatment.b_to_a_remove_b:
            if sl.artist_b is not None:
                new_sl = RainbowSoundsLike(
                    artist_a=sl.artist_b,
                    artist_b=None,
                    descriptor_a=sl.descriptor_a,
                    descriptor_b=sl.descriptor_b,
                    location=sl.location
                )
                new_rainbow_sounds_like.append(new_sl)
            else:
                new_rainbow_sounds_like.append(sl)
            break
        elif treatment == SoundsLikeTreatment.swap_a:
            artist_list = POSITIVE_ARTISTS if positive else NEGATIVE_ARTISTS
            new_artist_name = RainbowArtist(name=random.choice(artist_list))
            enriched_new_artist = await enrich_sounds_like(new_artist_name)
            if enriched_new_artist:
                new_sl = RainbowSoundsLike(
                    artist_a=enriched_new_artist,
                    artist_b=sl.artist_b,
                    descriptor_a=sl.descriptor_a,
                    descriptor_b=sl.descriptor_b,
                    location=sl.location
                )
                new_rainbow_sounds_like.append(new_sl)
            break
        elif treatment == SoundsLikeTreatment.swap_b:
            if sl.artist_b is not None:
                artist_list = POSITIVE_ARTISTS if positive else NEGATIVE_ARTISTS
                new_artist_name = RainbowArtist(name=random.choice(artist_list))
                enriched_new_artist = await enrich_sounds_like(new_artist_name)
                if enriched_new_artist:
                    new_sl = RainbowSoundsLike(
                        artist_a=sl.artist_a,
                        artist_b=enriched_new_artist,
                        descriptor_a=sl.descriptor_a,
                        descriptor_b=sl.descriptor_b,
                        location=sl.location
                    )
                    new_rainbow_sounds_like.append(new_sl)
            else:
                new_rainbow_sounds_like.append(sl)
            break
        elif treatment == SoundsLikeTreatment.swap_both_a_and_b:
            if sl.artist_b is not None:
                artist_list = POSITIVE_ARTISTS if positive else NEGATIVE_ARTISTS
                new_artist_a_name = RainbowArtist(name=random.choice(artist_list))
                enriched_new_artist_a = await enrich_sounds_like(new_artist_a_name)
                if enriched_new_artist_a:
                    new_artist_b_name = RainbowArtist(name=random.choice(artist_list))
                    enriched_new_artist_b = await enrich_sounds_like(new_artist_b_name)
                    if enriched_new_artist_b:
                        new_sl = RainbowSoundsLike(
                            artist_a=enriched_new_artist_a,
                            artist_b=enriched_new_artist_b,
                            descriptor_a=sl.descriptor_a,
                            descriptor_b=sl.descriptor_b,
                            location=sl.location
                        )
                        new_rainbow_sounds_like.append(new_sl)
            else:
                artist_list = POSITIVE_ARTISTS if positive else NEGATIVE_ARTISTS
                new_artist_a_name = RainbowArtist(name=random.choice(artist_list))
                enriched_new_artist_a = await enrich_sounds_like(new_artist_a_name)
                if enriched_new_artist_a:
                    new_sl = RainbowSoundsLike(
                        artist_a=enriched_new_artist_a,
                        artist_b=None,
                        descriptor_a=sl.descriptor_a,
                        descriptor_b=sl.descriptor_b,
                        location=sl.location
                    )
                    new_rainbow_sounds_like.append(new_sl)
            break
        elif treatment == SoundsLikeTreatment.change_location:
            if sl.location is not None:
                location_list = POSITIVE_SOUNDS_LIKE_LOCATIONS if positive else NEGATIVE_SOUNDS_LIKE_LOCATIONS
                new_location = random.choice(location_list)
                new_sl = RainbowSoundsLike(
                    artist_a=sl.artist_a,
                    artist_b=sl.artist_b,
                    descriptor_a=sl.descriptor_a,
                    descriptor_b=sl.descriptor_b,
                    location= new_location)
                new_rainbow_sounds_like.append(new_sl)
            else:
                new_rainbow_sounds_like.append(sl)
            break
        elif treatment == SoundsLikeTreatment.remove_location:
            new_sl = RainbowSoundsLike(
                artist_a=sl.artist_a,
                artist_b=sl.artist_b,
                descriptor_a=sl.descriptor_a,
                descriptor_b=sl.descriptor_b,
                location=None
            )
            new_rainbow_sounds_like.append(new_sl)
            break
        elif treatment == SoundsLikeTreatment.change_descriptor_a:
            descriptor_list = POSITIVE_DESCRIPTORS if positive else NEGATIVE_DESCRIPTORS
            new_descriptor = random.choice(descriptor_list)
            new_sl = RainbowSoundsLike(
                artist_a=sl.artist_a,
                artist_b=sl.artist_b,
                descriptor_a=new_descriptor,
                descriptor_b=sl.descriptor_b,
                location=sl.location
            )
            new_rainbow_sounds_like.append(new_sl)
            break
        elif treatment == SoundsLikeTreatment.change_descriptor_b:
            descriptor_list = POSITIVE_DESCRIPTORS if positive else NEGATIVE_DESCRIPTORS
            new_descriptor = random.choice(descriptor_list)
            new_sl = RainbowSoundsLike(
                artist_a=sl.artist_a,
                artist_b=sl.artist_b,
                descriptor_a=sl.descriptor_a,
                descriptor_b=new_descriptor,
                location=sl.location
            )
            new_rainbow_sounds_like.append(new_sl)
            break
        elif treatment == SoundsLikeTreatment.change_descriptor_a_and_b:
            descriptor_list = POSITIVE_DESCRIPTORS if positive else NEGATIVE_DESCRIPTORS
            new_descriptor_a, new_descriptor_b = random.sample(descriptor_list, 2)
            new_sl = RainbowSoundsLike(
                artist_a=sl.artist_a,
                artist_b=sl.artist_b,
                descriptor_a=new_descriptor_a,
                descriptor_b=new_descriptor_b,
                location=sl.location
            )
            new_rainbow_sounds_like.append(new_sl)
            break
        elif treatment == SoundsLikeTreatment.change_descriptor_a_and_location:
            descriptor_list = POSITIVE_DESCRIPTORS if positive else NEGATIVE_DESCRIPTORS
            location_list = POSITIVE_SOUNDS_LIKE_LOCATIONS if positive else NEGATIVE_SOUNDS_LIKE_LOCATIONS
            new_location = random.choice(location_list)
            new_descriptor = random.choice(descriptor_list)
            new_sl = RainbowSoundsLike(
                artist_a=sl.artist_a,
                artist_b=sl.artist_b,
                descriptor_a=new_descriptor,
                descriptor_b=sl.descriptor_b,
                location=new_location
            )
            new_rainbow_sounds_like.append(new_sl)
            break
        elif treatment == SoundsLikeTreatment.change_descriptor_b_and_location:
            descriptor_list = POSITIVE_DESCRIPTORS if positive else NEGATIVE_DESCRIPTORS
            location_list = POSITIVE_SOUNDS_LIKE_LOCATIONS if positive else NEGATIVE_SOUNDS_LIKE_LOCATIONS
            new_location = random.choice(location_list)
            new_descriptor = random.choice(descriptor_list)
            new_sl = RainbowSoundsLike(
                artist_a=sl.artist_a,
                artist_b=sl.artist_b,
                descriptor_a=sl.descriptor_a,
                descriptor_b=new_descriptor,
                location=new_location
            )
            new_rainbow_sounds_like.append(new_sl)
            break
        elif treatment == SoundsLikeTreatment.change_descriptor_a_b_and_location:
            descriptor_list = POSITIVE_DESCRIPTORS if positive else NEGATIVE_DESCRIPTORS
            location_list = POSITIVE_SOUNDS_LIKE_LOCATIONS if positive else NEGATIVE_SOUNDS_LIKE_LOCATIONS
            new_location = random.choice(location_list)
            new_descriptor_a, new_descriptor_b = random.sample(descriptor_list, 2)
            new_sl = RainbowSoundsLike(
                artist_a=sl.artist_a,
                artist_b=sl.artist_b,
                descriptor_a=new_descriptor_a,
                descriptor_b=new_descriptor_b,
                location=new_location
            )
            new_rainbow_sounds_like.append(new_sl)
            break
        elif treatment == SoundsLikeTreatment.remove_descriptor_a:
            new_sl = RainbowSoundsLike(
                artist_a=sl.artist_a,
                artist_b=sl.artist_b,
                descriptor_a=None,
                descriptor_b=sl.descriptor_b,
                location=sl.location
            )
            new_rainbow_sounds_like.append(new_sl)
            break
        elif treatment == SoundsLikeTreatment.remove_descriptor_b:
            new_sl = RainbowSoundsLike(
                artist_a=sl.artist_a,
                artist_b=sl.artist_b,
                descriptor_a=sl.descriptor_a,
                descriptor_b=None,
                location=sl.location
            )
            new_rainbow_sounds_like.append(new_sl)
            break
        elif treatment == SoundsLikeTreatment.remove_descriptor_a_and_b:
            new_sl = RainbowSoundsLike(
                artist_a=sl.artist_a,
                artist_b=sl.artist_b,
                descriptor_a=None,
                descriptor_b=None,
                location=sl.location
            )
            new_rainbow_sounds_like.append(new_sl)
            break
        else:
            new_rainbow_sounds_like.append(sl)
    return new_rainbow_sounds_like

def randomly_modify_genres(current_genres: list[str], degradation: float, positive: bool) -> list[str]:
    """
    Randomly modifies the genres of a song based on a degradation factor.
    :param current_genres:
    :param degradation:
    :param positive:
    :return:
    """
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
    """
    Randomly modifies the moods of a song based on a degradation factor.
    :param current_moods:
    :param degradation:
    :param positive:
    :return:
    """
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
    if not sounds_like_artist or not sounds_like_artist.name:
        print(f"Missing artist name or object for sounds like enrichment")
    else:
        has_no_discogs_id = sounds_like_artist.discogs_id is None or sounds_like_artist.discogs_id == 0
        has_no_local_id = sounds_like_artist.id is None or sounds_like_artist.id <= 0
        if has_no_discogs_id and has_no_local_id:
            discogs_artist = await search_discogs_artist(sounds_like_artist.name)
            if discogs_artist:
                discogs_artist_id = discogs_artist.id
                if not discogs_artist_id:
                    print(f"Discogs artist ID not found for {sounds_like_artist.name}")
                    return None
                discogs_record = await get_discogs_artist(discogs_artist_id)
                if discogs_record:
                    new_artist = ArtistSchema(
                        name=sounds_like_artist.name,
                        discogs_id=discogs_artist_id,
                        profile=discogs_artist.profile if discogs_artist.profile else ''
                    )
                    created_artist = await get_or_create_artist(new_artist)
                    if created_artist:
                        return db_arist_to_rainbow_artist(created_artist)
                    else:
                        print(f"Failed to create artist in local database for {sounds_like_artist.name}")
                        return None
                else:
                    print(f"Artist not found in local database for Discogs ID {discogs_artist_id}")
                    return None
            else:
                print(f"Artist {sounds_like_artist.name} not found in Discogs")
                return None
        elif has_no_discogs_id and not has_no_local_id:
            discogs_artist = await search_discogs_artist(sounds_like_artist.name)
            if discogs_artist:
                discogs_record = await get_discogs_artist(discogs_artist.id)
                if discogs_record:
                    updated_artist = ArtistSchema(
                        id=sounds_like_artist.id,
                        discogs_id=discogs_artist.id,
                        name=sounds_like_artist.name,
                        profile=discogs_artist.profile if discogs_artist.profile else ''
                    )
                    updated_record = await update_artist(updated_artist)
                    if updated_record:
                        return db_arist_to_rainbow_artist(updated_record)
                    else:
                        print(f"Failed to update artist {sounds_like_artist.name} in local database")
                        return None
                else:
                    print(f"Artist {sounds_like_artist.name} not found in local database for Discogs ID {discogs_artist.get('id')}")
                    return None
            else:
                print(f"Artist {sounds_like_artist.name} not found in Discogs")
                return None
        elif not has_no_discogs_id and has_no_local_id:
            discogs_record = await get_discogs_artist(sounds_like_artist.discogs_id)
            if discogs_record:
                new_artist = ArtistSchema(
                    name=sounds_like_artist.name,
                    discogs_id=discogs_record.id,
                    profile= discogs_record.profile if discogs_record.profile else ''
                )
                created_artist = await get_or_create_artist(new_artist)
                if created_artist:
                    return db_arist_to_rainbow_artist(created_artist)
                else:
                    print(f"Failed to create artist in local database for {sounds_like_artist.name}")
                    return None
            else:
                print(f"Artist with Discogs ID {sounds_like_artist.discogs_id} not found in local database")
                return None
        elif not sounds_like_artist.profile:
            discogs_record = await get_discogs_artist(sounds_like_artist.discogs_id)
            if discogs_record:
                updated_artist = ArtistSchema(
                    id=sounds_like_artist.id,
                    discogs_id=discogs_record.id,
                    name=sounds_like_artist.name,
                    profile=discogs_record.profile if discogs_record.profile else ''
                )
                updated_record = await update_artist(updated_artist)
                if updated_record:
                    return db_arist_to_rainbow_artist(updated_record)
                else:
                    print(f"Failed to update artist {sounds_like_artist.name} in local database")
                    return None
            else:
                print(f"Artist with Discogs ID {sounds_like_artist.discogs_id} not found")
                return None
    return None



def split_song_structure(current_structure: list[RainbowSongStructureModel])-> list[RainbowSongStructureModel]:
    to_split = random.choice(current_structure)
    if to_split:
        split_index = random.randint(0, len(to_split.section_name) - 1)
        new_section_name = to_split.section_name[:split_index] + " Split" + to_split.section_name[split_index:]
        new_section = RainbowSongStructureModel(
            section_name=new_section_name,
            section_description=to_split.section_description,
            sequence=to_split.sequence
        )
        current_structure.append(new_section)
        return sorted(current_structure, key=lambda x: x.sequence)
    return current_structure


def combine_song_structure(current_structure: list[RainbowSongStructureModel]) -> list[RainbowSongStructureModel]:
    if len(current_structure) == 1:
        print("Cannot combine structure with only one section")
        return current_structure
    to_combine = random.sample(current_structure, 2)
    if to_combine:
        combined_section_name = f"{to_combine[0].section_name} & {to_combine[1].section_name}"
        combined_section_description = f"{to_combine[0].section_description} and {to_combine[1].section_description}"
        combined_section = RainbowSongStructureModel(
            section_name=combined_section_name,
            section_description=combined_section_description,
            sequence=min(to_combine[0].sequence, to_combine[1].sequence)
        )
        current_structure.remove(to_combine[0])
        current_structure.remove(to_combine[1])
        current_structure.append(combined_section)
        return sorted(current_structure, key=lambda x: x.sequence)
    return current_structure

async def process_single_manifest(manifest_file_path: str):
    """
    Process a single manifest file to create reference plans.

    Args:
        manifest_file_path: Path to the manifest YAML file

    Returns:
        bool: True if processing was successful, False otherwise
    """
    if not os.path.isfile(manifest_file_path):
        raise FileNotFoundError(f"The file {manifest_file_path} does not exist.")

    try:
        with open(manifest_file_path, 'r') as f:
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
                        sequence=section.get('sequence', 0),
                        start_time= section.get('start_time', None),
                        end_time= section.get('end_time', None),
                        duration=section.get('duration', None),
                        midi_group=section.get('midi_group', None),
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
                artist = RainbowArtist(
                    name=sound_like.get('name', ''),
                    id=sound_like.get('id', 0),
                    discogs_id=sound_like.get('discogs_id', 0)
                )
                if artist.name:
                    sounds_like.append(artist)
        genres = manifest_data.get('genres', ['Pop', 'Rock', 'Electronic'])
        mood = manifest_data.get('mood', ['Happy', 'Energetic', 'Uplifting'])
        manifest_id = manifest_data.get('manifest_id', 'default_manifest')
        color_value = manifest_data.get('rainbow_color', 'Z')
        concept_value = manifest_data.get('concept', 'Make a song that feels like a haunted house without the monsters.')
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
            manifest_concept=concept_value,
            manifest_color=color
        )
        print(f"Successfully processed manifest: {manifest_file_path}")
        return True
    except Exception as e:
        print(f"Error processing manifest {manifest_file_path}: {e}")
        return False


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
                            await process_single_manifest(yaml_file_path)
        update_reference_manifest_with_plans()
    asyncio.run(main())