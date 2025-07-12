import os
import random
import uuid
from random import uniform
from typing import List

import yaml

from app.enums.plan_state import PlanState
from app.objects.plan_feedback import RainbowPlanFeedback
from app.objects.rainbow_color import RainbowColor
from app.objects.rainbow_song_meta import RainbowSongStructureModel
from app.objects.song_plan import RainbowSongPlan
from app.objects.sounds_like import RainbowSoundsLike
from app.utils.string_util import get_random_musical_key, convert_to_rainbow_color

POSITIVE_REFERENCE_PLAN_NAMES = ["close", "closer", "closest"]
NEGATIVE_REFERENCE_PLAN_NAMES = ["far", "further", "furthest"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_TO_STAGED_RAW_MATERIALS = os.path.abspath(os.path.join(SCRIPT_DIR, "../..", "staged_raw_material"))
PATH_TO_REFERENCE_PLANS = os.path.join(SCRIPT_DIR,"../..", "plans/reference")

POSITIVE_PLAN_PARTS: List[RainbowSongStructureModel] = [
    RainbowSongStructureModel(
        section_name="Intro",
        section_description="Provides an overview of the song",
        sequence=0
    ),
    RainbowSongStructureModel(
        section_name="Verse",
        section_description="The storytelling part of the song",
        sequence=0
    ),
    RainbowSongStructureModel(
        section_name="Chorus",
        section_description="The repeated, catchy part of the song",
        sequence=0
    ),
    RainbowSongStructureModel(
        section_name="Bridge",
        section_description="Unites the song's themes",
        sequence=0
    ),
    RainbowSongStructureModel(
        section_name="Outro",
        section_description="Jams out the song",
        sequence=0
    ),
    RainbowSongStructureModel(
        section_name="Pre-Chorus",
        section_description="Builds up to the chorus",
        sequence=0
    ),
    RainbowSongStructureModel(
        section_name="Excursion",
        section_description="An extended instrumental section",
        sequence=0
    ),
    RainbowSongStructureModel(
        section_name="Interlude",
        section_description="A short new sonic landscape visited momentarily",
        sequence=0
    ),
    RainbowSongStructureModel(
        section_name="Coda",
        section_description="A concluding section that wraps up the song",
        sequence=0
    ),
    RainbowSongStructureModel(
        section_name="Found Sound",
        section_description="A section that incorporates found sounds or field recordings",
        sequence=0
    ),
    RainbowSongStructureModel(
        section_name="Melodic Guitar Lead",
        section_description="A section featuring a melodic guitar lead, without showy solos",
        sequence=0
    ),
    RainbowSongStructureModel(
        section_name="Protracted silence with room tone",
        section_description="Protracted silence with room tone. 3:33",
        sequence=0
    ),
    RainbowSongStructureModel(
        section_name="Art for art's sake",
        section_description="Expresss yourself or die",
        sequence=0
    ),
    RainbowSongStructureModel(
        section_name="Repetitive, hypnotic groove",
        section_description="Repetitive, hypnotic groove",
        sequence=0
    ),
]
NEGATIVE_PLAN_PARTS: List[RainbowSongStructureModel] = [
    RainbowSongStructureModel(
        section_name="Drum Solo",
        section_description="Really boring for audience",
        sequence=0
    ),
    RainbowSongStructureModel(
        section_name="Breakdown with Spoken Word",
        section_description="Why?",
        sequence=0
    ),
    RainbowSongStructureModel(
        section_name="Skit",
        section_description="A skit that interrupts the flow of the song",
        sequence=0
    ),
    RainbowSongStructureModel(
        section_name="Tempo Changes",
        section_description="As if the song wasn't already hard to follow",
        sequence=0
    ),
    RainbowSongStructureModel(
        section_name="Key Changes",
        section_description="Almost always unnecessary",
        sequence=0
    ),
    RainbowSongStructureModel(
        section_name="Gratuitous Guitar Solo",
        section_description="I'm sure it's fun for the guitarist, but it doesn't add anything to the song",
        sequence=0
    ),
    RainbowSongStructureModel(
        section_name="A direct plea to the listener to stand up and dance",
        section_description="Wedding stuff",
        sequence=0
    ),
    RainbowSongStructureModel(
        section_name="The whole song again in another language",
        section_description="And now I'll show off my mediocre language skills",
        sequence=0
    ),
    RainbowSongStructureModel(
        section_name="A long, drawn-out fade-out",
        section_description="A long, drawn-out fade-out",
        sequence=0
    ),
    RainbowSongStructureModel(
        section_name="Protracted silence without room tone",
        section_description="So edgy",
        sequence=0
    ),
    RainbowSongStructureModel(
        section_name="A hype man section",
        section_description="Insert hype",
        sequence=0
    ),
    RainbowSongStructureModel(
        section_name="Sex sounds",
        section_description="Ohhh girl",
        sequence=0
    ),
    RainbowSongStructureModel(
        section_name="Vocal solo with lots of trills and runs",
        section_description="Whitney Houston would be proud",
        sequence=0
    ),
]
POSITIVE_ARTISTS = [
    "Stereolab",
    "The Beatles",
    "David Bowie",
    "The Velvet Underground",
    "Syd Barrett",
    "Brian Wilson",
    "The Smiths",
    "The Cure",
    "Joy Division",
    "Radiohead",
    "Sigur Rós",
    "Mogwai",
    "Godspeed You! Black Emperor",
    "Beach House",
    "James Blake",
    "Boards of Canada",
    "Burial",
    "Flying Lotus",
    "Ólafur Arnalds",
    "Max Richter",
    "Philip Glass",
    "Steve Reich",
    "Arvo Pärt",
    "John Adams",
    "Gustavo Santaolalla",
    "Hildur Guðnadóttir",
    "Ludovico Einaudi",
    "Taylor Swift",
    "Beyoncé",
    "Billie Eilish",
    "Creedence Clearwater Revival",
    "The Ramones",
    "The Clash",
    "Sonic Youth",
    "Neutral Milk Hotel",
    "Olivia Tremor Control",
    "Pavement",
    "Sebadoh",
    "The Flaming Lips",
    "Wilco",
    "Fleetwood Mac",
    "Scott Walker",
    "Kate Bush",
    "Nick Drake",
    "Hood",
    "His Name Is Alive",
    "Os Mutantes",
    "The Zombies",
    "The Byrds",
    "The Kinks",
    "The Doors",
    "The Rolling Stones",
    "The Who",
    "The Tower Recordings",
    "The Incredible String Band",
    "The Band",
    "The Lumineers",
    "The Tall Dwarfs",
    "Lee Scratch Perry",
    "Can",
    "Kraftwerk",
    "Brian Eno",
    "Faust",
    "Tangerine Dream",
    "Neu!",
    "Amon Düül II",
    "Cluster",
    "Ash Ra Tempel",
    "Harmonia",
    "Popol Vuh",
    "Gong",
    "Soft Machine",
    "The United States of America",
    "The Silver Apples",
    "Television",
    "Television Personalities",
    "The Waterboys",
    "The Pogues",
    "Julian Cope",
    "The Teardrop Explodes",
    "Bruce Haack",
    "The Shaggs",
    "The Residents",
    "Leonard Cohen",
    "Woodie Guthrie",
    "Bob Dylan",
    "Neil Young",
    "Led Zeppelin",
    "Black Sabbath",
    "My Bloody Valentine",
    "Patti Smith",
    "Guided by Voices",
    "The Magnetic Fields",
    "Amen Dunes",
    "Swan Lake",
    "Mount Eerie",
    "Deerhunter",
    "Animal Collective",
    "Atlas Sound",
    "Derek Decker",
]
NEGATIVE_ARTISTS = [
    "Devendra Banhart",
    "Grizzly Bear",
    "The Microphones",
    "The Mountain Goats",
    "Whitney Houston",
    "Mariah Carey",
    "Celine Dion",
    "Justin Timberlake",
    "Britney Spears",
    "Backstreet Boys",
    "NSYNC",
    "Spice Girls",
    "Anthrax",
    "Metallica",
    "Megadeth",
    "Slayer",
    "Pantera",
    "Iron Maiden",
    "Judas Priest",
    "AC/DC",
    "Van Morrison",
    "Joni Mitchell",
    "Arlo Guthrie",
    "King Crimson",
    "The Allman Brothers Band",
    "The Doobie Brothers",
    "Peter, Paul and Mary",
    "Simon & Garfunkel",
    "The White Stripes",
    "The Pixies",
    "The Police",
    "Traveling Wilburys",
    "The Beach Boys",
    "Pink Floyd",
    "Morrissey",
    "Moby",
    "Aphex Twin",
    "Grimes",
    "FKA twigs",
    "Justin Bieber",
    "Nickelback",
    "Creed",
    "Maroon 5",
    "Imagine Dragons",
    "Coldplay",
    "Katy Perry",
    "Ed Sheeran",
    "Pitbull",
    "Black Eyed Peas",
    "One Direction",
    "Drake",
    "Kanye West",
    "Travis Scott",
    "Future",
    "Lil Uzi Vert",
    "Cardi B",
    "Nicki Minaj",
    "Migos",
    "Snoop Dogg",
    "50 Cent",
    "Flo Rida",
    "Pitbull",
    "W.A.S.P.",
    "Guns N' Roses",
    "Def Leppard",
    "Bon Jovi",
    "Aerosmith",
    "Van Halen",
    "Poison",
    "Motley Crue",
    "Scorpions",
    "Twisted Sister",
    "Europe",
    "Journey",
    "Foreigner",
    "REO Speedwagon",
    "Kansas",
    "Styx",
    "Boston",
    "Cheap Trick",
    "Toto",
    "Asia",
    "Survivor",
    "Warrant",
    "Cinderella",
    "Skid Row",
    "Winger",
    "Tesla",
    "Great White",
    "Slaughter",
    "LA Guns",
    "BulletBoys",
    "The Strokes",
    "Arctic Monkeys",
    "Tame Impala",
    "Fleet Foxes",
    "Bon Iver",
    "Sufjan Stevens",
    "Iron & Wine",
    "Death Cab for Cutie",
    "The Decemberists",
    "Mumford & Sons",
]
POSITIVE_MOODS: List[str] = [
    "contemplative",
    "ethereal",
    "abstract",
    "melancholic",
    "surreal",
    "existential",
    "otherworldly",
    "introspective",
    "cerebral",
    "atmospheric",
    "enigmatic",
    "reflective",
    "haunting",
    "transcendent",
    "dissonant",
    "wistful",
    "hypnagogic",
    "dreamlike",
    "pensive",
    "liminal",
    "deconstructed",
    "esoteric",
    "experimental",
    "meditative",
    "fragmented"
]
NEGATIVE_MOODS: List[str] = [
    "bombastic",
    "sentimental",
    "nostalgic",
    "simplistic",
    "formulaic",
    "sappy",
    "kitschy",
    "saccharine",
    "pandering",
    "jingoistic",
    "commercial",
    "derivative",
    "pedestrian",
    "conventional",
    "dogmatic",
    "blunt",
    "unsubtle",
    "garish",
    "predictable",
    "flippant",
    "didactic",
    "clichéd",
    "sanctimonious",
    "literal"
]
POSITIVE_GENRES: List[str] = [
    "Art Pop",
    "Art Rock",
    "Baroque Pop",
    "Goth",
    "Lo-Fi",
    "Musique Concrète",
    "Indie Rock",
    "New Wave",
    "Shoegaze",
    "Post-Punk",
    "Avant-Garde Classical",
    "Impressionist Classical",
    "Minimalism",
    "Modern Composition",
    "Americana",
    "Nashville Sound",
    "Minimal House",
    "Ambient Electronic",
    "Electropop",
    "Electroacoustic",
    "Sound Art",
    "Experimental Electronic",
    "Experimental Rock",
    "Experimental Pop",
    "Experimental Hip Hop",
    "Experimental Jazz",
    "Experimental Classical",
    "Experimental Folk",
    "Experimental Country",
    "Experimental Blues",
    "Industrial",
    "Vaporwave",
    "Psychedelic Folk",
    "Nerdcore",
    "Coldwave",
    "Dark Ambient",
    "Drone Music",
    "Post-Rock",
    "Noise",
    "Free Jazz",
    "Avantgarde Metal",
    "Psychedelic Pop",
    "Psychedelic Rock",
    "Psychedelic Soul",
    "Psychedelic Electronic",
    "Psychedelic Hip Hop",
    "Psychedelic Country",
    "Psychedelic Blues",
    "Psychedelic Folk",
    "Surf Pop",
    "Spaceage Rock",
    "Space Rock",
    "Glam Rock",
    "Surf Rock",
    "Soundtrack",
    "Celtic Folk"
]
NEGATIVE_GENRES: List[str] = [
    "Blues Rock",
    "Polka",
    "Vandeville",
    "Anti-Folk",
    "Grunge",
    "Gospel Blues",
    "Contemporary R&B",
    "Progressive Rock"
    "Canadian Blues",
    "Punk Blues",
    "Zydeco",
    "Wedding Music",
    "Novelty",
    "Comedy",
    "Reactionary Bluegrass",
    "Country Gospel",
    "Bluegrass Gospel",
    "Exercise",
    "Hi-NRG",
    "Acid House",
    "Background",
    "Lounge",
    "Chiptune",
    "Crunk",
    "Jazz Rap",
    "Latin Rap",
    "Gangsta Rap",
    "Holiday",
    "Christian Rap",
    "Christian Rock",
    "Christian Metal",
    "Christian Punk",
    "Christian Pop",
    "Christian Contemporary",
    "Christian Country",
    "Christian Hip Hop",
    "Christian Electronic",
    "Christian Indie",
    "Christian Alternative",
    "Christian Worship",
    "Marching Band",
    "Military",
    "Dixieland",
    "K-Pop",
    "Hair Metal",
    "New Age Healing",
    "Pop Punk",
    "Ska",
    "Adult-Oriented Rock",
    "Contemporary Folk",
    "Spoken Word",
]

MINIMUM_BPM = 40
MAXIMUM_BPM = 240

TEMPO_CHANGE_TABLE = [
    (80.0, None),
    (85.0, "3/4"),
    (90.0, "2/4"),
    (95.0, "6/8"),
    (99.0, "6/4"),
]


def swap_random_items(count: int, source_list: list, new_items_list: list) -> list:
    """
    Randomly removes 'count' items from source_list and replaces them with 'count'
    random items from new_items_list. Returns the modified list sorted alphabetically.

    Args:
        count: Number of items to swap
        source_list: List to modify
        new_items_list: List to draw new items from

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
    # Sort alphabetically and return
    return sorted(result)

def lookup_result_from_roll(roll: float, table: list) -> any:
    """
    Maps a percentage roll (0-100) to a result based on defined ranges.

    Args:
        roll: A float between 0.0 and 100.0
        table: List of tuples in format [(max_value, result), ...]

    Returns:
        The corresponding result for the roll
    """
    for threshold, result in table:
        if roll <= threshold:
            return result
    return table[-1][1]

def stub_out_reference_plans(current_manifest_id: str,
                             manifest_bpm: int,
                             manifest_tempo: str,
                             manifest_key: str,
                             manifest_structure: List[RainbowSongStructureModel],
                             manifest_sounds_like: List[str],
                             manifest_genres: List[str],
                             manifest_mood: List[str],
                             manifest_color: RainbowColor):
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
        positive_plan.sounds_like = stub_sounds_like(manifest_sounds_like)  # fixme
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
        positive_plan.plan = stub_plan_from_song_structure(manifest_structure)
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
        negative_plan.sounds_like = stub_sounds_like(manifest_sounds_like)  # fixme
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
        negative_plan.plan = stub_plan_from_song_structure(manifest_structure)
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


def stub_sounds_like(manifest_sounds_like: List[str]) -> RainbowSoundsLike:
        if len(manifest_sounds_like) >= 2:
            selected_artists = random.sample(manifest_sounds_like, 2)
            artist_a, artist_b = selected_artists
        elif len(manifest_sounds_like) == 1:
            artist_a = manifest_sounds_like[0]
            artist_b = "Unknown Artist"
        else:
            artist_a = "Unknown Artist"
            artist_b = "Unknown Artist"

        plan_sounds_like = RainbowSoundsLike(
            artist_name_a=artist_a,
            artist_a_local_id=str(uuid.uuid4()),
            artist_a_discogs_id=str(uuid.uuid4()),
            artist_a_musicbrainz_id=str(uuid.uuid4()),
            artist_name_b=artist_b,
            artist_b_local_id=str(uuid.uuid4()),
            artist_b_discogs_id=str(uuid.uuid4()),
            artist_b_musicbrainz_id=str(uuid.uuid4()),
            descriptor_a="Descriptor A",
            descriptor_b="Descriptor B",
            location="Unknown Location"
        )
        return plan_sounds_like

def stub_plan_from_song_structure(song_structure: List[RainbowSongStructureModel]) -> str:
    # for sec in song_structure:
    #     print(f"Section: {sec.section_name}, Description: {sec.section_description}, Sequence: {sec.sequence}")
    pl = "Make it magic!"
    return pl

def degrade_reference_plans(plan: RainbowSongPlan, degrade: float, positive:bool)-> RainbowSongPlan:
    new_plan: RainbowSongPlan = plan
    new_plan.bpm = randomly_modify_bpm(plan.bpm, degrade)
    key_roll = uniform(0.0, 100.0)
    if key_roll < degrade * 10.0:
        new_plan.key = get_random_musical_key()
    new_plan.tempo = randomly_modify_tempo(degrade, plan.tempo)
    new_plan.moods = randomly_modify_mood(plan.moods, degrade, positive)
    new_plan.genres = randomly_modify_genres(plan.genres, degrade, positive)
    return new_plan

def randomly_modify_bpm(current_bpm: int, degradation: float) -> int:
    r = degradation*3.33
    ri = int(r)
    modifier = uniform(ri, ri*-1)
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

def randomly_modify_structure():
    pass

def randomly_modify_sounds_like():
    pass

def randomly_modify_genres(current_genres: List[str], degradation: float, positive: bool) -> List[str]:
    i = int(degradation * 0.5)
    if i > 0:
        if positive:
            new_genres = swap_random_items(i, current_genres, POSITIVE_GENRES)
        else:
            new_genres = swap_random_items(i, current_genres, NEGATIVE_GENRES)
        return new_genres
    else:
        return current_genres

def randomly_modify_mood(current_moods: List[str], degradation: float, positive: bool) -> List[str]:
    i = int(degradation * 0.5)
    if i > 0:
        if positive:
            new_moods = swap_random_items(i, current_moods, POSITIVE_MOODS)
        else:
            new_moods = swap_random_items(i, current_moods, NEGATIVE_MOODS)
        return new_moods
    return current_moods

# update reference manifest with new plan ids
def update_reference_manifest_with_plans():
    pass

# get arist ids from names in sounds like
def enrich_sounds_like():
    pass

def split_song_structure():
    pass

def combine_song_structure():
    pass

if __name__ == "__main__":
    if not os.path.isdir(PATH_TO_STAGED_RAW_MATERIALS):
        raise FileNotFoundError(f"The directory {PATH_TO_STAGED_RAW_MATERIALS} does not exist. Please ensure the raw materials are staged correctly.")
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
                            sounds_like = manifest_data.get('sounds_like', [])
                            genres = manifest_data.get('genres', [
                                'Pop', 'Rock', 'Electronic'
                            ])
                            mood = manifest_data.get('mood', [
                                'Happy', 'Energetic', 'Uplifting'
                            ])
                            manifest_id = manifest_data.get('manifest_id', 'default_manifest')
                            color_value = manifest_data.get('rainbow_color', 'Z')
                            color = convert_to_rainbow_color(color_value)
                            stub_out_reference_plans(
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