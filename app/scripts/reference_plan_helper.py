import os
import uuid
from random import uniform
from typing import List

import yaml

from app.objects.rainbow_song_meta import RainbowSongStructureModel
from app.objects.song_plan import RainbowSongPlan

POSITIVE_REFERENCE_PLAN_NAMES = ["close", "closer", "closest"]
NEGATIVE_REFERENCE_PLAN_NAMES = ["far", "further", "furthest"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_TO_STAGED_RAW_MATERIALS = os.path.abspath(os.path.join(SCRIPT_DIR, "../..", "staged_raw_material"))
POSITIVE_PLAN_PARTS: List[RainbowSongStructureModel] = []
NEGATIVE_PLAN_PARTS: List[RainbowSongStructureModel] = []
POSITIVE_ARTISTS = []
NEGATIVE_ARTISTS = []
POSITIVE_MOODS: List[str] = []
NEGATIVE_MOODS: List[str] = []
MINIMUM_BPM = 40
MAXIMUM_BPM = 240

# Stub the six usual objects based on a reference manifest
def stub_out_reference_plans(current_manifest_id: str,
                             manifest_bpm: int,
                             manifest_tempo: str,
                             manifest_key: str,
                             manifest_structure: List[RainbowSongStructureModel],
                             manifest_sounds_like: List[str],
                             manifest_genres: List[str],
                             manifest_mood: List[str]):
    print(f"Stubbing out reference plans for manifest ID: {current_manifest_id}")
    print(f"Manifest Tempo: {manifest_tempo}")
    print(f"Manifest Key: {manifest_key}")
    print(f"Manifest BPM: {manifest_bpm}")
    print(f"Manifest Structure: {manifest_structure}")
    print(f"Manifest Sounds Like: {manifest_sounds_like}")
    print(f"Manifest Genres: {manifest_genres}")
    print(f"Manifest Mood: {manifest_mood}")
    for index, positive_name in enumerate(POSITIVE_REFERENCE_PLAN_NAMES):
        positive_plan_file_name = f"{current_manifest_id}_{positive_name}.yml"
        positive_plan = RainbowSongPlan()
        positive_plan.plan_id = uuid.uuid4()
        positive_plan.bpm = randomly_modify_bpm(manifest_bpm, float((index+1)*5)) # move to main degrader
        positive_plan.tempo = manifest_tempo
        positive_plan.key = manifest_key
        positive_plan.sounds_like = None # fixme
        positive_plan.genres = manifest_genres

        print(f"Stubbed positive plan: {positive_plan_file_name} {positive_plan.bpm}")
    for index, negative_name in enumerate(NEGATIVE_REFERENCE_PLAN_NAMES):
        negative_plan_file_name = f"{current_manifest_id}_{negative_name}.yml"
        negative_plan = RainbowSongPlan()
        negative_plan.bpm = randomly_modify_bpm(manifest_bpm, float(((index+1)*-1)*5)) # move to main degrader
        print(f"Stubbed negative plan: {negative_plan_file_name} {negative_plan.bpm}")

def degrade_reference_plans(plan: RainbowSongPlan, degrade: float=0.0)-> RainbowSongPlan:
    pass

def randomly_modify_bpm(current_bpm: int, degradation: float) -> int:
    r = degradation*1.33
    ri = int(r)
    modifier = uniform(ri, ri*-1)
    modified_bpm = current_bpm + modifier
    if modified_bpm <= MINIMUM_BPM:
        modified_bpm = MINIMUM_BPM
    if modified_bpm >= MAXIMUM_BPM:
        modified_bpm = MAXIMUM_BPM
    return int(modified_bpm)

def randomly_modify_tempo():
    pass

def randomly_modify_key():
    pass

def randomly_modify_structure():
    pass

def randomly_modify_sounds_like():
    pass

def randomly_modify_genres():
    pass

def randomly_modify_mood():
    pass

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
                print(f"Processing subdirectory: {subdir_path}")
                for file in os.listdir(subdir_path):
                    if file.endswith(".yml"):
                        yaml_file_path = os.path.join(subdir_path, file)
                        print(f"Found YAML file: {yaml_file_path}")
                        try:
                            with open(yaml_file_path, 'r') as f:
                                manifest_data = yaml.safe_load(f)
                            bpm = manifest_data.get('bpm', 120)
                            tempo = manifest_data.get('tempo', '4/4')
                            key = manifest_data.get('key', 'C major')
                            structure = manifest_data.get('structure', [
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
                            ])
                            sounds_like = manifest_data.get('sounds_like', [])
                            genres = manifest_data.get('genres', [
                                'Pop', 'Rock', 'Electronic'
                            ])
                            mood = manifest_data.get('mood', [
                                'Happy', 'Energetic', 'Uplifting'
                            ])
                            manifest_id = manifest_data.get('id', 'default_manifest')
                            stub_out_reference_plans(
                                current_manifest_id=manifest_id,
                                manifest_bpm=bpm,
                                manifest_tempo=tempo,
                                manifest_key=key,
                                manifest_structure=structure,
                                manifest_sounds_like=sounds_like,
                                manifest_genres=genres,
                                manifest_mood=mood
                            )
                            print(f"Loaded YAML file: {yaml_file_path}")
                        except Exception as e:
                            print(f"Error loading YAML file {yaml_file_path}: {e}")
                            continue
            else:
                print(f"Skipping non-directory item: {subdir_path}")