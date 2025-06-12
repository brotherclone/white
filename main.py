import os
import app.objects.rainbow_song_meta
import app.objects.rainbow_song

if __name__ == "__main__":
    meta = app.objects.rainbow_song_meta.RainbowSongMeta(yaml_file_name="01_01.yml", base_path=os.path.join(os.path.dirname(__file__), "staged_raw_material"), track_materials_path="01_01")
    song = app.objects.rainbow_song.RainbowSong(meta_data=meta, extracts=None)
    song.create_training_samples()



