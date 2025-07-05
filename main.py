import os
import shutil
import app.objects.rainbow_song_meta
import app.objects.rainbow_song


def clear_working_directory():
    working_dir = os.path.join(os.path.dirname(__file__), "working")
    if os.path.exists(working_dir):
        # Remove the directory and all its contents
        shutil.rmtree(working_dir)
        print(f"Removed contents of {working_dir}")

    # Create fresh empty directory
    os.makedirs(working_dir, exist_ok=True)
    print(f"Created fresh working directory at {working_dir}")


if __name__ == "__main__":
    raw_materials_path = os.path.join(os.path.dirname(__file__), "staged_raw_material")
    if not os.path.isdir(raw_materials_path):
        raise FileNotFoundError(f"The directory {raw_materials_path} does not exist. Please ensure the raw materials are staged correctly.")
    else:
        for subdir in os.listdir(raw_materials_path):
            subdir_path = os.path.join(raw_materials_path, subdir)
            if os.path.isdir(subdir_path):
                print(f"Processing subdirectory: {subdir_path}")
                for file in os.listdir(subdir_path):
                    if file.endswith(".yml"):
                        yaml_file_path = os.path.join(subdir_path, file)
                        print(f"Found YAML file: {yaml_file_path}")
                        meta = app.objects.rainbow_song_meta.RainbowSongMeta(
                            yaml_file_name=file,
                            base_path=os.path.dirname(subdir_path),
                            track_materials_path=subdir
                        )
                        song = app.objects.rainbow_song.RainbowSong(meta_data=meta, extracts=None)
                        song.create_training_samples()
                        clear_working_directory()
            else:
                print(f"Skipping non-directory item: {subdir_path}")
