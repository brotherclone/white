import os

def rename_mid_audio_extensions(root_dir):
    """
    Recursively renames files ending with .mid.wav or .mid.aif to .wav or .aif.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.mid.wav'):
                old_path = os.path.join(dirpath, filename)
                new_filename = filename.replace('.mid.wav', '.wav')
                new_path = os.path.join(dirpath, new_filename)
                os.rename(old_path, new_path)
            elif filename.endswith('.mid.aif'):
                old_path = os.path.join(dirpath, filename)
                new_filename = filename.replace('.mid.aif', '.aif')
                new_path = os.path.join(dirpath, new_filename)
                os.rename(old_path, new_path)