import os
import argparse

def rename_mid_audio_extensions(root_dir):
    """
    Recursively renames files ending with .mid.wav or .mid.aif to .wav or .aif.
    """
    renamed_count = 0
    total_files_checked = 0

    print(f"Starting recursive scan of: {root_dir}")

    for dirpath, _, filenames in os.walk(root_dir):
        print(f"Scanning directory: {dirpath}")

        for filename in filenames:
            total_files_checked += 1

            if filename.endswith('.mid.wav'):
                old_path = os.path.join(dirpath, filename)
                new_filename = filename.replace('.mid.wav', '.wav')
                new_path = os.path.join(dirpath, new_filename)

                print(f"  Renaming: {filename} -> {new_filename}")
                try:
                    os.rename(old_path, new_path)
                    renamed_count += 1
                    print(f"    ✓ Success")
                except Exception as e:
                    print(f"    ✗ Error: {e}")

            elif filename.endswith('.mid.aif'):
                old_path = os.path.join(dirpath, filename)
                new_filename = filename.replace('.mid.aif', '.aif')
                new_path = os.path.join(dirpath, new_filename)

                print(f"  Renaming: {filename} -> {new_filename}")
                try:
                    os.rename(old_path, new_path)
                    renamed_count += 1
                    print(f"    ✓ Success")
                except Exception as e:
                    print(f"    ✗ Error: {e}")

    print(f"\nScan complete!")
    print(f"Files checked: {total_files_checked}")
    print(f"Files renamed: {renamed_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively rename .mid.wav/.mid.aif files to .wav/.aif in a directory.")
    parser.add_argument("directory", type=str, help="Root directory to process.")
    args = parser.parse_args()
    print(f"Processing directory: {args.directory}")
    rename_mid_audio_extensions(args.directory)
    print("Renaming complete.")
