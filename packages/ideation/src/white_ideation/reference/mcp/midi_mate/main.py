import base64
import logging
import os
import sys

from mcp.server.fastmcp import FastMCP
from mido import MidiFile

USER_AGENT = "midi_mate/1.0"

logging.basicConfig(level=logging.INFO)

mcp = FastMCP("midi_mate")


@mcp.tool()
def save_midi_from_base64(base64_data: str, file_name: str, output_dir: str) -> str:
    """Save base64-encoded MIDI data to a file"""
    try:

        # Decode base64 to bytes
        midi_bytes = base64.b64decode(base64_data)
        print(f"Decoded {len(midi_bytes)} bytes")

        # Write to file
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        full_path = os.path.join(output_dir, file_name)
        print(f"Saving MIDI to {full_path}")

        if not full_path.endswith(".mid"):
            full_path += ".mid"  # Ensure the file has a .mid-extension

        with open(full_path, "wb") as f:
            f.write(midi_bytes)

        # Validate with mido
        midi_file = MidiFile(full_path)
        track_count = len(midi_file.tracks)

        return (
            f"MIDI saved: {full_path} ({track_count} tracks, {len(midi_bytes)} bytes)"
        )

    except Exception as err:
        return f"Error: {type(err).__name__}: {str(err)}"


mcp.tools = [save_midi_from_base64]

if __name__ == "__main__":
    try:
        mcp.run(transport="stdio")
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        raise
