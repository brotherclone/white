import sys
import logging
import base64

from mido import MidiFile


from mcp.server.fastmcp import FastMCP

USER_AGENT = "midi_mate/1.0"

logging.basicConfig(level=logging.INFO)

mcp = FastMCP("midi_mate")


@mcp.tool()
def save_midi_from_base64(base64_data: str, file_name: str) -> str:
    """Save base64-encoded MIDI data to a file"""
    try:
        midi_bytes = base64.b64decode(base64_data)
        with open(file_name, 'wb') as f:
            f.write(midi_bytes)

        midi_file = MidiFile(file_name)
        track_count = len(midi_file.tracks)
        if track_count == 0:
            return f"No tracks found in MIDI file: {file_name}"
        else:
            return f"MIDI file saved successfully: {file_name} ({track_count} tracks)"

    except Exception as e:
        return f"Error saving MIDI file: {str(e)}"

mcp.tools = [save_midi_from_base64]

if __name__ == "__main__":
    try:
        mcp.run(transport='stdio')
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)
        raise