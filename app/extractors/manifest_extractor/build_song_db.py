class BuildSongDB:
    pass


# Aggregate all time stamps

# validate time stamps

# go through the time stamps - when a lyric start slightly BEFORE a section break that lyric the start of the section
# If a section is longer than 30 seconds, split it into multiple sections

# Load the manifest DB

# Aggregate all lyrics in that section per track
# See if there's audio info on the current track in that section
# if so slice the audio and save it
# Get the audio features for that slice
# Do the same with MIDI

# Copy all of the new data to a new DB with binary embeddings for lyrics, audio features, and MIDI features
