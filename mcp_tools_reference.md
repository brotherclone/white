# MCP Tools Status & Usage Guide

## EarthlyFrames API ✅ WORKING

### Available Tools
- `get_all_recordings(filter_for_rainbow_table: bool)` - Get all albums, optionally filter to Rainbow Table only
- `album_for_color(color: str)` - Get specific album by color (e.g., "indigo", "blue")  
- `get_album(album_id: str|int)` - Get album by ID
- `get_song(album_id: str|int, song_id: str|int)` - Get individual song with lyrics

### Usage Examples
```
# Get all Rainbow Table albums
get_all_recordings(filter_for_rainbow_table=true)

# Get Pulsar Palace 
album_for_color("yellow")

# Get The Conjurer's Thread song
get_song(album_id=4, song_id=20)
```

### Data Structure
Albums include: id, title, description, rainbow_table color, songs array, streaming_links
Songs include: id, title, lyrics (HTML), trt (duration), streaming_links

## Discogs API ✅ WORKING

### Available Tools  
- `look_up_artist_by_name(artist_name: str)` - Search for artist
- `look_up_artist_by_id(artist_id: str|int)` - Get detailed artist info
- `get_group_members(group_name: str)` - Get band member details
- `get_release_list(artist_id, per_page=50, page=1, sort='year')` - Get artist discography

### Usage Examples
```
# Look up David Bowie
look_up_artist_by_name("David Bowie")

# Get Broadcast members
get_group_members("Broadcast")

# Get Bowie's releases
get_release_list(artist_id=10263, per_page=5)
```

### Data Quality
Returns clean JSON with artist profiles, biographical data, discographies, group relationships

## Setup Notes

### Fixed Issues
- ✅ Accept header typo fixed (`application/json` not `applications/json`)
- ✅ Object serialization fixed (convert Discogs objects to dicts)
- ✅ List vs dict confusion resolved in album filtering

### Current Status
Both APIs stable and providing rich reference data for creative work.

## Creative Workflow Integration

### For White Album Development
1. Use EarthlyFrames API to pull previous album concepts/moods
2. Use Discogs API to research "sounds like" artists and expand musical DNA  
3. Cross-reference themes and ontological modes across Rainbow Table
4. Generate new concepts that bridge INFORMATION → TIME → SPACE progression

### Reference Workflow
```
# Get a color album for inspiration
album_for_color("yellow") 

# Research its "sounds like" artists
look_up_artist_by_name("David Bowie")
get_release_list(artist_id=10263)

# Pull specific songs for detailed analysis
get_song(album_id=8, song_id=56)  # Pulsar Palace "Entrance"
```
