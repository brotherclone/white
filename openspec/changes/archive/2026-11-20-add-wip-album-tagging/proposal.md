# Change: Tag White Album WiP mp3s with a shared album title and track number

## Why
The WiP mp3s copied into `White Album WiP/` keep whatever album tag their mix export
carried (often none, or a per-session value), so Apple Music scatters them across
multiple "albums" and plays them in arbitrary order. Stamping one shared album title
and a track number derived from `sides.yml` makes the import group into a single
album that plays in Side A→D order.

## What Changes
- Add `wip_album_title` to `playlist_config.yml` (`PlaylistConfig`), default
  `"White Album WiP"`.
- During sync, after copying each mp3 into `White Album WiP/`, write ID3 tags on the
  **copy only**:
  - title (`TIT2`) from `song_context.yml`: `song_slug` when non-blank, else its
    `title`, else the scanned title (filenames keep using the scanned title)
  - album (`TALB`) = `wip_album_title`
  - track number (`TRCK`) = `seq/total` for sequenced songs, using the same global
    1-based counter already used for the `{seq:02d}_` filename prefix; unsequenced
    songs get no track number (any existing one is cleared).
- Rejects and Review copies, non-mp3 mixes, and the source mix files are never tagged
  or modified.
- `GET/POST /playlists/config` expose `wip_album_title`; POST becomes a partial update
  so setting one field no longer resets the other.
- New dependency: `mutagen` in `white-composition`.

## Impact
- Affected specs: `listening-playlist-sync`
- Affected code: `packages/composition/src/white_composition/playlist_sync.py`,
  `packages/api/src/white_api/candidate_server.py` (config endpoints),
  `packages/client/lib/api.ts` (type only), `packages/composition/pyproject.toml`
- No UI field for the title in this change — edit `playlist_config.yml` or call
  `POST /playlists/config`.
