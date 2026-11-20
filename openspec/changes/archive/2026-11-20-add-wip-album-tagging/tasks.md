## 1. Config
- [x] 1.1 Add `wip_album_title: str = "White Album WiP"` to `PlaylistConfig`; confirm an existing `playlist_config.yml` without the key loads with the default
- [x] 1.2 Add `mutagen` to `packages/composition/pyproject.toml` and `uv sync`

## 2. Tagging
- [x] 2.1 Add a `_tag_wip_copy(path, album, track, total)` helper using `mutagen.id3` that writes `TALB`, writes or clears `TRCK`, and creates an ID3 header if the file has none
- [x] 2.2 Thread `wip_album_title` into `sync_playlists()` and apply tagging only for `.mp3` files in the WiP rebuild (sequenced → `seq/total`, unsequenced → no track)
- [x] 2.3 Tagging failure on one file logs a warning and leaves the untagged copy in place; it does not abort the sync

- [x] 2.4 Write `TIT2` on WiP copies from `song_context.yml` (`song_slug`, else `title`, else scanned title); filenames unchanged

## 3. API / client
- [x] 3.1 `GET /playlists/config` returns `wip_album_title`
- [x] 3.2 `POST /playlists/config` accepts optional `output_dir` and `wip_album_title`, updating only the fields provided
- [x] 3.3 Update the playlist config type in `packages/client/lib/api.ts`

## 4. Tests
- [x] 4.1 WiP mp3 copy has `TALB` = configured title and `TRCK` = `seq/total`
- [x] 4.2 Unsequenced WiP mp3 has the album tag and no track number
- [x] 4.3 Source mix file bytes are unchanged after sync
- [x] 4.4 Rejects/Review copies keep the source's original tags
- [x] 4.5 Non-mp3 WiP mix is copied untagged without error
- [x] 4.7 Title tag uses context `song_slug`, then context `title`, then scanned title; filename unchanged
- [x] 4.6 POST with only `wip_album_title` preserves `output_dir` (and vice versa)

## 5. Finish
- [x] 5.1 Minor-bump `white-composition` and `white-api` versions (and `client` if versioned)
- [x] 5.2 `graphify update .`
