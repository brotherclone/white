# Proposal: wire-singer-voices-registry

## Summary

Wire `app/reference/mcp/ace_studio/singer_voices.yml` into `ace_studio_export.py`
so that the correct ACE Studio voice name is used when loading a singer, rather than
the White project name which ACE Studio does not recognise.

## Problem

`ace_studio_export.py` calls `ace.find_singer(singer_name)` with the White project
singer name (e.g. `"Shirley"`). ACE Studio's sound source library uses different
names (e.g. `"Ember Rose"`). The lookup always fails for mapped singers, so no voice
is ever loaded automatically.

`singer_voices.yml` already contains the White → ACE name mapping but is not read
anywhere in the export path.

## Scope

- **In scope**: Load `singer_voices.yml` in `ace_studio_export.py`; resolve the ACE
  voice name before calling `find_singer()`; define clear fallback behaviour when
  `ace_studio_voice` is `null`.
- **Out of scope**: Consolidating MIDI ranges from `singer_voices.yml` into
  `melody_patterns.py` — that is a separate concern touching melody generation and
  its tests. It can be a follow-up spec once all voices are confirmed.

## Design Decisions

### Fallback for `null` ace_studio_voice

When a singer's `ace_studio_voice` entry is `null` (voice not yet confirmed in ACE
Studio), the export should:

1. Log a warning naming the singer and stating the voice is unassigned.
2. Attempt `find_singer()` with the White project name as a best-effort fallback
   (the operator may have manually loaded a voice with that name).
3. If that also finds nothing, skip singer loading and continue — same as the current
   not-found path.

This is more useful than hard-failing: the human operator can still select the voice
manually in ACE Studio, and we don't block the export.

### Fallback when singer not in registry

If the singer name from the review file is absent from `singer_voices.yml` entirely
(e.g. a test/legacy name), fall back to passing the name directly to `find_singer()`
with a debug log. This preserves compatibility with any names not yet added to the
YAML.

### Registry loading

Load the YAML once per export call (not module-level) via a small helper
`load_singer_registry(path)` that returns a dict. This keeps the function testable
and avoids import-time side effects.

## Files Affected

- `app/generators/midi/production/ace_studio_export.py` — add registry lookup
- `app/reference/mcp/ace_studio/singer_voices.yml` — no structural changes needed;
  confirm Busyayo spelling matches `melody_patterns.py` ("busyayo" key)
- `tests/generators/midi/production/test_ace_studio_export.py` — new/updated tests

## Out of Scope

- MIDI range consolidation from YAML into `melody_patterns.py`
- Adding unconfirmed voices to `singer_voices.yml`
- Changes to `melody_pipeline.py` or `melody_patterns.py`
