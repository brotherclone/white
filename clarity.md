# Notes on iterative song production

This document maps out some ideas on how to make the process of generating MIDI loops a bit clearer (at least for me).

## Phase 1: Chord Generation
Let's say there are four chord sequences produced in the shrinkewrapped directory:
- `thread_name/production/song_name/chords/candidates/chord_1.mid`
- `thread_name/production/song_name/chords/candidates/chord_2.mid`
- `thread_name/production/song_name/chords/candidates/chord_3.mid`
- `thread_name/production/song_name/chords/candidates/chord_4.mid`
as well as the review yaml:
- `thread_name/production/song_name/chords/review.yaml`
## Phase 2: Chord Generation Review
During this phase chords are marked as rejected or approved and given labels.
I've accepted and labeled:
- `chord_1.mid` as `outro`
- `chord_2.mid` as `chorus`
I've rejected:
- `chord_3.mid`
- `chord_4.mid`

## Phase 3 Chord Promotion:

I then run promotion and now have:

- `thread_name/production/song_name/chords/approved/outro.mid`
- `thread_name/production/song_name/chords/candidates/chorus.mid`

This really isn't enough to work with – so I go back to phase 1 and get:
- `thread_name/production/song_name/chords/candidates/chord_5.mid`
- `thread_name/production/song_name/chords/candidates/chord_6.mid`
- `thread_name/production/song_name/chords/candidates/chord_7.mid`
- `thread_name/production/song_name/chords/candidates/chord_8.mid`

I go back to phase 2 and accept `chord_5.mid` as `verse` and `chord_6.mid` as `bridge` and I like `chord_8.mid` as `chorus`.
I reject `chord_7.mid`. Again, I run promotion and now have:
- `thread_name/production/song_name/chords/approved/verse.mid`
- `thread_name/production/song_name/chords/approved/bridge.mid`
- `thread_name/production/song_name/chords/approved/chorus_2.mid`

## Phase 3: Drum Primitive Generation

Now I run drum generation and get:

- `thread_name/production/song_name/drums/candidates/drum_verse_01.mid`
- `thread_name/production/song_name/drums/candidates/drum_verse_02.mid`
- `thread_name/production/song_name/drums/candidates/drum_bridge_01.mid`
- `thread_name/production/song_name/drums/candidates/drum_bridge_02.mid`
- `thread_name/production/song_name/drums/candidates/drum_chorus_01.mid`
- `thread_name/production/song_name/drums/candidates/drum_chorus_02.mid`
- `thread_name/production/song_name/drums/candidates/drum_outro_01.mid`
- `thread_name/production/song_name/drums/candidates/drum_outro_02.mid`
- `thread_name/production/song_name/drums/candidates/drum_chorus_2_01.mid`

We are beginning to get some confusing names right here with chorus_2_01 – but nothing
too bad. I can easily find the chorus_2_01 candidate and listen to it. 

## Phase 4: Drum Primitive Review

During this phase I'm plugging the two parts into two tracks on the loop
grid for evaluation. I approve the following:

- `thread_name/production/song_name/drums/candidates/drum_outro_02.mid`
- `thread_name/production/song_name/drums/candidates/drum_chorus_2_01.mid`
- `thread_name/production/song_name/drums/candidates/drum_chorus_01.mid`

I don't have a verse drum pattern so I re-run Phase 3 and get:

- `thread_name/production/song_name/drums/candidates/drum_verse_03.mid`
- `thread_name/production/song_name/drums/candidates/drum_verse_04.mid`
- `thread_name/production/song_name/drums/candidates/drum_verse_05.mid`
- `thread_name/production/song_name/drums/candidates/drum_verse_06.mid`

These are all great so I approve them all.

## Phase 5: Drum Primitive Promotion

I then run promotion and now have:

- `thread_name/production/song_name/drums/approved/drum_verse_03.mid`
- `thread_name/production/song_name/drums/approved/drum_verse_04.mid`
- `thread_name/production/song_name/drums/approved/drum_verse_05.mid`
- `thread_name/production/song_name/drums/approved/drum_verse_06.mid`
- `thread_name/production/song_name/drums/approved/drum_outro_02.mid`
- `thread_name/production/song_name/drums/approved/drum_chorus_2_01.mid`
- `thread_name/production/song_name/drums/approved/drum_chorus_01.mid`
 
## Phase 5: (Optional) Harmonic Rhythm Generation

Many of these chord patterns are great, but the verse and outro are a bit sluggish because they are one chord per bar. So I run harmonic rhythm generation and get:

- `thread_name/production/song_name/harmonic_rhythm/candidates/hr_verse_001.mid`
- `thread_name/production/song_name/harmonic_rhythm/candidates/hr_verse_002.mid`
- `thread_name/production/song_name/harmonic_rhythm/candidates/hr_verse_003.mid`
- `thread_name/production/song_name/harmonic_rhythm/candidates/hr_verse_004.mid`
- `thread_name/production/song_name/harmonic_rhythm/candidates/hr_bridge_001.mid`
- `thread_name/production/song_name/harmonic_rhythm/candidates/hr_bridge_002.mid`
- `thread_name/production/song_name/harmonic_rhythm/candidates/hr_bridge_003.mid`
- `thread_name/production/song_name/harmonic_rhythm/candidates/hr_bridge_004.mid`
- `thread_name/production/song_name/harmonic_rhythm/candidates/hr_chorus_2_001.mid`
- `thread_name/production/song_name/harmonic_rhythm/candidates/hr_chorus_2_002.mid`
- `thread_name/production/song_name/harmonic_rhythm/candidates/hr_outro_001.mid`
- `thread_name/production/song_name/harmonic_rhythm/candidates/hr_outro_002.mid`
- `thread_name/production/song_name/harmonic_rhythm/candidates/hr_outro_003.mid`
- `thread_name/production/song_name/harmonic_rhythm/candidates/hr_outro_004.mid`
- `thread_name/production/song_name/harmonic_rhythm/candidates/hr_outro_005.mid`
- `thread_name/production/song_name/harmonic_rhythm/candidates/hr_chorus_001.mid`
- `thread_name/production/song_name/harmonic_rhythm/candidates/hr_chorus_002.mid`
- `thread_name/production/song_name/harmonic_rhythm/candidates/hr_chorus_003.mid`

The name confusion is piling up – but the real problem is that I chose four verse drum patterns. Which one was used?

## Phase 6: (Optional) Harmonic Rhythm Review

I approve the following:

- `thread_name/production/song_name/harmonic_rhythm/candidates/hr_verse_003.mid`
- `thread_name/production/song_name/harmonic_rhythm/candidates/hr_verse_004.mid`
- `thread_name/production/song_name/harmonic_rhythm/candidates/hr_chorus_002.mid`

I don't know if I'm supposed to relabel these new chords though?

## Phase 7: (Optional) Harmonic Rhythm Promotion

Now I have:

- `thread_name/production/song_name/harmonic_rhythm/approved/hr_verse_003.mid`
- `thread_name/production/song_name/harmonic_rhythm/approved/hr_verse_004.mid`
- `thread_name/production/song_name/harmonic_rhythm/approved/hr_chorus_002.mid`

But which drum do I align hr_verse_003 and hr_verse_004 with?

## Phase 8: (Optional) Strum Generation

At this point it is my understanding that if I have:
- `thread_name/production/song_name/harmonic_rhythm/approved/hr_verse_003.mid`
and:
- `thread_name/production/song_name/chords/approved/verse.mid`
that only the `hr_verse_003` will get a strum because of some regex stuff. I run strum generation and get:
- `thread_name/production/song_name/strums/candidates/hr_verse_003.mid`
- `thread_name/production/song_name/strums/candidates/hr_verse_004.mid`
- `thread_name/production/song_name/strums/candidates/hr_chorus_002.mid`
- `thread_name/production/song_name/strums/candidates/outro.mid`
- `thread_name/production/song_name/strums/candidates/bridge.mid`
- `thread_name/production/song_name/strums/candidates/chorus_2.mid`



## Phase 9: (Optional) Strum Review
I approve `chorus_2` and `hr_chorus_002` only.

## Phase 10: (Optional) Strum Promotion
I have:
- `thread_name/production/song_name/strums/approved/chorus_2.mid`
- `thread_name/production/song_name/strums/approved/hr_chorus_002.mid`

(skipping bass as it has the same problems as melody and melody is clearer)

## Phase 9: Melody Generation

For melody - I have officially lost the thread. But I think what I'd get is this:


- `thread_name/production/song_name/melody/candidates/melody_outro.mid`  (original)
- `thread_name/production/song_name/melody/candidates/melody_chorus.mid`  (original)
- `thread_name/production/song_name/melody/candidates/melody_bridge.mid`  (original)
- `thread_name/production/song_name/melody/candidates/melody_hr_verse_003.mid` (hr)
- `thread_name/production/song_name/melody/candidates/melody_hr_verse_004.mid` (hr)
- `thread_name/production/song_name/melody/candidates/melody_chorus_2.mid` (strums)
- `thread_name/production/song_name/melody/candidates/melody_hr_chorus_002.mid` (strums)

With perhaps multiple versions such as `melody_outro_001.mid` etc

## Phase 10: Melody Review

I can approve some - but who cares? They are used with anything.

## Phase 11: (Logic) Assembly
So here's what I have in my loop grid:
--
|        | Column 1          | Column 2          | Column 3   | Column 4                | Column 5                | Column 6            | Column 7                 |
|--------|-------------------|-------------------|------------|-------------------------|-------------------------|---------------------|--------------------------|
| Piano  | outro.mid         | chorus.mid        | bridge.mid | hr_verse_003.mid        | hr_verse_004.mid        | chorus_2.mid        | hr_chorus_002.mid        |
| Drums  | drum_outro_02.mid | drum_chorus_01    | !Opps      | drum_verse_04           | drum_verse_05           | drum_chorus_2_01    | ?                        | 
| Melody | melody_outro.mid  | melody_chorus.mid | !Opps      | melody_hr_verse_003.mid | melody_hr_verse_004.mid | melody_chorus_2.mid | melody_hr_chorus_002.mid |

Realizing I have nothing for the drums - I go back to phase 3 and run drum generation again.

## Phase 12: Manifest Generation

After arranging and looping the columns in the grid I copy the midi events to the arrangement.txt and this informs the manifest generation.

## Phase 13: Lyric Generation
Exporting a single melody file and turning over to an Agent (just Claude interfaces) we get lyrics.

## Phase 14: (Logic) Stem Creation
Using ACE Studio we align the lyrics to the melody and then export all tracks as audio
