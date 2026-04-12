#!/usr/bin/env zsh
# White project shell aliases — source this from ~/.zshrc:
#   source /Volumes/LucidNonsense/White/white_aliases.sh
#
# ── Standard pipeline ──────────────────────────────────────────────────────
#   wpipe status  --production-dir <path>
#   wpipe run     --production-dir <path>
#   wpipe next    --production-dir <path>
#   wpipe promote --production-dir <path>
#   wscore --mix-file <audio> --production-dir <path>
#   wshrink / wshrink --thread <uuid> / wshrink --dry-run
#   wstart / wstart --with-html / wstart --no-browser
#
# ── Evolutionary pattern breeding ──────────────────────────────────────────
#   wevolve drums   --production-dir <path>              (8 generations, pop 30)
#   wevolve bass    --production-dir <path> --generations 16
#   wevolve melody  --production-dir <path>
#   Evolved candidates appear in review.yml alongside normal ones (is_evolved: true)
#
# ── ACE Studio vocal synthesis ─────────────────────────────────────────────
#   wpipe ace export --production-dir <path>   push MIDI+lyrics → ACE Studio
#   wpipe ace status --production-dir <path>   check association
#   wpipe ace import --production-dir <path>   ingest WAV render ← ACE Studio
#   (ACE Studio 2.0 must be running at localhost:21572)
#
# ── Other ──────────────────────────────────────────────────────────────────
#   wnc          generate negative constraints
#   wopenspec    openspec CLI wrapper
#   wdash        song completion dashboard

WHITE=/Volumes/LucidNonsense/White

function wpipe()   { (cd "$WHITE" && python -m app.generators.midi.production.pipeline_runner "$@") }
function wscore()  { (cd "$WHITE" && python -m app.generators.midi.production.score_mix "$@") }
function wshrink() { (cd "$WHITE" && python -m app.util.shrinkwrap_chain_artifacts "$@") }
function wstart()  { (cd "$WHITE" && python -m run_white_agent start "$@") }
function wnc()     { (cd "$WHITE" && python -m app.util.generate_negative_constraints "$@") }
function wopenspec() { (cd "$WHITE" && openspec "$@") }
function wdash()   { (cd "$WHITE" && python -m app.tools.song_dashboard "$@") }

# Evolutionary pattern breeding — wevolve <drums|bass|melody> --production-dir <path> [--generations N]
function wevolve() {
  local phase=$1; shift
  case "$phase" in
    drums)  (cd "$WHITE" && python -m app.generators.midi.pipelines.drum_pipeline --evolve "$@") ;;
    bass)   (cd "$WHITE" && python -m app.generators.midi.pipelines.bass_pipeline --evolve "$@") ;;
    melody) (cd "$WHITE" && python -m app.generators.midi.pipelines.melody_pipeline --evolve "$@") ;;
    *)      echo "Usage: wevolve <drums|bass|melody> --production-dir <path> [--generations N] [--population N]" ;;
  esac
}
