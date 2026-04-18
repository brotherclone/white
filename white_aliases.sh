#!/usr/bin/env zsh
# White project shell aliases — source this from ~/.zshrc:
#   source /Volumes/LucidNonsense/White/white_aliases.sh
#
# ── Song generation ────────────────────────────────────────────────────────
#   wstart / wstart --with-html / wstart --no-browser
#   wshrink / wshrink --thread <uuid> / wshrink --dry-run
#
# ── Production pipeline ────────────────────────────────────────────────────
#   winit  --song-proposal <path>               (auto-derives production dir)
#   wpipe run    --production-dir <path>        (first run: add --song-proposal)
#   wpipe status --production-dir <path>
#   wpipe promote --production-dir <path> --phase <phase>
#   wreset --production-dir <path> --phase <phase>
#
# ── Evolutionary pattern breeding ──────────────────────────────────────────
#   wevolve drums   --production-dir <path>
#   wevolve bass    --production-dir <path>
#   wevolve melody  --production-dir <path>
#
# ── Review UI ──────────────────────────────────────────────────────────────
#   wui --production-dir <path>   launch API server + Next.js UI
#   wui-api --production-dir <path>   backend only (port 8000)
#   wui-web                           Next.js frontend only (port 3000)
#
# ── Other ──────────────────────────────────────────────────────────────────
#   wnc          generate negative constraints
#   wopenspec    openspec CLI wrapper
#   wdash        song completion dashboard

WHITE=/Volumes/LucidNonsense/White

function winit()   { (cd "$WHITE" && python -m app.generators.midi.production.init_production "$@") }
function wreset()  { (cd "$WHITE" && python -c "
from app.generators.midi.production.pipeline_runner import write_phase_status
from pathlib import Path
import sys, argparse
p = argparse.ArgumentParser(); p.add_argument('--production-dir',required=True); p.add_argument('--phase',required=True)
a = p.parse_args()
write_phase_status(Path(a.production_dir), a.phase, 'pending')
print(f'Reset {a.phase} -> pending')
" "$@") }
function wui-api() { (cd "$WHITE" && python -m app.tools.candidate_server "$@") }
function wui-web() { (cd "$WHITE/web" && nvm use 20 && npm run dev) }
function wui() {
  local prod_dir=""
  for arg in "$@"; do
    if [[ -n "$prod_dir_next" ]]; then prod_dir="$arg"; prod_dir_next=""; fi
    [[ "$arg" == "--production-dir" ]] && prod_dir_next=1
  done
  (cd "$WHITE" && python -m app.tools.candidate_server --production-dir "$prod_dir" &)
  (cd "$WHITE/web" && nvm use 20 && npm run dev)
}
function wpipe()   { (cd "$WHITE" && python -m app.generators.midi.production.pipeline_runner "$@") }
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
