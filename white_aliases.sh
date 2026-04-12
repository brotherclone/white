#!/usr/bin/env zsh
# White project shell aliases — source this from ~/.zshrc:
#   source /Volumes/LucidNonsense/White/white_aliases.sh
#
# Usage examples (all take the same args as the underlying commands):
#   wpipe status --production-dir <path>
#   wpipe run    --production-dir <path>
#   wpipe next   --production-dir <path>
#   wpipe promote --production-dir <path>
#   wpipe ace    --production-dir <path>
#   wscore --mix-file <audio> --production-dir <path>
#   wshrink
#   wshrink --thread <uuid>
#   wshrink --dry-run
#   wstart
#   wstart --with-html
#   wnc      (generate negative constraints)
#   wopenspec validate <id> --strict

WHITE=/Volumes/LucidNonsense/White

function wpipe()   { (cd "$WHITE" && python -m app.generators.midi.production.pipeline_runner "$@") }
function wscore()  { (cd "$WHITE" && python -m app.generators.midi.production.score_mix "$@") }
function wshrink() { (cd "$WHITE" && python -m app.util.shrinkwrap_chain_artifacts "$@") }
function wstart()  { (cd "$WHITE" && python -m run_white_agent start "$@") }
function wnc()     { (cd "$WHITE" && python -m app.util.generate_negative_constraints "$@") }
function wopenspec() { (cd "$WHITE" && openspec "$@") }
function wdash()   { (cd "$WHITE" && python -m app.tools.song_dashboard "$@") }
