#!/bin/bash
# Copies the game twin and its spec into the static site (the site folder must be self-contained for
# `python3 -m http.server` and for static hosting). tests/test_site_sync.py checks the copies are identical.
set -euo pipefail
cd "$(dirname "$0")/.."
cp game/raycaster.mjs site/js/game.mjs
cp game/spec.json site/js/spec.json
echo "synced game.mjs and spec.json into site/js/"
