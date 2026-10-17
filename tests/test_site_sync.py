"""The static site carries copies of the game twin and its spec; they must be byte-identical to the sources."""
import hashlib
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def test_site_game_engine_is_the_tested_engine():
    assert sha("site/js/game.mjs") == sha("game/raycaster.mjs"), "run tools/sync_site.sh"


def test_site_spec_is_the_tested_spec():
    assert sha("site/js/spec.json") == sha("game/spec.json"), "run tools/sync_site.sh"
