"""Rechnet die Ereignisstudie zum Austritt aus der Altkonvention (vor_mechanism) und schreibt data/mechanism.json samt Erzeuger-Hash.

Aufruf (im Demo-Ordner, mit dem Test-venv):  python tools/make_mechanism.py
Ändert sich vor_mechanism.py, vor_learning.py, vor_games.py, vor_rng.py oder vor_constants.py, stimmt der Hash nicht mehr (tests/test_mechanism.py schlägt an); danach diesen Aufruf wiederholen.
Dauer: einige Minuten (4 Zellen zu je 4 Blöcken mit Aufzeichnung in jedem Schritt)."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vor_mechanism as M  # noqa: E402


def main():
    start = time.time()
    payload = M.run_all()
    M.write(payload)
    print(f"geschrieben: {M.FILE}  ({time.time() - start:.0f} s, Hash {M.generator_hash()}, Austritte je Zelle {[c['exits'] for c in payload['cells']]})")


if __name__ == "__main__":
    main()
