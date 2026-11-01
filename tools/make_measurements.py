"""Rechnet alle vorgerechneten Messreihen (Altkonvention verlassen, Handschlag, Kampf der Geschlechter, zufälliger Start, Lewis-Spiel) neu und schreibt data/measurements.json samt Erzeuger-Hash.

Aufruf (im Demo-Ordner, mit dem Test-venv):  python tools/make_measurements.py [--workers 14]
Ändert sich eine der Dateien vor_games.py, vor_learning.py, vor_lewis.py, vor_scenario.py, vor_experiments.py oder vor_rng.py, stimmt der Hash nicht mehr (tests/test_measurements.py schlägt an, die App
zeigt die vorgerechneten Reihen dann nicht); danach diesen Aufruf wiederholen. Jeder Block hat einen festen Zufalls-Seed aus der Zellenbezeichnung (vor_experiments.seed_of), die Reihen sind also bei
gleichem Code Zahl für Zahl dieselben, auch bei anderer Zahl von Prozessen und anderer Reihenfolge der Ergebnisse. Dauer: einige Minuten bis eine halbe Stunde je nach Rechner."""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vor_experiments as XP  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    start = time.time()
    jobs = XP.jobs()
    # die langen Aufträge zuerst, damit am Ende keine Lücke entsteht
    order = sorted(range(len(jobs)), key=lambda i: -(jobs[i][2]["steps"] * jobs[i][2]["n_agents"] if jobs[i][0] == "block" else 1e6))
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(XP.run_job, jobs[i]) for i in order]
        for done, f in enumerate(as_completed(futures), 1):
            key, res = f.result()
            results[key] = res
            if done % 10 == 0 or done == len(jobs):
                print(f"[{time.time() - start:6.1f} s] {done}/{len(jobs)} Aufträge", flush=True)
    payload = XP.assemble(results)
    XP.write_measurements(payload)
    print(f"geschrieben: {XP.MEAS_FILE}  ({time.time() - start:.0f} s, Hash {XP.generator_hash()}, {len(jobs)} Aufträge, {payload['blocks']} Blöcke zu je {payload['runs_per_block']} Läufen)")


if __name__ == "__main__":
    main()
