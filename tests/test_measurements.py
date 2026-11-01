"""Die vorgerechneten Messreihen (data/measurements.json) gehören zum aktuellen Rechencode und werden von der Rechnung reproduziert.

Eigene Datei, weil jede Änderung an vor_games, vor_learning, vor_lewis, vor_scenario, vor_experiments oder vor_rng den Erzeuger-Hash ändert: so bleibt die Meldung „Messdatei neu erzeugen“ von den Tests
der Rechnung getrennt. Reproduziert wird plattformsicher: Zellen mit anderen Seeds (anderen Zellenbezeichnungen) innerhalb einer Marge, die aus der Streuung der Blöcke und einer festen
Untergrenze folgt; nichts an einzelnen Läufen."""

import json
import math

import pytest

import vor_constants as C
import vor_experiments as XP
import vor_games as G
import vor_scenario as S


@pytest.fixture(scope="module")
def stored():
    return json.loads(XP.MEAS_FILE.read_text(encoding="utf-8"))


def test_file_matches_the_generator_hash(stored):
    assert stored["generator_hash"] == XP.generator_hash(), "vor_games/learning/lewis/scenario/experiments/rng wurde geändert: python tools/make_measurements.py ausführen"
    assert XP.load_measurements() is not None


def test_load_returns_none_for_a_stale_hash_and_for_a_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(XP, "generator_hash", lambda: "0000000000000000")
    assert XP.load_measurements() is None
    monkeypatch.undo()
    monkeypatch.setattr(XP, "MEAS_FILE", tmp_path / "fehlt.json")
    assert XP.load_measurements() is None


def test_cells_and_axes_match_the_code(stored):
    assert stored["blocks"] == C.MEAS_BLOCKS and stored["runs_per_block"] == C.MEAS_RUNS
    specs = XP.cell_specs()
    assert [c["spec"] for c in stored["cells"]] == json.loads(json.dumps(specs))
    assert all(c["result"]["blocks"] == C.MEAS_BLOCKS and c["result"]["runs"] == C.MEAS_BLOCKS * C.MEAS_RUNS for c in stored["cells"])
    assert [(c["n"], c["ns"]) for c in stored["lewis"]] == list(C.LEWIS_PAIRS)
    assert all(c["runs"] == C.LEWIS_BLOCKS * C.LEWIS_RUNS and set(str(k) for k in C.LEWIS_CHECKS) <= set(c) for c in stored["lewis"])


def test_every_cell_spec_is_unique_and_has_its_own_seed_label():
    labels = [XP.spec_label(s) for s in XP.cell_specs()]
    assert len(set(labels)) == len(labels) == 86
    assert len(XP.jobs()) == 86 * C.MEAS_BLOCKS + len(C.LEWIS_PAIRS)
    keys = [j[1] for j in XP.jobs()]
    assert len(set(keys)) == len(keys)


def walk(obj):
    """Alle Kennzahl-Einträge {"mean", "se"} der Datei."""
    if isinstance(obj, dict):
        if set(obj) == {"mean", "se"}:
            yield obj
        else:
            for v in obj.values():
                yield from walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v)


def test_every_stored_statistic_is_finite_and_has_a_nonnegative_error(stored):
    count = none = 0
    for st in walk({"cells": stored["cells"], "lewis": stored["lewis"]}):
        count += 1
        if st["mean"] is None:
            none += 1
            assert st["se"] is None
            continue
        assert math.isfinite(st["mean"]) and st["se"] >= 0.0 and math.isfinite(st["se"]), st
    assert count > 2000


def test_the_columns_are_no_dead_branches(stored):
    """Nullspalten-Regel: keine Kennzahl, die in allen Zellen einer Familie gleich ist (die Variante liefe nie)."""
    by_family = {}
    for c in stored["cells"]:
        by_family.setdefault(c["spec"]["family"], []).append(c["result"])
    assert set(by_family) == {"e3", "e3n", "hs", "hsn", "bos", "rand"}
    for family, cells in by_family.items():
        for key in ("eff", "aa", "bb", "mis", "p1", "info"):
            vals = {round(c[key]["mean"], 6) for c in cells}
            assert len(vals) > 1, (family, key)


def _fresh_agrees(a, b, keys, floor=0.05):
    for key in keys:
        tol = max(4 * math.hypot(a[key]["se"], b[key]["se"]), floor)
        assert abs(a[key]["mean"] - b[key]["mean"]) < tol, (key, a[key]["mean"], b[key]["mean"], tol)


def test_a_cell_without_talk_is_reproduced_with_other_seeds(stored):
    spec = next(c["spec"] for c in stored["cells"] if c["spec"]["family"] == "bos" and c["spec"]["talk"] == G.TALK_NONE and c["spec"]["start"] == C.START_RANDOM)
    fresh = XP.cell(XP.settings_of_spec(spec, C.MEAS_RUNS), C.MEAS_BLOCKS, "neu|bos|keins")
    old = XP.find_cell(stored, family="bos", talk=G.TALK_NONE, start=C.START_RANDOM)
    _fresh_agrees(old, fresh, ("p1", "p2", "eff"), floor=0.1)


def test_a_one_sided_cell_is_reproduced_with_other_seeds(stored):
    spec = next(c["spec"] for c in stored["cells"] if c["spec"]["family"] == "bos" and c["spec"]["talk"] == G.TALK_ONE and c["spec"]["K"] == 2 and c["spec"]["start"] == C.START_RANDOM)
    fresh = XP.cell(XP.settings_of_spec(spec, C.MEAS_RUNS), C.MEAS_BLOCKS, "neu|bos|einseitig")
    old = XP.find_cell(stored, family="bos", talk=G.TALK_ONE, K=2, start=C.START_RANDOM)
    _fresh_agrees(old, fresh, ("p1", "p2", "eff"), floor=0.1)


def test_an_escape_cell_is_reproduced_with_other_seeds(stored):
    spec = next(c["spec"] for c in stored["cells"] if c["spec"]["family"] == "e3" and c["spec"]["talk"] == G.TALK_ONE and c["spec"]["K"] == 3 and abs(c["spec"]["b"] - 3.6) < 1e-9 and c["spec"]["beta"] == 20.0)
    fresh = XP.cell(XP.settings_of_spec(spec, C.MEAS_RUNS), C.MEAS_BLOCKS, "neu|e3|einseitig3", True)
    old = XP.find_cell(stored, family="e3", talk=G.TALK_ONE, K=3, b=3.6, beta=20.0)
    _fresh_agrees(old, fresh, ("aa", "exit"), floor=0.15)


def test_a_lewis_cell_is_reproduced_with_other_seeds(stored):
    fresh = XP.lewis_cell(3, 3, 100, 2, 10_000, (1_000, 10_000), "neu|lewis|3|3")
    old = XP.find_lewis(stored, 3, 3)
    for c in ("1000", "10000"):
        _fresh_agrees(old[c], fresh[c], ("eff", "share"), floor=0.08)


def test_assemble_is_independent_of_the_order_in_which_the_results_arrive():
    jobs = XP.jobs()
    results = {}
    for j in jobs:
        if j[0] == "block":
            results[j[1]] = {"x": float(len(results))}
        else:
            results[j[1]] = {"n": j[2][0], "ns": j[2][1], "mark": float(len(results))}
    shuffled = dict(reversed(list(results.items())))
    a, b = XP.assemble(results), XP.assemble(shuffled)
    assert json.dumps(a) == json.dumps(b)


def test_find_cell_requires_exactly_one_match(stored):
    with pytest.raises(KeyError):
        XP.find_cell(stored, family="gibt es nicht")
    with pytest.raises(KeyError):
        XP.find_cell(stored, family="e3")
    with pytest.raises(KeyError):
        XP.find_lewis(stored, 9, 9)
    assert isinstance(S.Settings(), S.Settings)
