"""Die Ereignisstudie (data/mechanism.json) gehört zum aktuellen Rechencode, hat die erwartete Gestalt und wird von einer kleinen Neuberechnung reproduziert."""

import json
import math

import pytest

import vor_experiments as XP
import vor_mechanism as M


@pytest.fixture(scope="module")
def stored():
    return json.loads(M.FILE.read_text(encoding="utf-8"))


def test_file_matches_the_generator_hash(stored):
    assert stored["generator_hash"] == M.generator_hash(), "vor_mechanism/learning/games/rng/constants wurde geändert: python tools/make_mechanism.py ausführen"
    assert M.load() is not None


def test_load_returns_none_for_a_stale_hash_and_for_a_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "generator_hash", lambda: "0000000000000000")
    assert M.load() is None
    monkeypatch.undo()
    monkeypatch.setattr(M, "FILE", tmp_path / "fehlt.json")
    assert M.load() is None


def test_shape_of_the_stored_study(stored):
    assert [(c["b"], c["K"], c["beta"]) for c in stored["cells"]] == list(M.CELLS)
    for c in stored["cells"]:
        assert c["runs"] == M.RUNS * M.BLOCKS and c["steps"] == M.STEPS and c["n_agents"] == M.N and c["lags"] == list(M.LAGS)
        for key in ("n", "s", "a"):
            assert set(c[key]) == {str(k) for k in M.LAGS} and all(len(v) == c["exits"] for v in c[key].values())
        assert all(1 <= x <= M.N for x in c["n"]["0"]) and all(0.0 <= x <= 1.0 for x in c["s"]["20"])
    assert stored["lottery"]["runs"] == M.RUNS * M.BLOCKS and 0.0 <= stored["lottery"]["mean_share"] <= 1.0


def test_summary_of_a_hand_cell():
    cell = {"lags": [100, 0], "n": {"100": [4, 6], "0": [10, 12]}, "s": {"100": [0.2, 0.4], "0": [0.6, 0.8]}, "a": {"100": [0, 1], "0": [10, 10]}, "limit": 4}
    sm = M.summary(cell)
    assert sm[100] == {"n": 5.0, "s": pytest.approx(0.3), "a": 0.5, "within_limit": 0.5} and sm[0]["within_limit"] == 0.0
    empty = {"lags": [0], "n": {"0": []}, "s": {"0": []}, "a": {"0": []}, "limit": 1}
    assert M.summary(empty)[0]["n"] is None


def test_event_study_invariants_on_a_small_run():
    """Kleine Studie (6 Läufe, N = 10): die mittlere Zahl der Sprecher je Nachricht ist vor dem Austritt genau N / K (die Sprecher verteilen sich auf die K Nachrichten); die Austrittsnachricht hat im
    Austrittsschritt mindestens N / 2 A-Sprecher (sie hat die meisten, ein Austritt heißt: Hälfte der Sprecher spielt A, höchstens K Nachrichten teilen sie); a ≤ n; Hörer-Anteile in [0, 1]."""
    b, K, beta, N, steps, runs, seed = 3.6, 2, 20.0, 10, 3000, 6, 11
    res = M.event_study(b, K, beta, N, steps, runs, seed)
    assert res["mean_n_before"] == pytest.approx(N / K)
    assert 0 <= res["exits"] <= runs
    for k in M.LAGS:
        assert len(res["n"][str(k)]) == len(res["s"][str(k)]) == len(res["a"][str(k)]) == res["exits"]
    for n, a in zip(res["n"]["0"], res["a"]["0"]):
        assert a <= n <= N and a >= math.ceil(N / 2 / K)
    for k in M.LAGS:
        assert all(0 <= a <= n for n, a in zip(res["n"][str(k)], res["a"][str(k)])) and all(0.0 <= v <= 1.0 for v in res["s"][str(k)])
    same = M.event_study(b, K, beta, N, steps, runs, seed)
    assert same == res                                                                     # feste Zufalls-Seeds, derselbe Lauf


def test_lottery_study_returns_one_share_per_run():
    res = M.lottery_study(20.0, 10, 400, 5, 3)
    assert len(res["shares"]) == 5 and all(0.0 <= v <= 1.0 for v in res["shares"]) and res["uniform"] == pytest.approx(2 / 32)


def test_one_cell_is_reproduced_with_other_seeds(stored):
    cell = stored["cells"][0]                                                                 # p* = 0,4, K = 2, β = 20: 191 von 800 Austritten
    fresh = M.event_study(cell["b"], cell["K"], cell["beta"], M.N, M.STEPS, 400, XP.seed_of("neu|mech|0"))
    share_old, share_new = cell["exits"] / cell["runs"], fresh["exits"] / fresh["runs"]
    tol = 4 * math.sqrt(share_old * (1 - share_old) / cell["runs"] + share_new * (1 - share_new) / fresh["runs"]) + 0.03
    assert abs(share_old - share_new) < tol
    sm_old, sm_new = M.summary(cell), M.summary({**fresh, "lags": list(M.LAGS), "limit": cell["limit"]})
    assert abs(sm_old[100]["n"] - sm_new[100]["n"]) < 1.5 and abs(sm_old[20]["s"] - sm_new[20]["s"]) < 0.08
