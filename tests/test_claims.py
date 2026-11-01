"""Jede Zahl aus der README und den Hilfetexten, nachgerechnet (DEMO-PLAYBOOK §4).

Drei Quellen, alle plattformsicher:
* die gespeicherten Messreihen (`data/measurements.json`: 4 Blöcke zu je 200 Läufen je Zelle, Erzeuger-Hash geprüft in `test_measurements.py`) und die Ereignisstudie (`data/mechanism.json`,
  `test_mechanism.py`): die README-Zahl ist der gespeicherte Wert, gerundet; eine Stichprobe der Zellen wird mit anderen Seeds innerhalb einer Marge reproduziert. Das Lesen der Dateien hängt nicht
  von der Plattform ab;
* exakte Rechnung (Aufzählung, Brüche, Formeln): Gleitkommagleichheit nur mit `approx`;
* die Zeilenbauer der App (`vor_evaluation.*_table`): jede Zeile der README-Tabellen steht als Text in der README (Test `test_readme_tables_are_built_by_the_same_row_builders`).
Keine Zeit-Assertion, keine exakten Zufallswerte einzelner Läufe. Zählungen über das Raster („in k von m Zellen“) stehen als Zählungen im Test, nicht als Urteilswort."""

import math
import re
from pathlib import Path

import pytest

import vor_constants as C
import vor_evaluation as V
import vor_experiments as XP
import vor_games as G
import vor_lewis as LW
import vor_mechanism as M
import vor_texts as T

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def r(x, nd=3):
    return round(x, nd)


@pytest.fixture(scope="module")
def meas():
    return XP.load_measurements()


@pytest.fixture(scope="module")
def mech():
    return M.load()


def m(c, key):
    return c[key]["mean"]


def e3(meas, p, talk, K, beta):
    return V.e3_cell(meas, p, talk, K, beta)


TALKS = C.E3_TALKS                                  # kein Reden, einseitig K = 1 / 2 / 3, beidseitig K = 2


# --- H1: die Gleichgewichtsmenge bleibt -------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def e1():
    return V.e1_rows(C.DEFAULT_B, 0.0)


def test_h1_payoff_sets_are_unchanged_in_8_of_8_and_16_of_16_cells(e1):
    base = {x["label"]: x for x in e1 if x["talk"] == G.TALK_NONE}
    exact = [x for x in e1 if x["all_payoffs"] is not None and x["talk"] != G.TALK_NONE]
    assert len(exact) == 8 and sum(1 for x in exact if V.same_points(base[x["label"]]["all_payoffs"], x["all_payoffs"])) == 8
    pure = [x for x in e1 if x["talk"] != G.TALK_NONE]
    assert len(pure) == 16 and sum(1 for x in pure if V.same_points(base[x["label"]]["pure_payoffs"], x["pure_payoffs"])) == 16
    assert not any(x["miscoordination"] for x in e1)


def test_h1_numbers_of_pure_equilibria(e1):
    def counts(label):
        return [x["pure"] for x in e1 if x["label"] == label]
    assert counts("Reine Koordination") == [2, 2, 8, 24, 512]
    assert counts("Hirschjagd") == [2, 2, 6, 15, 320] and counts("Hirschjagd mit Verlockung t = 1") == [2, 2, 6, 15, 320]
    assert counts("Kampf der Geschlechter") == [2, 2, 6, 15, 256]
    three = {lab: V.two_sided_three(g, 7.0, t) for lab, g, t in V.E1_GAMES}
    assert [three[k]["pure"] for k in ("Reine Koordination", "Hirschjagd", "Hirschjagd mit Verlockung", "Kampf der Geschlechter")] == [1179648, 626688, 626688, 294912]
    assert all(v["strategies"] == 1536 for v in three.values())
    assert three["Hirschjagd"]["pure_payoffs"] == [(7.0, 7.0), (9.0, 9.0)] and three["Kampf der Geschlechter"]["pure_payoffs"] == [(1.0, 2.0), (2.0, 1.0)]


def test_h1_mixed_equilibria_without_talk(e1):
    by = {x["label"]: x for x in e1 if x["talk"] == G.TALK_NONE}
    assert by["Hirschjagd mit Verlockung t = 1"]["all_payoffs"] == [(7.0, 7.0), (7.875, 7.875), (9.0, 9.0)]
    assert [r(a) for a, _ in by["Kampf der Geschlechter"]["all_payoffs"]] == [0.667, 1.0, 2.0]
    assert [a for a, _ in by["Reine Koordination"]["all_payoffs"]] == [0.5, 1.0]


# --- H2: informative Gleichgewichte und die Verlockung ----------------------------------------------------------------------------------------
def test_h2_informative_equilibria(e1):
    k2 = {x["label"]: x for x in e1 if x["talk"] == G.TALK_ONE and x["K"] == 2}
    assert {lab: x["informative"] for lab, x in k2.items()} == {"Reine Koordination": True, "Hirschjagd": True, "Hirschjagd mit Verlockung t = 1": False, "Kampf der Geschlechter": False}
    assert r(k2["Reine Koordination"]["difference"]) == 1.0 and r(k2["Hirschjagd"]["difference"]) == 0.778


def test_h2_every_lure_destroys_the_informative_equilibrium_at_b_7():
    rows = V.e2_rows(7.0, (G.TALK_ONE, 2))
    assert [(x["t"], x["informative"]) for x in rows] == [(0.0, True), (0.25, False), (0.5, False), (1.0, False)]
    assert [r(x["pstar"]) for x in rows] == [0.778, 0.8, 0.824, 0.875]
    for b in (3.6, 4.5, 5.4):
        assert [x["informative"] for x in V.e2_rows(b, (G.TALK_ONE, 2))] == [True, False, False, False]


# --- H3: die Lotterie -------------------------------------------------------------------------------------------------------------------------
def test_h3_the_lottery_is_an_equilibrium_in_4_of_4_games_with_these_payoffs():
    rows = V.e5_rows(7.0, 0.0)
    assert all(x["lottery"]["equilibrium"] for x in rows)
    assert [(x["lottery"]["payoff1"], x["lottery"]["payoff2"]) for x in rows] == [(1.0, 1.0), (8.0, 8.0), (8.0, 8.0), (1.5, 1.5)]
    text = V.verdict_e5(rows, None)[1]
    assert "in 4 von 4 Spielen ein Gleichgewicht" in text and "des schlechtesten Gleichgewichts ohne Reden in 4 von 4 Spielen" in text and "des besten in 2 von 4 (Reine Koordination, Kampf der Geschlechter)" in text


def test_h3_the_lottery_is_not_learned(mech):
    lot = mech["lottery"]
    assert (r(lot["mean_share"]), r(lot["uniform_share"], 4), lot["runs_at_least_half"], lot["runs"], r(lot["max_share"])) == (0.055, 0.0625, 0, 800, 0.4)
    assert lot["mean_share"] < lot["uniform_share"] + 0.02


# --- H4 und H7: zufälliger Start --------------------------------------------------------------------------------------------------------------
def test_h4_stag_hunt_is_not_solved_by_talk_at_random_start(meas):
    cells = [(g_t, tk, k, beta, V.rand_cell(meas, G.GAME_HIRSCH, g_t, tk, k, beta)) for g_t in (0.0, 1.0) for tk, k in TALKS for beta in C.RANDOM_BETAS]
    assert len(cells) == 20 and sum(1 for *_, c in cells if m(c, "aa") <= 0.03) == 20
    worst = max(cells, key=lambda x: m(x[4], "aa"))
    assert (worst[1], worst[2], worst[3], r(m(worst[4], "aa"))) == (G.TALK_TWO, 2, 6.0, 0.025)
    assert all(m(c, "aa") == 0.0 or m(c, "aa") < 0.01 for g_t, tk, k, beta, c in cells if beta == 20.0)
    assert [r(m(V.rand_cell(meas, G.GAME_HIRSCH, 0.0, tk, k, 20.0), "bb")) for tk, k in TALKS] == [1.0, 1.0, 0.998, 0.986, 0.997]


def test_h7_more_strategies_more_miscoordination_at_equal_beta(meas):
    mis = [m(V.rand_cell(meas, G.GAME_HIRSCH, 0.0, tk, k, 6.0), "mis") for tk, k in TALKS]
    assert [r(x) for x in mis] == [0.020, 0.021, 0.110, 0.203, 0.229]
    assert all(b > a for a, b in zip(mis[1:], mis[2:]))                                       # K = 2 < K = 3 < beidseitig
    eff = [m(V.rand_cell(meas, G.GAME_COORD, 0.0, tk, k, 6.0), "eff") for tk, k in TALKS]
    assert [r(x) for x in eff] == [0.995, 0.995, 0.945, 0.906, 0.870]
    one = V.rand_cell(meas, G.GAME_HIRSCH, 0.0, G.TALK_ONE, 1, 6.0)
    none = V.rand_cell(meas, G.GAME_HIRSCH, 0.0, G.TALK_NONE, 0, 6.0)
    assert V.differs(one, none, "mis") == 0                                                   # Kontrolle K = 1
    assert [r(m(V.rand_cell(meas, G.GAME_COORD, 0.0, tk, k, 20.0), "eff")) for tk, k in TALKS] == [1.0, 1.0, 1.0, 1.0, 0.999]


# --- H5: die Altkonvention verlassen ----------------------------------------------------------------------------------------------------------
def test_h5_exit_shares(meas):
    for beta, expected in ((20.0, {0.4: [0.0, 0.0, 0.244, 1.0, 0.999], 0.5: [0.0, 0.0, 0.001, 0.406, 0.813], 0.6: [0.0, 0.0, 0.0, 0.01, 0.24]}),
                           (40.0, {0.4: [0.0, 0.0, 0.005, 0.399, 0.779], 0.5: [0.0, 0.0, 0.0, 0.034, 0.544], 0.6: [0.0, 0.0, 0.0, 0.0, 0.326]})):
        for p, vals in expected.items():
            assert [r(m(e3(meas, p, tk, k, beta), "aa")) for tk, k in TALKS] == vals, (beta, p)


def test_h5_counts(meas):
    higher = {beta: sum(1 for tk, k in TALKS[1:] for p in C.THRESHOLD_PSTARS if V.differs(e3(meas, p, tk, k, beta), e3(meas, p, G.TALK_NONE, 0, beta), "aa") == 1) for beta in C.E3_BETAS}
    assert higher == {20.0: 6, 40.0: 5}
    assert all(m(e3(meas, p, G.TALK_NONE, 0, beta), "aa") == 0.0 and m(e3(meas, p, G.TALK_ONE, 1, beta), "aa") == 0.0 for p in C.THRESHOLD_PSTARS for beta in C.E3_BETAS)       # Kontrolle K = 1
    cmp = {1: 0, -1: 0, 0: 0}
    for tk, k in TALKS[1:]:
        for p in C.THRESHOLD_PSTARS:
            cmp[V.differs(e3(meas, p, tk, k, 40.0), e3(meas, p, tk, k, 20.0), "aa")] += 1
    assert cmp == {-1: 5, 1: 1, 0: 6}
    text = V.verdict_e3(meas, 20.0)[1]
    assert "Zeile spricht, K = 2: 0,4" in text and "Zeile spricht, K = 3: 0,4, 0,5" in text and "beide sprechen, K = 2: 0,4, 0,5, 0,6" in text and "steigt in 1 (beide sprechen, K = 2 bei p* = 0,6)" in text


def test_h5_the_exit_runs_are_whole_runs_not_averages(meas):
    for p, tk, k in ((0.4, G.TALK_ONE, 2), (0.5, G.TALK_ONE, 3), (0.4, G.TALK_TWO, 2)):
        c = e3(meas, p, tk, k, 20.0)
        assert abs(m(c, "exit") - m(c, "aa")) < 0.01                                          # Läufe enden ganz drin oder ganz draußen


# --- H6: der Sprecher gewinnt -----------------------------------------------------------------------------------------------------------------
def test_h6_the_speaker_wins_and_the_sum_stays_three(meas):
    def pay(start, tk, k):
        c = XP.find_cell(meas, family="bos", talk=tk, K=k, start=start)
        return r(m(c, "p1")), r(m(c, "p2"))
    assert [pay(C.START_RANDOM, tk, k) for tk, k in TALKS] == [(1.545, 1.455), (1.506, 1.494), (1.898, 1.101), (1.999, 1.0), (1.491, 1.49)]
    assert [pay(C.START_OLD, tk, k) for tk, k in TALKS] == [(1.0, 2.0), (1.0, 2.0), (1.245, 1.753), (1.999, 1.0), (1.054, 1.938)]
    sums = [m(XP.find_cell(meas, family="bos", talk=tk, K=k, start=s), "p1") + m(XP.find_cell(meas, family="bos", talk=tk, K=k, start=s), "p2") for tk, k in TALKS for s in C.STARTS]
    assert len(sums) == 10 and max(abs(x - 3.0) for x in sums) < 0.02 and r(max(abs(x - 3.0) for x in sums)) == 0.018
    none = XP.find_cell(meas, family="bos", talk=G.TALK_NONE, K=0, start=C.START_RANDOM)
    one = XP.find_cell(meas, family="bos", talk=G.TALK_ONE, K=1, start=C.START_RANDOM)
    assert V.differs(one, none, "p1") == 0 and "höher als ohne Reden in 4 von 4 Zellen" in V.verdict_e4(meas)[1]


# --- H8 und H9: Lewis -------------------------------------------------------------------------------------------------------------------------
def test_h8_signalling_systems_share_falls_with_n_and_extra_signals_heal(meas):
    def share(n, ns, cp):
        return r(m(XP.find_lewis(meas, n, ns)[str(cp)], "share"))
    assert [share(n, n, 100000) for n in (2, 3, 4, 6)] == [0.983, 0.876, 0.721, 0.499]
    assert [share(n, n, 10000) for n in (2, 3, 4, 6)] == [0.955, 0.806, 0.581, 0.295]
    assert [share(3, ns, 100000) for ns in (3, 4, 6)] == [0.876, 0.984, 1.0] and share(4, 8, 100000) == 0.999 and share(2, 3, 100000) == 1.0
    assert [(share(n, ns, 100000), r(XP.find_lewis(meas, n, ns)["best"])) for n, ns in ((3, 2), (4, 2), (4, 3))] == [(1.0, 0.667), (1.0, 0.5), (0.981, 0.75)]
    eff = [r(m(XP.find_lewis(meas, n, n)["100000"], "eff")) for n in (2, 3, 4, 6)]
    assert eff == [0.995, 0.963, 0.934, 0.913]


def test_h9_cross_play_against_one_over_n(meas):
    def cross(n, ns):
        c = XP.find_lewis(meas, n, ns)["100000"]["cross"]
        return r(c["mean"]), r(c["se"])
    assert [cross(n, n)[0] for n in (2, 3, 4, 6)] == [0.498, 0.341, 0.238, 0.187]
    n6 = XP.find_lewis(meas, 6, 6)["100000"]["cross"]
    assert r((n6["mean"] - 1 / 6) / n6["se"], 1) == 2.1
    inside = sum(1 for n, ns in C.LEWIS_PAIRS if XP.find_lewis(meas, n, ns)["100000"]["cross"]["mean"] is not None
                 and abs(XP.find_lewis(meas, n, ns)["100000"]["cross"]["mean"] - 1 / n) <= 2 * XP.find_lewis(meas, n, ns)["100000"]["cross"]["se"] + 0.005)
    assert inside == 10 and "in 10 von 11 Zellen" in V.verdict_e6(meas)[1] and "n = 4, ns = 3: 0,232 gegen 0,250" in V.verdict_e6(meas)[1]
    assert XP.find_lewis(meas, 4, 4)["1000"]["cross"]["mean"] is None                         # nach 1.000 Runden hat kein Paar beide am Bestwert


def test_lewis_equilibrium_counts_and_signalling_systems():
    assert [(x["Zustände n = Signale"], x["reine Gleichgewichte"], x["Signalsysteme (Effizienz 1)"], x["n!"]) for x in V.lewis_ne_table()] == [(2, "6", "2", "2"), (3, "87", "6", "6"), (4, "2.056", "24", "24")]
    assert math.factorial(2) == 2 and LW.pure_ne_lewis(3)["by_efficiency"].keys() >= {1.0}


# --- H10: der Mechanismus ---------------------------------------------------------------------------------------------------------------------
def test_h10_snapshots_contradict_the_neutral_drift_guess(meas):
    cells = V.mech_cells(meas)
    assert len(cells) == 12
    smax = [m(c, "smax_pre") for _, _, _, c in cells]
    assert (r(min(smax)), r(max(smax))) == (0.0, 0.107) and max(smax) < 0.2
    above = [m(c, "above_pre") for _, _, _, c in cells]
    assert sum(1 for a in above if a == 0.0) == 8 and r(max(above), 3) == 0.003
    nmin = [m(c, "nmin_pre") for _, _, _, c in cells]
    assert (r(min(nmin), 1), r(max(nmin), 1)) == (2.2, 8.1)


def test_h10_event_study(mech):
    cells = mech["cells"]
    assert [(c["pstar"], c["K"], c["beta"], c["exits"], c["runs"]) for c in cells] == [(0.4, 2, 20.0, 191, 800), (0.4, 3, 20.0, 800, 800), (0.5, 3, 20.0, 358, 800), (0.4, 3, 40.0, 319, 800)]
    sums = [M.summary(c) for c in cells]
    assert [r(c["mean_n_before"], 1) for c in cells] == [10.0, 6.7, 6.7, 6.7]
    assert [r(s[100]["n"], 1) for s in sums] == [5.9, 4.0, 3.5, 3.5] and [r(s[50]["n"], 1) for s in sums] == [5.3, 4.3, 2.9, 1.6]
    assert [r(s[200]["s"], 2) for s in sums] == [0.03, 0.07, 0.07, 0.02]
    assert [r(s[50]["s"], 2) for s in sums] == [0.27, 0.29, 0.4, 0.22] and [r(s[20]["s"], 2) for s in sums] == [0.47, 0.5, 0.59, 0.5]
    assert [r(s[20]["a"], 1) for s in sums] == [4.4, 4.1, 4.5, 3.1] and [r(s[0]["a"], 0) for s in sums] == [10.0] * 4
    assert [c["limit"] for c in cells] == [3, 3, 1, 3]
    assert all(s[100]["n"] < c["mean_n_before"] for s, c in zip(sums, cells))                 # die Austrittsnachricht ist vorher seltener als der Durchschnitt
    assert all(s[200]["s"] < s[100]["s"] < s[50]["s"] < s[20]["s"] for s in sums)


def test_h10_formula_limits():
    assert [XP.lone_sender_limit(beta, 20, p) for beta in (20.0, 40.0) for p in (0.4, 0.5, 0.6)] == [3, 1, 0, 3, 1, 1]
    assert r(XP.hearer_share(20, 20, 0.4, 1), 3) == 0.646 and r(XP.hearer_share(20, 20, 0.4, 4), 3) == 0.354 and r(XP.hearer_share(20, 20, 0.4, 0, 0), 3) == 0.5


# --- H11: Abhängigkeit von N ------------------------------------------------------------------------------------------------------------------
def test_h11_dependence_on_n(meas):
    one_sided = {n: [r(m(XP.find_cell(meas, family=("e3" if n == 20 else "e3n"), b=round(9 * p, 6), talk=G.TALK_ONE, K=3, beta=20.0, n_agents=n), "aa")) for p in (0.4, 0.5)] for n in (10, 20, 40)}
    assert one_sided == {10: [0.999, 0.998], 20: [1.0, 0.406], 40: [0.481, 0.0]}
    hs = {beta: [r(m(V.hs_cell(meas, 0.0, G.TALK_TWO, 2, beta, n), "aa")) for n in C.HANDSHAKE_NS] for beta in C.HANDSHAKE_BETAS}
    assert hs == {40.0: [0.388, 0.038, 0.004], 100.0: [0.615, 0.184, 0.065]}
    assert all(a > b > c for a, b, c in hs.values())


def test_handshake_with_and_without_the_lure(meas):
    cells = {(t, beta): V.hs_cell(meas, t, G.TALK_TWO, 2, beta) for t in (0.0, 1.0) for beta in C.HANDSHAKE_BETAS}
    assert {k: r(m(c, "aa")) for k, c in cells.items()} == {(0.0, 40.0): 0.038, (0.0, 100.0): 0.184, (1.0, 40.0): 0.0, (1.0, 100.0): 0.025}
    ctrl = [V.hs_cell(meas, t, G.TALK_ONE, 1, beta) for t in (0.0, 1.0) for beta in C.HANDSHAKE_BETAS]
    assert len(ctrl) == 4 and all(m(c, "aa") == 0.0 for c in ctrl)
    assert abs(m(cells[(0.0, 100.0)], "hs") - m(cells[(0.0, 100.0)], "aa")) < 0.01            # der Austritt läuft über Akteure mit Handschlag


def test_handshake_payoffs_by_hand(handshake_hand):
    """Mutant gegen Bewohner 7, gegen seinesgleichen 9; die Zahlen der Hand-Instanz stimmen mit dem erweiterten Spiel überein."""
    ext = G.build(G.GAME_HIRSCH, 7.0, 0.0, G.TALK_TWO, 2)
    res = ext.s1.index((0, (1, 1, 1, 1)))
    mut = ext.s1.index((1, (1, 1, 1, 0)))
    assert (ext.U1[mut, mut], ext.U1[mut, res], ext.U1[res, mut], ext.U1[res, res]) == (handshake_hand["mutant_vs_mutant"], handshake_hand["mutant_vs_resident"],
                                                                                       handshake_hand["resident_vs_mutant"], handshake_hand["resident_vs_resident"])
    z = 0.1
    assert z * ext.U1[mut, mut] + (1 - z) * ext.U1[mut, res] == pytest.approx(handshake_hand["mutant_at_01"])
    assert z * ext.U1[res, mut] + (1 - z) * ext.U1[res, res] == pytest.approx(handshake_hand["resident_at_01"])


# --- Hilfetexte der Presets --------------------------------------------------------------------------------------------------------------------
def test_preset_help_numbers(meas, mech):
    h = C.PRESET_HELP
    std = h["Standard (Hirschjagd, Zeile spricht)"]
    assert all(x in std for x in ("7 (Schwelle p* = 0,778)", "(9; 9) und (7; 7)", "0,778", "99,8 %", "4 Blöcke zu je 200 Läufen", "40.000 Schritte"))
    assert r(m(V.rand_cell(meas, G.GAME_HIRSCH, 0.0, G.TALK_ONE, 2, 20.0), "bb"), 3) == 0.998 and r(G.threshold(7.0), 3) == 0.778
    au = h["Beispiel in der Art von Aumann"]
    assert "8" in au and "kein informatives Gleichgewicht" in au and "kein Zitat" in au
    assert V.e2_rows(7.0, (G.TALK_ONE, 2))[3]["informative"] is False and C.PRESETS["Beispiel in der Art von Aumann"]["t"] == 1.0
    bos = h["Kampf der Geschlechter: wer spricht?"]
    assert all(x in bos for x in ("1,545", "1,999", "1,455", "1,000", "Summe bleibt 3"))
    ab = h["Altkonvention verlassen"]
    assert all(x in ab for x in ("p* = 0,4", "100,0 %", "24,4 %", "0,0 %", "N = 20"))
    assert (r(m(e3(meas, 0.4, G.TALK_ONE, 3, 20.0), "aa"), 3), r(m(e3(meas, 0.4, G.TALK_ONE, 2, 20.0), "aa"), 3)) == (1.0, 0.244)
    hs = h["Geheimer Handschlag"]
    assert all(x in hs for x in ("18,4 %", "3,8 %", "61,5 %", "6,5 %", "p* = 0,778", "200.000 Schritte"))
    lw = h["Lewis: drei Zustände"]
    assert all(x in lw for x in ("87,6 %", "0,341", "0,333", "100.000 Runden"))
    assert r(m(XP.find_lewis(meas, 3, 3)["100000"], "share"), 3) == 0.876 and r(1 / 3, 3) == 0.333
    assert C.PRESETS["Altkonvention verlassen"]["b"] == 3.6 and C.PRESETS["Altkonvention verlassen"]["K"] == 3 and C.PRESETS["Geheimer Handschlag"]["steps"] == 200_000 == C.HANDSHAKE_STEPS_PER_AGENT * 20


# --- README und App: dieselben Texte, dieselben Tabellenzeilen ---------------------------------------------------------------------------------
def test_assumption_table_is_the_same_text_in_app_and_readme():
    assert T.ASSUMPTIONS_MARKDOWN.strip() in README


def _md_row(d):
    return "| " + " | ".join(str(v) for v in d.values()) + " |"


def test_readme_tables_are_built_by_the_same_row_builders(meas, e1):
    for row in V.e3_table(meas, 20.0) + V.e3_table(meas, 40.0):
        assert _md_row(row) in README, row
    for row in V.e6_table(meas, 100000):
        assert _md_row(row) in README, row
    for row in V.e4_table(meas, C.START_RANDOM):
        assert _md_row(row) in README, row
    for row in V.hs_table(meas):
        assert _md_row(row) in README, row
    for row in V.lewis_ne_table():
        assert _md_row(row) in README, row
    for row in V.e2_table(V.e2_rows(7.0, (G.TALK_ONE, 2))):
        assert _md_row(row) in README, row


def test_every_number_with_a_unit_in_the_readme_headline_exists_only_in_hypothesis_rows():
    """Die Hypothesentabelle hat genau H1 bis H11 und jede Zeile genau ein Ja, Nein oder Offen am Anfang der Antwort."""
    rows = re.findall(r"^\| (H\d+) \|[^|]*\|\s*\*\*([^*]+)\*\*", README, flags=re.M)
    assert [h for h, _ in rows] == [f"H{i}" for i in range(1, 12)]
    assert all(a.split()[0].rstrip(",") in ("Ja", "Nein", "Offen") for _, a in rows)


def test_the_lottery_check_covers_64_pure_deviations():
    ext = G.build(G.GAME_BOS, 7.0, 0.0, G.TALK_TWO, 2)
    assert (ext.S1, ext.S2, ext.S1 + ext.S2) == (32, 32, 64)


def test_the_budget_cuts_the_handshake_run_to_37_runs():
    import vor_budget as BU
    import vor_scenario as S
    assert BU.used_runs(S.Settings(**C.PRESETS["Geheimer Handschlag"] | {"runs": 200})) == 37
    assert BU.used_runs(S.Settings(**C.PRESETS["Geheimer Handschlag"])) == 37 and C.PRESETS["Geheimer Handschlag"]["runs"] == 40 and C.MAX_SECONDS == 25.0
