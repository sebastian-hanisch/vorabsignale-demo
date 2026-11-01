"""Verdikte und Formate an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import numpy as np

import vor_constants as C
import vor_evaluation as V
import vor_games as G
import vor_learning as L
import vor_scenario as S


def row(label, talk, K, pure, pure_payoffs, all_payoffs, informative=None, difference=None, miscoordination=False, lottery=None, game=G.GAME_HIRSCH):
    return {"label": label, "game": game, "talk": talk, "K": K, "strategies": (2, 2), "pure": pure, "pure_payoffs": pure_payoffs, "all_payoffs": all_payoffs, "informative": informative,
            "difference": difference, "miscoordination": miscoordination, "lottery": lottery}


# --- Formate ----------------------------------------------------------------------------------------------------------------------------------
def test_number_formats_use_decimal_comma_thousands_dot_and_the_true_minus():
    assert V.de(1.5, 2) == "1,50" and V.de(-0.25, 2) == "−0,25" and V.de(-0.0001, 2) == "0,00"
    assert V.thousands(1536) == "1.536" and V.thousands(999) == "999" and V.thousands(626688) == "626.688"
    assert V.pct(0.1234, 1) == "12,3 %" and V.pct(1.0, 0) == "100 %"
    assert V.num(9.0) == "9" and V.num(7.875) == "7,875" and V.num(2 / 3) == "0,667" and V.num(0.5, 2) == "0,5" and V.num(0.0, 2) == "0"
    assert V.pairs_text([(7.0, 7.0), (9.0, 9.0)]) == "(9; 9), (7; 7)" and V.pairs_text(None) == "nicht gerechnet"
    assert V.talk_label(G.TALK_NONE, 0) == "kein Reden" and V.talk_label(G.TALK_TWO, 2) == "beide sprechen, K = 2"


def test_differs_uses_four_standard_errors_and_the_floor():
    a, b = {"x": {"mean": 0.50, "se": 0.01}}, {"x": {"mean": 0.40, "se": 0.01}}
    assert V.differs(a, b, "x") == 1 and V.differs(b, a, "x") == -1                                  # 0,10 > max(4 · 0,0141, 0,02) = 0,0566
    zero = {"x": {"mean": 0.50, "se": 0.0}}
    assert V.differs({"x": {"mean": 0.51, "se": 0.0}}, zero, "x") == 0                               # 0,01 liegt unter der Untergrenze 0,02
    assert V.differs({"x": {"mean": 0.53, "se": 0.0}}, zero, "x") == 1 and V.differs({"x": {"mean": 0.47, "se": 0.0}}, zero, "x") == -1


# --- Gleichgewichte ----------------------------------------------------------------------------------------------------------------------------
NONE = row("Hirschjagd", G.TALK_NONE, 0, 2, [(7.0, 7.0), (9.0, 9.0)], [(7.0, 7.0), (9.0, 9.0)])
TALK2 = row("Hirschjagd", G.TALK_ONE, 2, 6, [(7.0, 7.0), (9.0, 9.0)], [(7.0, 7.0), (9.0, 9.0)], informative=True, difference=7 / 9)


def test_verdict_equilibria_compares_with_the_game_without_talk():
    kind, text = V.verdict_equilibria(NONE, TALK2)
    assert kind == "info"
    assert "6 statt 2 reine Gleichgewichte" in text and "dieselben: (9; 9), (7; 7)" in text and "Alle Gleichgewichte (auch gemischte) haben dieselben Auszahlungspaare" in text
    assert "Fehlkoordination in einem reinen Gleichgewicht: nein" in text and "ja (Reaktionsdifferenz 0,778)" in text


def test_verdict_equilibria_names_a_difference_and_the_lottery():
    other = dict(TALK2, pure_payoffs=[(5.0, 5.0)], all_payoffs=[(5.0, 5.0)], miscoordination=True)
    text = V.verdict_equilibria(NONE, other)[1]
    assert "andere: (5; 5)" in text and "Alle Gleichgewichte (auch gemischte) haben andere" in text and "Fehlkoordination in einem reinen Gleichgewicht: ja" in text
    two = row("Hirschjagd", G.TALK_TWO, 2, 320, [(7.0, 7.0), (9.0, 9.0)], None, lottery={"equilibrium": True, "payoff1": 8.0, "payoff2": 8.0})
    text = V.verdict_equilibria(NONE, two)[1]
    assert "320 statt 2" in text and "Lotterie (gleiche Nachricht → A, sonst B): ein Gleichgewicht mit den Auszahlungen (8; 8)" in text and "Alle Gleichgewichte (auch gemischte)" not in text
    assert "nicht gerechnet" in V.informative_text(two)
    assert "ohne Reden hat das Spiel 2 reine Gleichgewichte" in V.verdict_equilibria(NONE, NONE)[1].replace("Ohne", "ohne")


def test_informative_text_covers_every_branch():
    assert V.informative_text(NONE) == "–"
    assert V.informative_text(row("x", G.TALK_ONE, 1, 2, [], [], informative=None)) == "nein (eine Nachricht)"
    assert V.informative_text(TALK2) == "ja (Reaktionsdifferenz 0,778)"
    assert V.informative_text(dict(TALK2, informative=False, difference=0.0)) == "nein"
    assert V.informative_text(row("x", G.TALK_ONE, 3, 15, [], None)) == "nicht gerechnet"


def test_verdict_e1_counts_cells():
    rows = [NONE, TALK2, row("Hirschjagd", G.TALK_ONE, 1, 2, [(7.0, 7.0), (9.0, 9.0)], [(7.0, 7.0), (9.0, 9.0)]),
            row("Kampf", G.TALK_NONE, 0, 2, [(1.0, 2.0)], [(1.0, 2.0), (2 / 3, 2 / 3)]), row("Kampf", G.TALK_ONE, 2, 6, [(1.0, 2.0)], [(1.0, 2.0)], informative=False, difference=0.0, miscoordination=True),
            row("Kampf", G.TALK_TWO, 2, 256, [(1.0, 2.0)], None)]
    kind, text = V.verdict_e1(rows)
    assert kind == "success"
    assert "dieselben wie ohne in 2 von 3 Zellen" in text                                           # Hirschjagd K = 2 und K = 1 gleich, Kampf K = 2 verliert den gemischten Punkt
    assert "die der reinen Gleichgewichte in 4 von 4 Zellen" in text and "mit Fehlkoordination: 1" in text
    assert "gibt es bei: Hirschjagd;" in text and "keines bei: Kampf." in text


def test_verdict_e2_names_lures_with_and_without_an_informative_equilibrium():
    rows = [dict(r, t=t, b=7.0, pstar=7 / (9 - t)) for r, t in ((dict(TALK2), 0.0), (dict(TALK2, informative=False, difference=0.0), 0.25), (dict(TALK2, informative=False, difference=0.0), 1.0))]
    kind, text = V.verdict_e2(rows)
    assert kind == "success" and "in 1 von 3 Verlockungen (t = 0)" in text and "keines bei t = 0,25, 1" in text and "Reaktionsdifferenz der Spalte 0,778" in text and "p* = 0,778" in text


def test_verdict_e5_counts_games_with_the_lottery():
    lot = {"equilibrium": True, "payoff1": 1.5, "payoff2": 1.5}
    rows = [{"label": "Kampf", "lottery": lot, "none_all": [(1.0, 2.0), (2.0, 1.0), (2 / 3, 2 / 3)], "none_pure": [(1.0, 2.0), (2.0, 1.0)]},
            {"label": "Hirsch", "lottery": {"equilibrium": True, "payoff1": 8.0, "payoff2": 8.0}, "none_all": [(7.0, 7.0), (9.0, 9.0)], "none_pure": [(7.0, 7.0), (9.0, 9.0)]},
            {"label": "Kaputt", "lottery": {"equilibrium": False, "payoff1": 0.0, "payoff2": 0.0}, "none_all": [(1.0, 1.0)], "none_pure": [(1.0, 1.0)]}]
    text = V.verdict_e5(rows, None)[1]
    assert "in 2 von 3 Spielen ein Gleichgewicht" in text and "des schlechtesten Gleichgewichts ohne Reden in 2 von 3 Spielen" in text and "des besten in 1 von 3 (Kampf)" in text
    mech = {"lottery": {"beta": 20.0, "n_agents": 20, "steps": 40000, "mean_share": 0.059, "uniform_share": 0.0625, "runs_at_least_half": 0, "runs": 800}}
    assert "5,9 % der Akteure die Lotterie-Regel" in V.verdict_e5(rows, None, mech)[1] and "in 0 von 800 Läufen" in V.verdict_e5(rows, None, mech)[1]


# --- Hauptansicht ------------------------------------------------------------------------------------------------------------------------------
def _view(aa, bb, mis, p1, p2, info, runs=4, steps=100):
    s = S.Settings(steps=steps, runs=runs, start=C.START_OLD)
    snap = L.Snapshot(steps, np.full(runs, 0.9), np.full(runs, p1), np.full(runs, p2), np.array(aa), np.full(runs, bb), np.full(runs, mis), np.full(runs, info), np.zeros((runs, 0)), np.zeros((runs, 0)), np.zeros(runs))
    import vor_evaluation as ev
    return ev.MainView(s, runs, np.array([steps]), [snap])


def test_verdict_main_names_every_figure_and_counts_the_exits():
    view = _view([0.0, 0.5, 0.75, 1.0], 0.2, 0.05, 7.5, 7.25, 0.0123)
    kind, text = V.verdict_main(view)
    assert kind == "info"
    assert "Nach 100 Schritten (Start in der Altkonvention „alle B“, β = 20, N = 20, Mittel über 4 Läufe)" in text and "(A, A) 56,2 %" in text and "(B, B) 20,0 %" in text and "Fehlkoordination 5,0 %" in text
    assert "Auszahlung Zeile 7,500, Spalte 7,250" in text and "Information der Nachricht 0,012 Bit" in text and "In 3 von 4 Läufen sind mindestens 50 % der Paarungen (A, A)" in text


def test_verdict_main_for_an_unlearnable_game_says_so():
    view = V.analyse(S.Settings(talk=G.TALK_TWO, K=3))
    assert "1.536 Strategien je Seite" in V.verdict_main(view)[1] and "nicht gelernt" in V.verdict_main(view)[1]


def test_verdict_main_uses_random_start_wording():
    view = _view([0.0] * 4, 1.0, 0.0, 7.0, 7.0, 0.0)
    view = V.MainView(S.Settings(steps=100, runs=4), 4, view.steps, view.snaps)
    assert "zufälliger Start" in V.verdict_main(view)[1] and "In 0 von 4 Läufen" in V.verdict_main(view)[1]


def test_verdict_lewis_live_names_reached_runs_cross_play_and_the_signal_shortage():
    s = S.Settings(part=C.PART_LEWIS, lewis_n=3, lewis_ns=2, lewis_rounds=5000, runs=4)
    ps = np.zeros((4, 3, 2))
    ps[:, 0, 0] = ps[:, 1, 1] = ps[:, 2, 1] = 1.0
    pr = np.zeros((4, 2, 3))
    pr[:, 0, 0] = 1.0
    pr[:, 1, 1] = 1.0
    from vor_lewis import LewisSnapshot
    view = V.LewisView(s, 4, 2 / 3, np.array([5000]), [LewisSnapshot(5000, ps, pr)])
    kind, text = V.verdict_lewis_live(view)
    assert kind == "info"
    assert "Nach 5.000 Runden (n = 3 Zustände, ns = 2 Signale, 4 Läufe) erreichen 4 von 4 Läufen mindestens 95 % des Bestwerts 0,667" in text and "mittlere Effizienz 0,667" in text
    assert "Fremdspiel: 0,667 aus 4 Paaren" in text and "der Bestwert ist ns/n" in text
    none_reached = V.LewisView(s, 4, 1.0, np.array([5000]), [LewisSnapshot(5000, ps, pr)])
    assert "nicht definiert" in V.verdict_lewis_live(none_reached)[1]
    one_pair = V.LewisView(s, 1, 2 / 3, np.array([5000]), [LewisSnapshot(5000, ps[:1], pr[:1])])
    assert "aus 1 Paar;" in V.verdict_lewis_live(one_pair)[1]


def test_budget_text_cut_is_consistent_with_the_estimate():
    import vor_budget as BU
    s = S.Settings(talk=G.TALK_TWO, K=2, steps=200_000, runs=200)
    used = BU.used_runs(s)
    ext = S.ext_of(s)
    assert BU.estimated_seconds(ext, s.steps, used) <= C.MAX_SECONDS < BU.estimated_seconds(ext, s.steps, used + 1)
