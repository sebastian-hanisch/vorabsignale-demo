"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): kleinste Zahlen, Reglerränder, Entartungen, Gleichstände, Division durch null."""

import numpy as np
import pytest

import vor_budget as BU
import vor_constants as C
import vor_equilibria as E
import vor_evaluation as V
import vor_experiments as XP
import vor_games as G
import vor_learning as L
import vor_lewis as LW
import vor_scenario as S


def test_one_message_is_the_control_and_has_the_same_payoffs_as_no_talk():
    """K = 1: einseitig 2 Zeilenstrategien (eine Nachricht, zwei Aktionen) und 2 Regeln, beidseitig 2 Strategien je Seite; die Auszahlungsmatrix ist die des Spiels ohne Reden."""
    for game in G.GAMES:
        none = G.build(game, 7.0, 0.0, G.TALK_NONE, 0)
        one = G.build(game, 7.0, 0.0, G.TALK_ONE, 1)
        two = G.build(game, 7.0, 0.0, G.TALK_TWO, 1)
        assert (one.S1, one.S2, two.S1, two.S2) == (2, 2, 2, 2)
        assert np.array_equal(none.U1, one.U1) and np.array_equal(none.U2, one.U2)
        assert np.array_equal(none.U1, two.U1) and np.array_equal(none.U2, two.U2)


def test_zero_messages_for_talking_is_rejected():
    with pytest.raises(ValueError):
        G.extended(*G.payoffs(G.GAME_BOS), "one", 0)
    with pytest.raises(ValueError):
        G.extended(*G.payoffs(G.GAME_BOS), "two", 0)
    with pytest.raises(ValueError):
        G.payoffs("unbekannt")


def test_a_run_with_zero_steps_returns_the_start_and_tiny_populations_work():
    ext = G.build(G.GAME_HIRSCH, 7.0, 0.0, G.TALK_ONE, 2)
    snap = L.population_run(ext, 10.0, 5, 0, 3, 1, L.START_OLD)[-1]
    assert snap.step == 0 and snap.bb == pytest.approx(1.0) and snap.aa == pytest.approx(0.0)
    for n_agents in (1, 2):
        s = L.population_run(ext, 10.0, n_agents, 50, 2, 1, L.START_RANDOM)[-1]
        assert np.isfinite(s.efficiency).all() and (s.aa + s.bb + s.mis == pytest.approx(1.0))


def test_checkpoints_may_include_the_start_and_must_stay_inside_the_run():
    ext = G.build(G.GAME_COORD, 7.0, 0.0, G.TALK_NONE, 0)
    snaps = L.population_run(ext, 5.0, 4, 30, 2, 3, L.START_RANDOM, (0, 10, 30, 10))
    assert [s.step for s in snaps] == [0, 10, 30]
    for bad in ((31,), (-1,)):
        with pytest.raises(ValueError):
            L.population_run(ext, 5.0, 4, 30, 2, 3, L.START_RANDOM, bad)
    with pytest.raises(ValueError):
        L.population_run(ext, 5.0, 4, 30, 0, 3, L.START_RANDOM)


def test_ties_in_pure_equilibria_count_and_the_tolerance_is_inclusive():
    """Gleiche Auszahlungen: 3 × 3 mit lauter Gleichständen hat 9 Gleichgewichte (Gleichheit zählt); ein Unterschied unter der Toleranz ändert das nicht."""
    flat = np.ones((3, 3))
    assert len(E.pure_ne(flat, flat)) == 9
    assert len(E.pure_ne(flat, flat - 1e-13)) == 9


def test_hare_payoff_and_lure_at_the_slider_edges_keep_a_stag_hunt():
    """Am Rand b = 7,5 und t = 1 ist Hase gegen Hirsch 8,5 < 9: Hirsch bleibt gegen Hirsch besser; am anderen Rand b = 1 liegt die Schwelle bei 1/9."""
    assert C.B_MAX + C.T_MAX < G.HIRSCH_PAYOFF
    p1, _ = G.payoffs(G.GAME_HIRSCH, C.B_MAX, C.T_MAX)
    assert p1[0, 0] > p1[1, 0] and p1[1, 1] > p1[0, 1]
    assert G.threshold(C.B_MIN) == pytest.approx(1 / 9) and G.threshold(7.0, 1.0) == pytest.approx(7 / 8)
    for b, t in ((C.B_MIN, C.T_MIN), (C.B_MAX, C.T_MAX)):
        row = XP.equilibrium_row(G.GAME_HIRSCH, b, t, G.TALK_ONE, 2)
        assert row["pure"] == 6 and not row["miscoordination"]


def test_budget_never_returns_zero_runs_for_a_learnable_game_and_reports_unlearnable_ones():
    for talk, K in ((G.TALK_NONE, 0), (G.TALK_ONE, 3), (G.TALK_TWO, 2)):
        s = S.Settings(talk=talk, K=K, steps=C.STEPS_MAX, runs=C.RUNS_MAX)
        assert BU.learnable(s) and 1 <= BU.used_runs(s) <= C.RUNS_MAX
    s3 = S.Settings(talk=G.TALK_TWO, K=3)
    assert not BU.learnable(s3) and BU.used_runs(s3) == 0
    view = V.analyse(s3)
    assert view.runs_used == 0 and view.snaps == []


def test_effective_parameters_ignore_hare_and_lure_outside_the_stag_hunt():
    s = S.Settings(game=G.GAME_BOS, b=3.0, t=1.0)
    assert S.effective_t(s) == 0.0 and S.effective_b(s) == C.DEFAULT_B and S.threshold(s) is None
    assert np.array_equal(S.ext_of(s).U1, S.ext_of(S.with_(s, b=5.0, t=0.0)).U1)


def test_lewis_with_one_signal_and_two_states_cannot_exceed_one_half():
    """n = 2, ns = 1: der Hörer kann nur eine Handlung wählen, die Effizienz ist höchstens 1/2 (Bestwert min(1, 1/2))."""
    assert LW.best_efficiency(2, 1) == 0.5
    snap = LW.roth_erev(2, 1, 400, 20, 5)[-1]
    assert (snap.efficiency <= 0.5 + 1e-12).all()
    assert LW.roth_erev(3, 3, 0, 2, 1)[0].round == 0


def test_lewis_rejects_degenerate_arguments():
    for args in ((1, 2, 10, 2, 0), (2, 0, 10, 2, 0), (2, 2, -1, 2, 0), (2, 2, 10, 0, 0)):
        with pytest.raises(ValueError):
            LW.roth_erev(*args)


def test_experiment_statistics_handle_a_single_block_and_missing_values():
    one = XP.stats([{"x": 0.5}])
    assert one["x"] == {"mean": 0.5, "se": 0.0}
    two = XP.stats([{"x": 1.0, "y": None}, {"x": 3.0, "y": None}])
    assert two["x"]["mean"] == 2.0 and two["x"]["se"] == pytest.approx(1.0) and two["y"] == {"mean": None, "se": None}
    seq = XP.stats([{"v": [1.0, None]}, {"v": [3.0, None]}])
    assert seq["v"][0]["mean"] == 2.0 and seq["v"][1]["mean"] is None


def test_hearer_share_formula_has_the_right_limits():
    """Eine Nachricht, die kein Sprecher sendet (a = n = 0): Anteil 1/2. Viele Absender ohne A-Sprecher: Anteil nahe 0. Ein einzelner A-Sprecher allein: 1 / (1 + exp(−β (1 − p*) / N))."""
    assert XP.hearer_share(20, 20, 0.4, 0, 0) == pytest.approx(0.5)
    assert XP.hearer_share(20, 20, 0.4, 20, 0) < 1e-3
    assert XP.hearer_share(20, 20, 0.4, 1, 1) == pytest.approx(1 / (1 + np.exp(-20 * 0.6 / 20)))
