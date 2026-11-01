"""Zweig-Tests (Nullspalten-Regel, DEMO-PLAYBOOK §0): jede Variante weicht auf mindestens einer Instanz von der Nachbarvariante ab, und keine Spalte der Messreihen ist ein toter Zweig.

* K = 1 verhält sich wie kein Reden (hier sogar Zahl für Zahl, weil die Strategienmengen gleich aufgebaut sind);
* jede Art des Redens (einseitig K = 2 und 3, beidseitig K = 2) ändert mindestens eine Kennzahl gegenüber dem Spiel ohne Reden;
* der Handschlag-Zweig tritt bei β = 100 auf (Anteil größer als null), die Kontrolle ohne Reden nie;
* jede Verlockung t > 0 verändert die Existenz des informativen Gleichgewichts."""

import numpy as np
import pytest

import vor_constants as C
import vor_evaluation as V
import vor_experiments as XP
import vor_games as G
import vor_learning as L
import vor_scenario as S


@pytest.fixture(scope="module")
def meas():
    m = XP.load_measurements()
    assert m is not None, "data/measurements.json fehlt oder gehört zu anderem Code: python tools/make_measurements.py"
    return m


@pytest.mark.parametrize("game,b,t", [(G.GAME_COORD, 7.0, 0.0), (G.GAME_HIRSCH, 7.0, 0.0), (G.GAME_BOS, 7.0, 0.0)])
@pytest.mark.parametrize("start", [L.START_RANDOM, L.START_OLD])
def test_one_message_gives_exactly_the_same_run_as_no_talk(game, b, t, start):
    """K = 1: dieselben Auszahlungen, dieselbe Zahl und Reihenfolge der Strategien, derselbe Zufallsstrom: alle Kennzahlen stimmen Zahl für Zahl überein."""
    none = G.build(game, b, t, G.TALK_NONE, 0)
    ref = L.population_run(none, 15.0, 8, 400, 6, 77, start, (0, 200, 400))
    for talk in (G.TALK_ONE, G.TALK_TWO):
        ext = G.build(game, b, t, talk, 1)
        got = L.population_run(ext, 15.0, 8, 400, 6, 77, start, (0, 200, 400))
        for a, c in zip(ref, got):
            for field in ("efficiency", "payoff1", "payoff2", "aa", "bb", "mis"):
                assert np.array_equal(getattr(a, field), getattr(c, field)), (talk, field)
            assert np.allclose(a.info, c.info, atol=1e-12) and np.allclose(c.info, 0.0, atol=1e-12)                # eine Nachricht verrät nichts


def test_every_kind_of_talk_changes_at_least_one_measured_figure_against_no_talk(meas):
    """Kampf der Geschlechter, zufälliger Start, β = 20: Auszahlung der Zeile, Fehlkoordination oder Information weichen um mehr als 4 Standardfehler (und die feste Marge) ab."""
    none = XP.find_cell(meas, family="bos", talk=G.TALK_NONE, start=C.START_RANDOM)
    changed = {}
    for talk, K in C.E3_TALKS[1:]:
        cell = XP.find_cell(meas, family="bos", talk=talk, K=K, start=C.START_RANDOM)
        changed[(talk, K)] = [key for key in ("p1", "p2", "mis", "info") if V.differs(cell, none, key) != 0]
    assert changed[(G.TALK_ONE, 1)] == []                                         # die Kontrolle ändert nichts
    for key in ((G.TALK_ONE, 2), (G.TALK_ONE, 3), (G.TALK_TWO, 2)):
        assert changed[key], key


def test_the_handshake_branch_occurs_at_beta_100_and_never_without_talk(meas):
    two = XP.find_cell(meas, family="hs", talk=G.TALK_TWO, K=2, t=0.0, beta=100.0)
    control = XP.find_cell(meas, family="hs", talk=G.TALK_ONE, K=1, t=0.0, beta=100.0)
    assert two["hs"]["mean"] > 0.0 and two["aa"]["mean"] > 0.0
    assert control["hs"]["mean"] == 0.0 and control["aa"]["mean"] == 0.0


def test_every_lure_changes_the_existence_of_the_informative_equilibrium():
    """Hirschjagd mit b = 7, einseitiges Reden mit K = 2: bei t = 0 gibt es eines, bei jedem t aus 0,25 / 0,5 / 1 keines."""
    flags = {t: XP.equilibrium_row(G.GAME_HIRSCH, 7.0, t, G.TALK_ONE, 2)["informative"] for t in C.E2_TS}
    assert flags == {0.0: True, 0.25: False, 0.5: False, 1.0: False}
    assert XP.equilibrium_row(G.GAME_BOS, 7.0, 1.0, G.TALK_ONE, 2)["informative"] is False        # die Verlockung wirkt nur in der Hirschjagd


def test_hirsch_and_coordination_and_bos_differ_in_informative_equilibria():
    got = {game: XP.equilibrium_row(game, 7.0, 0.0, G.TALK_ONE, 2)["informative"] for game in G.GAMES}
    assert got == {G.GAME_COORD: True, G.GAME_HIRSCH: True, G.GAME_BOS: False}


def test_the_mechanism_columns_are_filled_for_one_sided_cells_and_empty_for_no_talk(meas):
    one = XP.find_cell(meas, family="e3", talk=G.TALK_ONE, K=2, b=3.6, beta=20.0)
    none = XP.find_cell(meas, family="e3", talk=G.TALK_NONE, K=0, b=3.6, beta=20.0)
    assert one["pre_obs"]["mean"] > 0 and one["smax_pre"]["mean"] is not None
    assert "smax_pre" not in none and none["aa"]["mean"] == 0.0 and none["exit"]["mean"] == 0.0


def test_settings_of_a_run_with_the_old_convention_and_no_talk_stay_in_the_old_convention():
    """Ohne Reden bleibt die Altkonvention bei p* = 0,4 und β = 20 stehen (Hirsch/Hirsch nicht erreichbar): Kontrolle des Zweigs „Start = Altkonvention“."""
    ext = S.ext_of(S.Settings(b=3.6, talk=G.TALK_NONE))
    snap = L.population_run(ext, 20.0, 20, 4000, 20, 5, L.START_OLD)[-1]
    assert snap.aa.max() == 0.0 and snap.bb.min() > 0.9
