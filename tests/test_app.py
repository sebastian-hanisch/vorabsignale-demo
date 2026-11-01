"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Knopf, fährt jeden Regler an seine Grenzen und prüft, dass
kein Python-Fehler auftritt - insbesondere `StreamlitDuplicateElementId` (mehrere st.plotly_chart-Aufrufe ohne eindeutiges key=) und `StreamlitAPIException` bei Reglern mit
berechneten Grenzen. Die ▶️-Knöpfe laufen einzeln (die Schleife lässt sie aus)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import vor_constants as C
import vor_games as G

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer
AUTOPLAY_MARKER = "▶"  # ▶: Knöpfe mit Rechenlast, einzeln getestet
PRESET_LABELS = set(C.PRESETS)


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join([w.value for w in at.warning] + [s.value for s in at.success] + [i.value for i in at.info] + [e.value for e in at.error])


def _tables(at, column):
    return [df for df in at.dataframe if column in df.value.columns.tolist()]


def test_app_loads_without_exception_and_the_measurement_file_fits_the_code():
    at = _fresh_app()
    assert "Nach 40.000 Schritten" in _texts(at)                                       # Standard: Hirschjagd, Zeile spricht, K = 2
    assert not [w for w in at.warning if "vorgerechneten Messreihen" in w.value]


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label or label in PRESET_LABELS:
            continue                                                 # Presets und Rechenknöpfe laufen einzeln
        at = _fresh_app()
        at.button[index].click().run()
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            if slider.disabled:
                continue
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run()
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("game", G.GAMES)
@pytest.mark.parametrize("talk", G.TALKS)
@pytest.mark.parametrize("edge", ["min", "max"])
def test_corner_matrix_all_sidebar_sliders_at_their_minima_or_maxima(game, talk, edge):
    """Eckfall-Matrix (DEMO-PLAYBOOK §1d): je Spiel und Art des Redens alle Regler zugleich am Minimum bzw. Maximum (b 1 bzw. 7,5, t 0 bzw. 1, K 1 bzw. 3, β 2 bzw. 100, N 10 bzw. 40,
    Schritte 2 000 bzw. 200 000, Läufe 4 bzw. 200). Das Maximum rechnet mit dem Budget (die Läufe werden gekürzt) und beidseitig mit K = 3 gar nicht (nur Gleichgewichte gezählt)."""
    at = _fresh_app()
    at.selectbox(key="game_select").set_value(game)
    at.selectbox(key="talk_select").set_value(talk)
    at.run()
    for slider in at.sidebar.slider:
        if slider.disabled:
            continue
        slider.set_value(slider.min if edge == "min" else slider.max)
    at.run()
    assert not at.exception, (game, talk, edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("edge", ["min", "max"])
def test_corner_matrix_for_the_lewis_part(edge):
    at = _fresh_app()
    at.selectbox(key="part_select").set_value(C.PART_LEWIS).run()
    for slider in at.sidebar.slider:
        if slider.disabled:
            continue
        slider.set_value(slider.min if edge == "min" else slider.max)
    at.run()
    assert not at.exception, (edge, [str(e) for e in at.exception])
    assert any("Runden (n =" in i.value for i in at.info)


def test_two_sided_talk_with_three_messages_is_counted_not_learned():
    at = _fresh_app()
    at.selectbox(key="talk_select").set_value(G.TALK_TWO)
    at.slider(key="k_slider").set_value(3)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert any("nicht gelernt" in i.value for i in at.info)
    assert not [m for m in at.metric if m.label == "(A, A)"]


def test_the_largest_run_is_cut_to_the_budget_and_says_so():
    at = _fresh_app()
    at.selectbox(key="talk_select").set_value(G.TALK_TWO)
    at.slider(key="steps_slider").set_value(C.STEPS_MAX)
    at.slider(key="runs_slider").set_value(C.RUNS_MAX)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert any(i.value.startswith("Gerechnet mit ") and "statt 200 Läufen" in i.value for i in at.info)


def test_the_default_run_is_not_cut():
    at = _fresh_app()
    assert not [i for i in at.info if i.value.startswith("Gerechnet mit")]


def test_sliders_that_do_not_apply_are_disabled_instead_of_dead():
    """b und t wirken nur in der Hirschjagd, K nur mit Reden, die Lewis-Regler nur im Lewis-Teil, die Lernregler nur im Redeteil."""
    at = _fresh_app()
    assert not at.slider(key="b_slider").disabled and at.slider(key="lewis_n_slider").disabled
    at.selectbox(key="game_select").set_value(G.GAME_BOS).run()
    assert at.slider(key="b_slider").disabled and at.slider(key="t_slider").disabled
    at.selectbox(key="talk_select").set_value(G.TALK_NONE).run()
    assert at.slider(key="k_slider").disabled
    at.selectbox(key="part_select").set_value(C.PART_LEWIS).run()
    assert not at.slider(key="lewis_n_slider").disabled and at.slider(key="beta_slider").disabled and not at.slider(key="runs_slider").disabled


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_loads_and_shows_a_verdict(name):
    at = _fresh_app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    assert _texts(at)
    preset = C.PRESETS[name]
    assert at.session_state["part_select"] == preset["part"] and at.session_state["talk_select"] == preset["talk"] and at.session_state["b_slider"] == preset["b"]


def test_the_play_buttons_work_individually():
    at = _fresh_app()
    at.button(key="e1_button").click().run()
    assert not at.exception and any(c.value.startswith("Beidseitig mit K = 3: 1.536 Strategien") for c in at.caption)

    at = _fresh_app()
    at.button(key="e2_button").click().run()
    assert not at.exception and len(_tables(at, "Schwelle p* = b / (9 − t)")) == 2

    at = _fresh_app()
    at.slider(key="steps_slider").set_value(4_000)
    at.slider(key="runs_slider").set_value(8)
    at.run()
    at.button(key="e3_button").click().run()
    assert not at.exception and len(_tables(at, "Läufe mit mindestens der Hälfte (A, A)")) >= 1 and any("ein Block" in c.value for c in at.caption)

    at = _fresh_app()
    at.slider(key="steps_slider").set_value(4_000)
    at.slider(key="runs_slider").set_value(8)
    at.run()
    at.button(key="e4_button").click().run()
    assert not at.exception and len(_tables(at, "Auszahlung Zeile")) >= 2

    at = _fresh_app()
    at.slider(key="steps_slider").set_value(4_000)
    at.slider(key="runs_slider").set_value(8)
    at.run()
    at.button(key="e5_button").click().run()
    assert not at.exception and len(_tables(at, "gelernt: Information (Bit)")) == 1

    at = _fresh_app()
    at.session_state["lewis_rounds_slider"] = 2_000                                     # ausgegraut: nur über den Zustand setzbar
    at.slider(key="runs_slider").set_value(6)
    at.run()
    at.button(key="e6_button").click().run()
    assert not at.exception and len(_tables(at, "Zufallswert 1/n")) >= 2


def test_play_results_are_hidden_when_the_sidebar_settings_change():
    at = _fresh_app()
    at.button(key="e2_button").click().run()
    assert len(_tables(at, "Schwelle p* = b / (9 − t)")) == 2
    at.slider(key="beta_slider").set_value(30.0).run()
    assert len(_tables(at, "Schwelle p* = b / (9 − t)")) == 1
