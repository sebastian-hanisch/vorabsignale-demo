"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import random
from collections.abc import Callable
from dataclasses import dataclass

import streamlit as st

import vor_constants as C
import vor_games as G


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None      # bei Auswahlreglern: die erlaubten Werte


B = C.BASE
SETTING_SPECS = {
    "part_select": SettingSpec("teil", str, B["part"], choices=C.PARTS),
    "game_select": SettingSpec("spiel", str, B["game"], choices=G.GAMES),
    "b_slider": SettingSpec("b", float, B["b"], C.B_MIN, C.B_MAX),
    "t_slider": SettingSpec("t", float, B["t"], C.T_MIN, C.T_MAX),
    "talk_select": SettingSpec("reden", str, B["talk"], choices=G.TALKS),
    "k_slider": SettingSpec("k", int, B["K"], C.K_MIN, C.K_MAX),
    "beta_slider": SettingSpec("beta", float, B["beta"], C.BETA_MIN, C.BETA_MAX),
    "n_slider": SettingSpec("n", int, B["n_agents"], C.N_MIN, C.N_MAX),
    "start_select": SettingSpec("start", str, B["start"], choices=C.STARTS),
    "steps_slider": SettingSpec("schritte", int, B["steps"], C.STEPS_MIN, C.STEPS_MAX),
    "runs_slider": SettingSpec("laeufe", int, B["runs"], C.RUNS_MIN, C.RUNS_MAX),
    "lewis_n_slider": SettingSpec("zustaende", int, B["lewis_n"], C.LEWIS_N_MIN, C.LEWIS_N_MAX),
    "lewis_ns_slider": SettingSpec("signale", int, B["lewis_ns"], C.LEWIS_NS_MIN, C.LEWIS_NS_MAX),
    "lewis_rounds_slider": SettingSpec("runden", int, B["lewis_rounds"], C.LEWIS_ROUNDS_MIN, C.LEWIS_ROUNDS_MAX),
    "seed_input": SettingSpec("seed", int, B["seed"], 0, C.SEED_MAX),
}
PRESET_KEYS = {"part": "part_select", "game": "game_select", "b": "b_slider", "t": "t_slider", "talk": "talk_select", "K": "k_slider", "beta": "beta_slider", "n_agents": "n_slider",
               "start": "start_select", "steps": "steps_slider", "runs": "runs_slider", "lewis_n": "lewis_n_slider", "lewis_ns": "lewis_ns_slider", "lewis_rounds": "lewis_rounds_slider",
               "seed": "seed_input"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf erlaubte Werte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if spec.choices is not None:
                if value not in spec.choices:
                    continue
            else:
                if value != value:      # NaN
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
            st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS,
    dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
