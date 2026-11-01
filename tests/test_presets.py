"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Budget und Auswahllisten."""

import vor_budget as BU
import vor_constants as C
import vor_games as G
import vor_presets as P
import vor_scenario as S


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_choices():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.choices is not None:
                assert preset[key] in spec.choices, (name, key)
            else:
                assert spec.lo <= preset[key] <= spec.hi, (name, key)
        assert preset["b"] + preset["t"] < G.HIRSCH_PAYOFF


def test_default_preset_equals_the_default_settings():
    assert S.Settings(**C.PRESETS[next(iter(C.PRESETS))]) == S.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_every_talking_preset_is_learnable_within_the_run_budget():
    for name, preset in C.PRESETS.items():
        s = S.Settings(**preset)
        if s.part == C.PART_TALK:
            assert BU.learnable(s) and BU.used_runs(s) >= 1, name


def test_the_named_presets_have_the_values_of_their_names():
    altkonvention = S.Settings(**C.PRESETS["Altkonvention verlassen"])
    assert altkonvention.start == C.START_OLD and round(G.threshold(altkonvention.b, altkonvention.t), 6) == 0.4
    aumann = S.Settings(**C.PRESETS["Beispiel in der Art von Aumann"])
    assert (aumann.b, aumann.t, aumann.game) == (7.0, 1.0, G.GAME_HIRSCH)
    p1, _ = G.payoffs(aumann.game, aumann.b, aumann.t)
    assert sorted(p1.flatten().tolist()) == [0.0, 7.0, 8.0, 9.0]
    assert S.Settings(**C.PRESETS["Geheimer Handschlag"]).talk == G.TALK_TWO
    assert S.Settings(**C.PRESETS["Lewis: drei Zustände"]).part == C.PART_LEWIS
