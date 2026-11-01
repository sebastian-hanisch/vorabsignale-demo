"""Szenario: die Einstellungen der App (Spiel, Reden, Lernen, Lewis-Spiel) und die daraus abgeleiteten Größen. Der Zufall kommt aus SplitMix64 (vor_rng)."""

from __future__ import annotations

from dataclasses import dataclass, replace

import vor_constants as C
import vor_games as G


@dataclass(frozen=True, kw_only=True)
class Settings:
    """Alle Regler der Seitenleiste. Nur mit Schlüsselwörtern anlegen: ein eingefügtes Feld darf nie still Werte verschieben (Lehre aus früheren Demos)."""
    part: str = C.PART_TALK              # Kostenloses Reden oder Lewis-Signalspiel
    game: str = G.GAME_HIRSCH
    b: float = C.DEFAULT_B               # Auszahlung des Hasen (Hirschjagd)
    t: float = C.DEFAULT_T               # Verlockung: Hase allein gegen Hirsch bringt b + t
    talk: str = G.TALK_ONE
    K: int = C.DEFAULT_K                 # Zahl der Nachrichten
    beta: float = C.DEFAULT_BETA         # Schärfe der Logit-Wahl
    n_agents: int = C.DEFAULT_N          # Akteure je Population
    start: str = C.START_RANDOM
    steps: int = C.DEFAULT_STEPS
    runs: int = C.DEFAULT_RUNS
    seed: int = C.DEFAULT_SEED
    lewis_n: int = C.DEFAULT_LEWIS_N     # Zustände
    lewis_ns: int = C.DEFAULT_LEWIS_NS   # Signale
    lewis_rounds: int = C.DEFAULT_LEWIS_ROUNDS


def settings_of_preset(name: str) -> Settings:
    return Settings(**C.PRESETS[name])


def with_(s: Settings, **changes) -> Settings:
    """Kopie mit geänderten Feldern (nur Schlüsselwörter)."""
    return replace(s, **changes)


def effective_t(s: Settings) -> float:
    """Die Verlockung wirkt nur bei Hirschjagd; sonst 0."""
    return s.t if s.game == G.GAME_HIRSCH else 0.0


def effective_b(s: Settings) -> float:
    """Die Hase-Auszahlung wirkt nur bei Hirschjagd; sonst der Standardwert (ohne Wirkung)."""
    return s.b if s.game == G.GAME_HIRSCH else C.DEFAULT_B


def messages(s: Settings) -> int:
    """Zahl der Nachrichten, die das Spiel wirklich hat: ohne Reden 0."""
    return 0 if s.talk == G.TALK_NONE else s.K


def ext_of(s: Settings) -> G.Ext:
    """Das erweiterte Spiel der Einstellungen."""
    return G.build(s.game, effective_b(s), effective_t(s), s.talk, messages(s))


def threshold(s: Settings) -> float | None:
    """Schwelle p* der Hirschjagd ohne Reden; bei den anderen Spielen None."""
    return G.threshold(s.b, s.t) if s.game == G.GAME_HIRSCH else None
