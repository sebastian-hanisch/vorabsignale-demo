"""Die drei 2×2-Spiele und das um kostenloses Reden erweiterte Spiel (DEMO-PLAYBOOK §1b: eine Einheit je Regel).

Aktion 0 = A (Hirsch, Treffpunkt 1), Aktion 1 = B (Hase, Treffpunkt 2). `P1[a1, a2]` ist die Auszahlung der Zeile, `P2[a1, a2]` die der Spalte (Zeitersparnis je Sendung, willkürliche Einheiten).

Spiele: reine Koordination (1 bei gleicher Wahl); Hirschjagd (A/A 9, A allein 0, B gegen B `b`, B gegen A `b + t`; die Schwelle ist p* = b / (9 − t) im Sinne des Gleichgewichts ohne Reden,
für t = 0 also b / 9); Kampf der Geschlechter (Zeile 2/1, Spalte 1/2, sonst 0).

Erweitertes Spiel: vor den Aktionen schickt der Sprecher (einseitig die Zeile, beidseitig beide gleichzeitig) eine kostenlose Nachricht aus K Nachrichten; die Nachricht ändert keine Auszahlung.
Reine Strategien des erweiterten Spiels:
* keins: (None, a), je Seite 2 Strategien;
* einseitig: Zeile (Nachricht m, Aktion a), Spalte eine Regel r mit r[m] = Aktion nach Nachricht m, 2^K Regeln;
* beidseitig: (eigene Nachricht m, Regel r) mit r[m_eigen · K + m_fremd] = Aktion, K · 2^(K²) Strategien je Seite (K = 2: 32, K = 3: 1 536)."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

GAME_COORD = "Reine Koordination"
GAME_HIRSCH = "Hirschjagd"
GAME_BOS = "Kampf der Geschlechter"
GAMES = (GAME_COORD, GAME_HIRSCH, GAME_BOS)

TALK_NONE = "keins"
TALK_ONE = "Zeile spricht"
TALK_TWO = "beide sprechen"
TALKS = (TALK_NONE, TALK_ONE, TALK_TWO)
_KIND_OF_TALK = {TALK_NONE: "none", TALK_ONE: "one", TALK_TWO: "two"}

HIRSCH_PAYOFF = 9.0       # Auszahlung, wenn beide Hirsch jagen


def payoffs(game: str, b: float = 7.0, t: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Auszahlungsmatrizen (P1, P2) mit Index 0 = A, 1 = B. Hirschjagd: Hase gegen Hase `b`, Hase gegen Hirsch `b + t`, Hirsch allein 0, Hirsch gegen Hirsch 9; Spalte spiegelt die Zeile (P2 = P1ᵀ)."""
    if game == GAME_COORD:
        p = np.array([[1.0, 0.0], [0.0, 1.0]])
        return p, p.copy()
    if game == GAME_HIRSCH:
        p = np.array([[HIRSCH_PAYOFF, 0.0], [b + t, b]])
        return p, p.T.copy()
    if game == GAME_BOS:
        return np.array([[2.0, 0.0], [0.0, 1.0]]), np.array([[1.0, 0.0], [0.0, 2.0]])
    raise ValueError(f"unbekanntes Spiel: {game}")


def payoffs_exact(game: str, b: float = 7.0, t: float = 0.0) -> tuple[list[list[Fraction]], list[list[Fraction]]]:
    """Wie `payoffs`, aber als Brüche (b und t werden auf sechs Stellen gerundet gelesen), für die exakte Rechnung."""
    fb, ft = Fraction(f"{b:.6f}"), Fraction(f"{t:.6f}")
    if game == GAME_COORD:
        p = [[Fraction(1), Fraction(0)], [Fraction(0), Fraction(1)]]
        return p, [row[:] for row in p]
    if game == GAME_HIRSCH:
        p = [[Fraction(9), Fraction(0)], [fb + ft, fb]]
        return p, [[p[0][0], p[1][0]], [p[0][1], p[1][1]]]
    if game == GAME_BOS:
        return [[Fraction(2), Fraction(0)], [Fraction(0), Fraction(1)]], [[Fraction(1), Fraction(0)], [Fraction(0), Fraction(2)]]
    raise ValueError(f"unbekanntes Spiel: {game}")


def threshold(b: float, t: float = 0.0) -> float:
    """Schwelle p* der Hirschjagd ohne Reden: der Anteil der Hirsch-Jäger beim Partner, ab dem Hirsch mindestens so gut ist wie Hase: 9·p = b + t·p, also p* = b / (9 − t)."""
    return b / (HIRSCH_PAYOFF - t)


@dataclass(frozen=True)
class Ext:
    """Das erweiterte Spiel: Strategienlisten, die Aktionen `act1[i, j]` und `act2[i, j]`, die entstehen, wenn Zeilenstrategie i auf Spaltenstrategie j trifft, und die Auszahlungen `U1`, `U2`."""
    kind: str                  # "none", "one", "two"
    K: int                     # Zahl der Nachrichten (0 ohne Reden)
    s1: tuple                  # Strategien der Zeile
    s2: tuple                  # Strategien der Spalte
    msg1: np.ndarray           # gesendete Nachricht je Zeilenstrategie (−1: sendet nicht)
    msg2: np.ndarray           # ebenso für die Spalte
    act1: np.ndarray           # (S1, S2) Aktion der Zeile
    act2: np.ndarray           # (S1, S2) Aktion der Spalte
    P1: np.ndarray
    P2: np.ndarray
    U1: np.ndarray             # (S1, S2) Auszahlung der Zeile
    U2: np.ndarray             # (S1, S2) Auszahlung der Spalte

    @property
    def S1(self) -> int:
        return len(self.s1)

    @property
    def S2(self) -> int:
        return len(self.s2)


def talk_kind(talk: str) -> str:
    return _KIND_OF_TALK[talk]


def extended(P1: np.ndarray, P2: np.ndarray, kind: str, K: int) -> Ext:
    """Das erweiterte Spiel zu den Auszahlungen P1, P2: kind "none" (K wird ignoriert), "one" oder "two" mit K ≥ 1 Nachrichten."""
    if kind == "none":
        K = 0
        s1 = tuple((None, a) for a in range(2))
        s2 = tuple((None, a) for a in range(2))
        msg1, msg2 = np.full(2, -1), np.full(2, -1)
        act1 = np.array([[0, 0], [1, 1]])
        act2 = np.array([[0, 1], [0, 1]])
    elif kind == "one":
        if K < 1:
            raise ValueError("einseitiges Reden braucht mindestens eine Nachricht")
        s1 = tuple((m, a) for m in range(K) for a in range(2))
        s2 = tuple((None, r) for r in itertools.product(range(2), repeat=K))
        msg1 = np.array([m for m, _ in s1])
        msg2 = np.full(len(s2), -1)
        rule = np.array([r for _, r in s2])                          # (S2, K): Aktion der Spalte nach Nachricht m
        act1 = np.repeat(np.array([a for _, a in s1])[:, None], len(s2), axis=1)
        act2 = rule[:, msg1].T                                       # act2[i, j] = rule[j, msg1[i]]
    elif kind == "two":
        if K < 1:
            raise ValueError("beidseitiges Reden braucht mindestens eine Nachricht")
        s1 = tuple((m, r) for m in range(K) for r in itertools.product(range(2), repeat=K * K))
        s2 = s1
        M = np.array([m for m, _ in s1])
        R = np.array([r for _, r in s1])
        idx = np.arange(len(s1))
        msg1, msg2 = M, M.copy()
        act1 = R[idx[:, None], M[:, None] * K + M[None, :]]          # Zeile i: eigene Nachricht M[i], fremde M[j]
        act2 = R[idx[None, :], M[None, :] * K + M[:, None]]
    else:
        raise ValueError(f"unbekannte Art des Redens: {kind}")
    return Ext(kind, K, s1, s2, msg1, msg2, act1, act2, P1, P2, P1[act1, act2], P2[act1, act2])


def build(game: str, b: float, t: float, talk: str, K: int) -> Ext:
    """Das erweiterte Spiel zu Spiel, Hase-Auszahlung b, Verlockung t, Art des Redens und Zahl der Nachrichten."""
    P1, P2 = payoffs(game, b, t)
    return extended(P1, P2, talk_kind(talk), K)


def play_direct(ext: Ext, i: int, j: int) -> tuple[int, int, float, float]:
    """Unabhängiger Rechenweg für die Paarung (Zeilenstrategie i, Spaltenstrategie j) als Spielablauf: Nachricht senden, Regel anwenden, Aktion wählen, auszahlen.
    Gibt (Aktion Zeile, Aktion Spalte, Auszahlung Zeile, Auszahlung Spalte)."""
    if ext.kind == "none":
        a1, a2 = ext.s1[i][1], ext.s2[j][1]
    elif ext.kind == "one":
        m, a1 = ext.s1[i]
        a2 = ext.s2[j][1][m]
    else:
        m1, r1 = ext.s1[i]
        m2, r2 = ext.s2[j]
        a1 = r1[m1 * ext.K + m2]
        a2 = r2[m2 * ext.K + m1]
    return a1, a2, float(ext.P1[a1, a2]), float(ext.P2[a1, a2])


def strategy_label(ext: Ext, side: int, index: int) -> str:
    """Kurzbeschreibung einer reinen Strategie in Worten (A = Hirsch bzw. Treffpunkt 1, B = Hase bzw. Treffpunkt 2); side 0 = Zeile, 1 = Spalte."""
    names = ("A", "B")
    strat = (ext.s1 if side == 0 else ext.s2)[index]
    if ext.kind == "none":
        return f"spiele {names[strat[1]]}"
    if ext.kind == "one":
        if side == 0:
            return f"sende {strat[0] + 1}, spiele {names[strat[1]]}"
        return "Regel: " + ", ".join(f"nach {m + 1} → {names[a]}" for m, a in enumerate(strat[1]))
    m, r = strat
    K = ext.K
    rules = ", ".join(f"({own + 1},{fremd + 1}) → {names[r[own * K + fremd]]}" for own in range(K) for fremd in range(K))
    return f"sende {m + 1}; " + rules


def old_convention(ext: Ext, side: int) -> int:
    """Index der Altkonvention „alle spielen B“: Zeile sendet (falls möglich) Nachricht 1 und spielt B, Spalte ignoriert die Nachricht und spielt B."""
    strategies = ext.s1 if side == 0 else ext.s2
    for index, strat in enumerate(strategies):
        if ext.kind == "none" and strat[1] == 1:
            return index
        if ext.kind == "one":
            if side == 0 and strat == (0, 1):
                return index
            if side == 1 and all(a == 1 for a in strat[1]):
                return index
        if ext.kind == "two" and strat[0] == 0 and all(a == 1 for a in strat[1]):
            return index
    raise RuntimeError("keine Altkonvention gefunden")
