"""Lernen im erweiterten Spiel: zwei Populationen (Zeile, Spalte) zu je N Akteuren, jeder hält eine reine Strategie des erweiterten Spiels (Logit-Lernen mit Rauschen, die Lernregel aus
`gleichgewichtsauswahl-demo` als Modellannahme; hier eigener Code).

**Regel.** Je Schritt wählt ein Zufall eine der beiden Populationen (je ½) und einen Akteur darin gleichverteilt; er überprüft seine Strategie: er wählt die Strategie s mit Wahrscheinlichkeit
∝ exp(β · u(s)), wobei u(s) die erwartete Auszahlung gegen die aktuelle Verteilung der anderen Population ist (Auszahlungen auf Höchstwert 1 skaliert, damit β dieselbe Bedeutung in allen Spielen
hat). Umgesetzt als Gumbel-Max: argmax(β·u(s) + G_s) mit unabhängigen Gumbel-Variablen G_s = −ln(−ln V). Die Akteure, die nicht dran sind, bleiben unverändert.

**Reihenfolge der Zufallszahlen** (SplitMix64, `vor_rng.uniform_block`): erst die Anfangsstrategien (Form (Läufe, 2, N), Strategie = ⌊u · S⌋; entfällt bei Altkonvention, die Zahlen werden dann
trotzdem verbraucht), dann je Schritt und Lauf 2 + S_max Gleichverteilte: Population (u < ½: Zeile), Akteur (⌊u · N⌋), dann die S_max Zahlen für die Gumbel-Variablen. Läufe laufen vektorisiert
nebeneinander; die Blockgröße beim Ziehen ändert die Zahlenfolge nicht."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import vor_constants as C
import vor_games as G
from vor_rng import _GAMMA, _MASK, uniform_block

CHUNK = 50                 # so viele Schritte werden mit einem Block Zufallszahlen vorab gezogen
START_RANDOM, START_OLD = C.START_RANDOM, C.START_OLD


def block_seed(seed: int, offset: int) -> int:
    """Seed des SplitMix64-Stroms ab Position `offset` (der Generator ist ein Zähler: Seed + offset·γ)."""
    return (seed + offset * _GAMMA) & _MASK


class Stream:
    """Fortlaufender SplitMix64-Strom in Blöcken: `Stream(seed).take(a)` und danach `take(b)` zusammen sind genau `uniform_block(seed, a + b)`."""

    def __init__(self, seed: int):
        self.seed = seed & _MASK
        self.pos = 0

    def take(self, count: int) -> np.ndarray:
        u = uniform_block(block_seed(self.seed, self.pos), count)
        self.pos += count
        return u


def gumbel(u: np.ndarray) -> np.ndarray:
    """Gumbel-Variablen aus Gleichverteilten u ∈ [0, 1): G = −ln(−ln V) mit V = (u + ½ · 2⁻⁵³) in (0, 1), damit der Logarithmus endlich bleibt."""
    v = u + 0.5 * 2.0 ** -53
    return -np.log(-np.log(v))


def scaled_payoffs(ext: G.Ext) -> tuple[np.ndarray, np.ndarray]:
    """Auszahlungen auf Höchstwert 1 skaliert (Division durch das Maximum beider Matrizen)."""
    scale = max(float(ext.U1.max()), float(ext.U2.max()))
    return ext.U1 / scale, ext.U2 / scale


def initial_counts(ext: G.Ext, n_agents: int, runs: int, start: str, stream: Stream) -> tuple[np.ndarray, np.ndarray]:
    """Anfangsbesetzung beider Populationen, Form (Läufe, S1) und (Läufe, S2): zufällig (jeder Akteur gleichverteilt über die Strategien) oder alle auf der Altkonvention „alle B“."""
    u = stream.take(runs * 2 * n_agents).reshape(runs, 2, n_agents)
    cx = np.zeros((runs, ext.S1), dtype=np.int64)
    cy = np.zeros((runs, ext.S2), dtype=np.int64)
    if start == START_OLD:
        cx[:, G.old_convention(ext, 0)] = n_agents
        cy[:, G.old_convention(ext, 1)] = n_agents
        return cx, cy
    for pop, (cnt, S) in enumerate(((cx, ext.S1), (cy, ext.S2))):
        idx = np.minimum((u[:, pop, :] * S).astype(np.int64), S - 1)
        for r in range(runs):
            cnt[r] = np.bincount(idx[r], minlength=S)
    return cx, cy


def revise_step(cx: np.ndarray, cy: np.ndarray, pick: np.ndarray, rows: np.ndarray, U1n: np.ndarray, U2n: np.ndarray, beta: float, n_agents: int) -> None:
    """Ein Überprüfungsschritt für alle Läufe, in place. pick: Form (Läufe, 2 + S_max) Gleichverteilte [Population, Akteur, Gumbel-Zahlen]."""
    side = (pick[:, 0] < 0.5).astype(np.int64)                               # 1: die Zeile überprüft, 0: die Spalte
    who = np.minimum((pick[:, 1] * n_agents).astype(np.int64), n_agents - 1)
    g = gumbel(pick[:, 2:])
    inv = 1.0 / n_agents
    new1 = np.argmax(beta * ((cy * inv) @ U1n.T) + g[:, : cx.shape[1]], axis=1)
    new2 = np.argmax(beta * ((cx * inv) @ U2n) + g[:, : cy.shape[1]], axis=1)
    old1 = (np.cumsum(cx, axis=1) > who[:, None]).argmax(axis=1)
    old2 = (np.cumsum(cy, axis=1) > who[:, None]).argmax(axis=1)
    cx[rows, old1] -= side
    cx[rows, new1] += side
    cy[rows, old2] -= 1 - side
    cy[rows, new2] += 1 - side


def mutual_information(ext: G.Ext, cx: np.ndarray, cy: np.ndarray, n_agents: int) -> np.ndarray:
    """Information der Nachricht in Bit, je Lauf: bei einseitigem Reden I(Nachricht der Zeile; Aktion der Spalte), bei beidseitigem Reden die Summe beider Richtungen
    (I(Nachricht der Zeile; Aktion der Spalte) + I(Nachricht der Spalte; Aktion der Zeile)); ohne Reden 0. Exakt aus den Besetzungen, nicht aus Ziehungen."""
    x, y = cx / n_agents, cy / n_agents
    total = np.zeros(x.shape[0])
    legs = []
    if ext.kind in ("one", "two"):
        legs.append((ext.msg1, ext.act2, x, y, False))
    if ext.kind == "two":
        legs.append((ext.msg2, ext.act1, y, x, True))
    for msg, act, w_msg, w_act, swapped in legs:
        Km = int(msg.max()) + 1
        OHm = (msg[:, None] == np.arange(Km)[None, :]).astype(float)
        OHa = (act[..., None] == np.arange(2)).astype(float)
        if not swapped:
            joint = np.einsum("ri,rj,im,ija->rma", w_msg, w_act, OHm, OHa)
        else:                                                                 # Nachricht gehört zum Index j (Spalte), Aktion der Zeile steht in act[i, j]
            joint = np.einsum("rj,ri,jm,ija->rma", w_msg, w_act, OHm, OHa)
        pm = joint.sum(2, keepdims=True)
        pa = joint.sum(1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(joint > 0, joint * np.log2(joint / (pm * pa)), 0.0)
        total += term.sum(axis=(1, 2))
    return total


def outcome_shares(ext: G.Ext, cx: np.ndarray, cy: np.ndarray, n_agents: int) -> np.ndarray:
    """Verteilung der Aktionspaare je Lauf, Form (Läufe, 2, 2): q[r, a, b] = Wahrscheinlichkeit, dass bei zufälliger Paarung Zeile a und Spalte b spielt."""
    x, y = cx / n_agents, cy / n_agents
    q = np.zeros((x.shape[0], 2, 2))
    for a in range(2):
        for b in range(2):
            mask = ((ext.act1 == a) & (ext.act2 == b)).astype(float)
            q[:, a, b] = np.einsum("ri,ij,rj->r", x, mask, y)
    return q


def listener_share_hirsch(ext: G.Ext, cy: np.ndarray, n_agents: int) -> np.ndarray:
    """Einseitiges Reden: Anteil der Hörer, die auf Nachricht m mit A (Hirsch) antworten, Form (Läufe, K): s[r, m]. Sonst Form (Läufe, 0)."""
    if ext.kind != "one":
        return np.zeros((cy.shape[0], 0))
    resp = np.array([[1.0 if rule[m] == 0 else 0.0 for m in range(ext.K)] for _, rule in ext.s2])      # (S2, K)
    return (cy / n_agents) @ resp


def speaker_share(ext: G.Ext, cx: np.ndarray, n_agents: int) -> np.ndarray:
    """Anteil der Sprecher (Zeile), die Nachricht m senden, Form (Läufe, K); ohne Reden Form (Läufe, 0)."""
    if ext.kind == "none":
        return np.zeros((cx.shape[0], 0))
    onehot = (ext.msg1[:, None] == np.arange(ext.K)[None, :]).astype(float)
    return (cx / n_agents) @ onehot


def handshake_share(ext: G.Ext, cx: np.ndarray, cy: np.ndarray, n_agents: int) -> np.ndarray:
    """Beidseitiges Reden: Anteil der Akteure beider Populationen, die eine Nachricht ≠ 1 senden und bei gleicher fremder Nachricht A (Hirsch) spielen (der „Handschlag“), Form (Läufe,)."""
    if ext.kind != "two":
        return np.zeros(cx.shape[0])
    K = ext.K
    hs = np.array([1.0 if (m != 0 and r[m * K + m] == 0) else 0.0 for m, r in ext.s1])
    return 0.5 * ((cx / n_agents) @ hs + (cy / n_agents) @ hs)


@dataclass(frozen=True)
class Snapshot:
    """Kennzahlen aller Läufe zu einem Zeitpunkt (jede Reihe hat Form (Läufe,), bis auf die mit K Spalten)."""
    step: int
    efficiency: np.ndarray       # (Summe beider Auszahlungen) / bestmögliche Summe
    payoff1: np.ndarray
    payoff2: np.ndarray
    aa: np.ndarray               # Anteil (A, A) bei zufälliger Paarung
    bb: np.ndarray               # Anteil (B, B)
    mis: np.ndarray              # Anteil Fehlkoordination (A, B) oder (B, A)
    info: np.ndarray             # Information der Nachricht in Bit
    hear_a: np.ndarray           # (Läufe, K) Anteil der Hörer mit „Nachricht m → A“ (einseitig)
    speak: np.ndarray            # (Läufe, K) Anteil der Sprecher mit Nachricht m
    handshake: np.ndarray        # Handschlag-Anteil (beidseitig)


def snapshot(ext: G.Ext, cx: np.ndarray, cy: np.ndarray, n_agents: int, step: int) -> Snapshot:
    q = outcome_shares(ext, cx, cy, n_agents)
    e1 = (q * ext.P1).sum(axis=(1, 2))
    e2 = (q * ext.P2).sum(axis=(1, 2))
    best = float((ext.P1 + ext.P2).max())
    return Snapshot(step, (e1 + e2) / best, e1, e2, q[:, 0, 0], q[:, 1, 1], q[:, 0, 1] + q[:, 1, 0], mutual_information(ext, cx, cy, n_agents),
                    listener_share_hirsch(ext, cy, n_agents), speaker_share(ext, cx, n_agents), handshake_share(ext, cx, cy, n_agents))


def population_run(ext: G.Ext, beta: float, n_agents: int, steps: int, runs: int, seed: int, start: str = START_RANDOM, checkpoints: tuple[int, ...] | None = None) -> list[Snapshot]:
    """Lässt `runs` unabhängige Läufe `steps` Schritte lernen; Rückgabe: Momentaufnahmen an `checkpoints` (Standard: nur am Ende, Schritt 0 = Anfang ist erlaubt), aufsteigend."""
    if steps < 0 or runs < 1 or n_agents < 1:
        raise ValueError("steps ≥ 0, runs ≥ 1 und n_agents ≥ 1 verlangt")
    checks = sorted(set(checkpoints)) if checkpoints is not None else [steps]
    if checks and (checks[0] < 0 or checks[-1] > steps):
        raise ValueError("Zeitpunkte außerhalb von 0 bis steps")
    stream = Stream(seed)
    cx, cy = initial_counts(ext, n_agents, runs, start, stream)
    U1n, U2n = scaled_payoffs(ext)
    smax = max(ext.S1, ext.S2)
    width = 2 + smax
    rows = np.arange(runs)
    out: list[Snapshot] = []
    pending = list(checks)
    while pending and pending[0] == 0:
        out.append(snapshot(ext, cx, cy, n_agents, 0))
        pending.pop(0)
    done = 0
    while done < steps:
        n = min(CHUNK, steps - done)
        u = stream.take(n * runs * width).reshape(n, runs, width)
        for i in range(n):
            revise_step(cx, cy, u[i], rows, U1n, U2n, beta, n_agents)
            done += 1
            while pending and pending[0] == done:
                out.append(snapshot(ext, cx, cy, n_agents, done))
                pending.pop(0)
    return out


def standard_error(values: np.ndarray) -> float:
    """Standardfehler des Mittels über Blöcke (Standardabweichung mit ddof = 1 durch √Anzahl); 0 bei einem Wert."""
    v = np.asarray(values, dtype=float)
    return float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0
