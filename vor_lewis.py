"""Lewis-Signalspiel (Lewis 1969, im Lehrbuchmodell): n gleichwahrscheinliche Zustände, der Sprecher sieht den Zustand und sendet eines von ns Signalen, der Hörer sieht nur das Signal und wählt
eine von n Handlungen; stimmt die Handlung mit dem Zustand überein, bekommen beide 1, sonst 0. Lernen nach Roth und Erev (Urnen, Anfangsgewicht 1, Verstärkung 1 bei Erfolg).

**Reihenfolge der Zufallszahlen** (SplitMix64): je Runde und Lauf 3 Gleichverteilte [Zustand, Signal, Handlung], Form (Runden, Läufe, 3); Zustand = ⌊u · n⌋, Signal und Handlung aus den
kumulierten Gewichten der Urne (kleinster Index, dessen kumuliertes Gewicht u · Summe übersteigt)."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from vor_learning import Stream

CHUNK = 200


def efficiency(ps: np.ndarray, pr: np.ndarray) -> np.ndarray:
    """Erfolgswahrscheinlichkeit je Lauf: Ps (Läufe, n, ns) mit P(Signal | Zustand), Pr (Läufe, ns, n) mit P(Handlung | Signal); Effizienz = Σ_{k,s} Ps[k, s] · Pr[s, k] / n."""
    n = ps.shape[1]
    return np.einsum("rks,rsk->r", ps, pr) / n


def best_efficiency(n: int, ns: int) -> float:
    """Bestwert min(1, ns / n): mit weniger Signalen als Zuständen können höchstens ns Zustände unterschieden werden."""
    return min(1.0, ns / n)


@dataclass(frozen=True)
class LewisSnapshot:
    round: int
    ps: np.ndarray               # (Läufe, n, ns)
    pr: np.ndarray               # (Läufe, ns, n)

    @property
    def efficiency(self) -> np.ndarray:
        return efficiency(self.ps, self.pr)


def roth_erev(n: int, ns: int, rounds: int, runs: int, seed: int, checkpoints: tuple[int, ...] | None = None) -> list[LewisSnapshot]:
    """`runs` unabhängige Läufe des Lewis-Spiels mit Roth-Erev-Lernen, Momentaufnahmen der Wahrscheinlichkeiten an den Zeitpunkten (Standard: am Ende; 0 = Anfang)."""
    if n < 2 or ns < 1 or runs < 1 or rounds < 0:
        raise ValueError("n ≥ 2, ns ≥ 1, runs ≥ 1, rounds ≥ 0 verlangt")
    checks = sorted(set(checkpoints)) if checkpoints is not None else [rounds]
    ws = np.ones((runs, n, ns))
    wr = np.ones((runs, ns, n))
    ar = np.arange(runs)
    stream = Stream(seed)
    out: list[LewisSnapshot] = []
    pending = list(checks)

    def take(t):
        out.append(LewisSnapshot(t, ws / ws.sum(axis=2, keepdims=True), wr / wr.sum(axis=2, keepdims=True)))

    while pending and pending[0] == 0:
        take(0)
        pending.pop(0)
    done = 0
    while done < rounds:
        c = min(CHUNK, rounds - done)
        u = stream.take(c * runs * 3).reshape(c, runs, 3)
        for i in range(c):
            state = np.minimum((u[i, :, 0] * n).astype(np.int64), n - 1)
            row = ws[ar, state]
            sig = (np.cumsum(row, axis=1) > (u[i, :, 1] * row.sum(axis=1))[:, None]).argmax(axis=1)
            row = wr[ar, sig]
            act = (np.cumsum(row, axis=1) > (u[i, :, 2] * row.sum(axis=1))[:, None]).argmax(axis=1)
            win = (act == state).astype(float)
            ws[ar, state, sig] += win
            wr[ar, sig, act] += win
            done += 1
            while pending and pending[0] == done:
                take(done)
                pending.pop(0)
    return out


def pure_ne_lewis(n: int, ns: int | None = None) -> dict:
    """Alle reinen Gleichgewichte des Lewis-Spiels (n Zustände, ns Signale) durch Vollaufzählung aller Sprecherabbildungen Zustand → Signal und Hörerabbildungen Signal → Handlung.
    Gibt {"count", "by_efficiency": {Effizienz: Zahl}, "signalling_systems"} mit der Zahl der Gleichgewichte mit Effizienz 1 (bei ns = n die n! Signalsysteme)."""
    ns = n if ns is None else ns
    senders = list(itertools.product(range(ns), repeat=n))          # Zustand k → Signal
    receivers = list(itertools.product(range(n), repeat=ns))        # Signal s → Handlung
    S = np.array(senders)                                           # (n^... , n)
    R = np.array(receivers)                                         # (..., ns)
    # U[a, b] = Anteil der Zustände k mit R[b, S[a, k]] == k
    U = np.zeros((len(senders), len(receivers)))
    for a in range(len(senders)):
        U[a] = (R[:, S[a]] == np.arange(n)[None, :]).mean(axis=1)
    I, J = np.nonzero((U >= U.max(axis=0, keepdims=True) - 1e-12) & (U >= U.max(axis=1, keepdims=True) - 1e-12))
    effs = np.round(U[I, J], 9)
    values, counts = np.unique(effs, return_counts=True)
    by = {float(v): int(c) for v, c in zip(values, counts)}
    return {"count": int(len(I)), "by_efficiency": by, "signalling_systems": int(by.get(1.0, 0))}


def cross_efficiency(ps: np.ndarray, pr: np.ndarray, shift: int = 1) -> np.ndarray:
    """Fremdspiel: Sprecher aus Lauf r gegen Hörer aus Lauf r + shift (zyklisch), exakt."""
    return efficiency(ps, np.roll(pr, -shift, axis=0))
