"""Ereignisstudie zum Austritt aus der Altkonvention bei einseitigem Reden (Hypothese H10, DEMO-PLAYBOOK §4: erst messen, dann erklären).

Vermutung des Plans: ungenutzte Nachrichten lassen die Antwortregeln der Hörer neutral treiben (Anteil „Nachricht m → A“ etwa ½), und ein Sprecher probiert A, sobald dieser Anteil über der Schwelle p*
liegt. Die Momentaufnahmen der Messreihen (vor_experiments, `s_max_pre`) widersprechen dem: bei den noch nicht ausgetretenen Läufen liegt der größte Hörer-Anteil weit unter p*. Diese Studie sieht
jeden einzelnen Schritt: Sie hält für jeden Lauf in jedem Schritt fest, wie viele Sprecher jede Nachricht senden, wie viele davon A spielen und wie hoch der Hörer-Anteil ist, und schaut bei den
Läufen, die austreten, zurück auf den Zeitpunkt der ersten Zeit, zu dem mindestens die Hälfte der Sprecher A spielt.

Austritt: erstmals mindestens die Hälfte der Sprecher spielt A. Austrittsnachricht: die Nachricht mit den meisten A-Sprechern in diesem Schritt. Rückblick: dieselbe Nachricht k Schritte früher (Zahl der
Sprecher auf ihr, Hörer-Anteil auf sie). Die Ergebnisse stehen in data/mechanism.json (tools/make_mechanism.py) mit Erzeuger-Hash."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import vor_constants as C
import vor_experiments as XP
import vor_games as G
import vor_learning as L

ROOT = Path(__file__).resolve().parent
FILE = ROOT / "data" / "mechanism.json"
HASHED_FILES = ("vor_mechanism.py", "vor_learning.py", "vor_games.py", "vor_rng.py")
LAGS = (200, 100, 50, 20, 0)
CELLS = ((3.6, 2, 20.0), (3.6, 3, 20.0), (4.5, 3, 20.0), (3.6, 3, 40.0))       # (b, K, β): p* = 0,4 / 0,4 / 0,5 / 0,4
STEPS, RUNS, BLOCKS, N = 40_000, 200, 4, 20


def event_study(b: float, K: int, beta: float, n_agents: int, steps: int, runs: int, seed: int) -> dict:
    """Ein Block: Lernlauf in der Hirschjagd mit Hase-Auszahlung b (t = 0), einseitiges Reden mit K Nachrichten, Start in der Altkonvention, Aufzeichnung in jedem Schritt.
    Gibt für die ausgetretenen Läufe (Hälfte der Sprecher spielt A) die Zahl der Sprecher n und den Hörer-Anteil s der Austrittsnachricht k Schritte vor dem Austritt, dazu die Zahl der Austritte
    und, als Grundlage, den Mittelwert von n über alle Nachrichten und Schritte der noch nicht ausgetretenen Läufe."""
    ext = G.build(G.GAME_HIRSCH, b, 0.0, G.TALK_ONE, K)
    stream = L.Stream(seed)
    cx, cy = L.initial_counts(ext, n_agents, runs, C.START_OLD, stream)
    U1n, U2n = L.scaled_payoffs(ext)
    width = 2 + max(ext.S1, ext.S2)
    rows = np.arange(runs)
    by_msg = (ext.msg1[:, None] == np.arange(K)[None, :]).astype(np.int64)                                  # (S1, K)
    by_msg_a = by_msg * np.array([[1 if a == 0 else 0] for _, a in ext.s1])                                 # nur A-Sprecher
    resp = np.array([[1.0 if rule[m] == 0 else 0.0 for m in range(K)] for _, rule in ext.s2])              # (S2, K)
    n_hist = np.zeros((steps + 1, runs, K), dtype=np.int16)
    a_hist = np.zeros((steps + 1, runs, K), dtype=np.int16)
    s_hist = np.zeros((steps + 1, runs, K), dtype=np.float32)
    n_hist[0], a_hist[0], s_hist[0] = cx @ by_msg, cx @ by_msg_a, (cy / n_agents) @ resp
    done = 0
    while done < steps:
        c = min(L.CHUNK, steps - done)
        u = stream.take(c * runs * width).reshape(c, runs, width)
        for i in range(c):
            L.revise_step(cx, cy, u[i], rows, U1n, U2n, beta, n_agents)
            done += 1
            n_hist[done], a_hist[done], s_hist[done] = cx @ by_msg, cx @ by_msg_a, (cy / n_agents) @ resp
    a_total = a_hist.sum(axis=2)                                                                            # (steps + 1, runs)
    out = {"lags": list(LAGS), "exits": 0, "runs": runs, "n": {str(k): [] for k in LAGS}, "s": {str(k): [] for k in LAGS}, "a": {str(k): [] for k in LAGS}}
    out["mean_n_before"], out["mean_s_before"] = 0.0, 0.0
    hit = a_total >= n_agents / 2
    first = np.where(hit.any(axis=0), hit.argmax(axis=0), -1)
    pre_mask = np.arange(steps + 1)[:, None] < np.where(first < 0, steps + 1, first)[None, :]
    out["mean_n_before"] = float(n_hist[pre_mask].mean())
    out["mean_s_before"] = float(s_hist[pre_mask].mean())
    for r in np.nonzero(first >= 0)[0]:
        t = int(first[r])
        m = int(a_hist[t, r].argmax())
        out["exits"] += 1
        for k in LAGS:
            j = max(0, t - k)
            out["n"][str(k)].append(int(n_hist[j, r, m]))
            out["s"][str(k)].append(float(s_hist[j, r, m]))
            out["a"][str(k)].append(int(a_hist[j, r, m]))
    return out


def lottery_study(beta: float, n_agents: int, steps: int, runs: int, seed: int) -> dict:
    """Wird die Lotterie gelernt? Kampf der Geschlechter, beidseitiges Reden mit K = 2, zufälliger Start: Anteil der Akteure beider Populationen am Ende, die genau die Lotterie-Regel spielen
    (gleiche Nachricht → A, sonst B; zwei Strategien, je eine für jede eigene Nachricht), je Lauf. Bei gleichverteilter Wahl wären es 2 von 32 Strategien."""
    import vor_equilibria as E
    ext = G.build(G.GAME_BOS, 7.0, 0.0, G.TALK_TWO, 2)
    idx = np.nonzero(E.lottery_profile(ext)[0])[0]
    stream = L.Stream(seed)
    cx, cy = L.initial_counts(ext, n_agents, runs, C.START_RANDOM, stream)
    U1n, U2n = L.scaled_payoffs(ext)
    width = 2 + max(ext.S1, ext.S2)
    rows = np.arange(runs)
    done = 0
    while done < steps:
        c = min(L.CHUNK, steps - done)
        u = stream.take(c * runs * width).reshape(c, runs, width)
        for i in range(c):
            L.revise_step(cx, cy, u[i], rows, U1n, U2n, beta, n_agents)
        done += c
    share = (cx[:, idx].sum(axis=1) + cy[:, idx].sum(axis=1)) / (2 * n_agents)
    return {"shares": [float(v) for v in share], "uniform": len(idx) / ext.S1}


def run_all() -> dict:
    cells = []
    for b, K, beta in CELLS:
        blocks = [event_study(b, K, beta, N, STEPS, RUNS, XP.seed_of(f"mech|{b}|{K}|{beta}|b{i}")) for i in range(BLOCKS)]
        pooled = {"b": b, "K": K, "beta": beta, "pstar": round(b / 9, 6), "n_agents": N, "steps": STEPS, "runs": RUNS * BLOCKS, "exits": sum(x["exits"] for x in blocks),
                  "mean_n_before": float(np.mean([x["mean_n_before"] for x in blocks])), "mean_s_before": float(np.mean([x["mean_s_before"] for x in blocks])), "lags": list(LAGS)}
        for key in ("n", "s", "a"):
            pooled[key] = {str(k): [v for x in blocks for v in x[key][str(k)]] for k in LAGS}
        pooled["limit"] = XP.lone_sender_limit(beta, N, b / 9)
        cells.append(pooled)
    studies = [lottery_study(20.0, N, STEPS, RUNS, XP.seed_of(f"lottery|b{i}")) for i in range(BLOCKS)]
    shares = [v for st in studies for v in st["shares"]]
    lottery = {"beta": 20.0, "n_agents": N, "steps": STEPS, "runs": RUNS * BLOCKS, "mean_share": float(np.mean(shares)), "max_share": float(np.max(shares)), "runs_at_least_half": int(sum(v >= 0.5 for v in shares)),
               "uniform_share": studies[0]["uniform"]}
    return {"cells": cells, "lottery": lottery}


def generator_hash() -> str:
    h = hashlib.sha256()
    for name in HASHED_FILES + ("vor_constants.py",):
        h.update(name.encode("utf-8") + b"\0" + (ROOT / name).read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()[:16]


def load() -> dict | None:
    """Die Ergebnisse der Ereignisstudie, oder None, wenn die Datei fehlt oder zu anderem Rechencode gehört."""
    if not FILE.exists():
        return None
    data = json.loads(FILE.read_text(encoding="utf-8"))
    return data if data.get("generator_hash") == generator_hash() else None


def write(payload: dict) -> None:
    FILE.parent.mkdir(exist_ok=True)
    FILE.write_text(json.dumps({"generator_hash": generator_hash(), **payload}, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")


def summary(cell: dict) -> dict:
    """Mittelwerte der Rückblicke einer Zelle: je Abstand k vor dem Austritt mittlere Sprecherzahl n, Hörer-Anteil s und Zahl der A-Sprecher a der Austrittsnachricht; Anteil der Austritte, bei denen
    die Austrittsnachricht k Schritte vorher höchstens `limit` Sprecher hatte (limit: größtes n, bei dem ein einzelner A-Sprecher den Hörern einen Anteil über p* verschafft)."""
    out = {}
    for k in cell["lags"]:
        n = np.array(cell["n"][str(k)], dtype=float)
        s = np.array(cell["s"][str(k)], dtype=float)
        a = np.array(cell["a"][str(k)], dtype=float)
        out[k] = {"n": float(n.mean()) if len(n) else None, "s": float(s.mean()) if len(s) else None, "a": float(a.mean()) if len(a) else None,
                  "within_limit": float((n <= cell["limit"]).mean()) if len(n) else None}
    return out
