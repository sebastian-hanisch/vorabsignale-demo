"""Experimente E1 bis E6: exakte Tabellen (Gleichgewichte), Zellen aus Lernläufen (live mit wenigen Läufen) und die vorgerechneten Messreihen (data/measurements.json, mit vielen Läufen).

Eine Zelle sind `blocks` unabhängige Blöcke von Läufen (je ein vektorisierter Lernlauf mit eigenem SplitMix64-Seed aus der Zellenbezeichnung); ausgewiesen werden Mittel und Standardfehler der
Blockwerte. Die vorgerechneten Reihen (tools/make_measurements.py) benutzen genau diese Funktionen mit C.MEAS_BLOCKS Blöcken zu je C.MEAS_RUNS Läufen und tragen den Hash der Dateien, aus deren Code sie
stammen.

**Mechanismus-Kennzahlen** (einseitiges Reden, Start in der Altkonvention). Ein Lauf heißt „noch drin“, solange weniger als ein Zehntel der Paarungen (A, A) sind. Für die noch drin gebliebenen
Läufe wird an jedem Zeitpunkt gemessen: s_max, der größte Anteil der Hörer, die auf irgendeine Nachricht mit A (Hirsch) antworten; „über p*“, ob s_max über der Schwelle p* der Hirschjagd liegt;
n_min, die Zahl der Sprecher auf der am seltensten gesendeten Nachricht."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

import vor_constants as C
import vor_equilibria as E
import vor_games as G
import vor_learning as L
import vor_lewis as LW
import vor_scenario as S

ROOT = Path(__file__).resolve().parent
MEAS_FILE = ROOT / "data" / "measurements.json"
HASHED_FILES = ("vor_games.py", "vor_learning.py", "vor_lewis.py", "vor_scenario.py", "vor_experiments.py", "vor_rng.py")
EXIT_LEVEL = 0.5                 # ein Lauf hat die Altkonvention verlassen, wenn mindestens die Hälfte der Paarungen (A, A) ist
PRE_LEVEL = 0.1                  # „noch drin“: weniger als ein Zehntel der Paarungen ist (A, A)


# --------------------------------------------------------------------------------- Hilfen
def seed_of(label: str) -> int:
    """Zufalls-Seed aus einer Zellenbezeichnung (FNV-1a über die UTF-8-Bytes; Python-`hash` ist je Prozess gesalzen und taugt nicht)."""
    h = 0xCBF29CE484222325
    for byte in label.encode("utf-8"):
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h % (C.SEED_MAX + 1)


def stats(blocks: list[dict]) -> dict:
    """Mittel und Standardfehler (Standardabweichung der Blockwerte durch √Blöcke) je Kennzahl; fehlt ein Wert in einem Block (None), ist das Mittel None."""
    out: dict = {"blocks": len(blocks)}
    for key in blocks[0]:
        vals = [b[key] for b in blocks]
        if isinstance(vals[0], list):
            out[key] = [stats([{"v": b[key][i]} for b in blocks])["v"] for i in range(len(vals[0]))]
            continue
        if any(v is None for v in vals):
            out[key] = {"mean": None, "se": None}
            continue
        arr = np.array(vals, dtype=float)
        out[key] = {"mean": float(arr.mean()), "se": float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0}
    return out


# --------------------------------------------------------------------------------- Kennzahlen eines Blocks
def _mech(ext: G.Ext, snap: L.Snapshot, pstar: float | None, n_agents: int) -> tuple[int, float, int, float]:
    """(Zahl der Läufe, die noch drin sind; Summe von s_max über sie; Zahl mit s_max > p*; Summe von n_min) für einseitiges Reden, sonst Nullen."""
    if ext.kind != "one" or pstar is None:
        return 0, 0.0, 0, 0.0
    pre = snap.aa < PRE_LEVEL
    smax = snap.hear_a.max(axis=1)
    nmin = np.round(snap.speak.min(axis=1) * n_agents)
    return int(pre.sum()), float(smax[pre].sum()), int((smax[pre] > pstar + 1e-12).sum()), float(nmin[pre].sum())


def block_metrics(ext: G.Ext, snaps: list[L.Snapshot], pstar: float | None, n_agents: int, trace: bool) -> dict:
    """Kennzahlen eines Blocks: Endwerte (Mittel über die Läufe), bei trace zusätzlich Verläufe an den Stützstellen und Mechanismus-Kennzahlen über alle Stützstellen."""
    last = snaps[-1]
    runs = last.aa.shape[0]
    out = {
        "eff": float(last.efficiency.mean()), "p1": float(last.payoff1.mean()), "p2": float(last.payoff2.mean()), "aa": float(last.aa.mean()), "bb": float(last.bb.mean()),
        "mis": float(last.mis.mean()), "info": float(last.info.mean()), "exit": float((last.aa >= EXIT_LEVEL).mean()), "eff95": float((last.efficiency >= 0.95).mean()),
        "hs": float(last.handshake.mean()),
    }
    if not trace:
        return out
    out["tr_aa"] = [float(s.aa.mean()) for s in snaps]
    out["tr_hs"] = [float(s.handshake.mean()) for s in snaps]
    n_pre = s_sum = n_above = n_sum = 0
    tr_pre, tr_smax, tr_above = [], [], []
    for s in snaps:
        k, ssum, above, nsum = _mech(ext, s, pstar, n_agents)
        n_pre += k
        s_sum += ssum
        n_above += above
        n_sum += nsum
        tr_pre.append(k / runs)
        tr_smax.append(ssum / k if k else None)
        tr_above.append(above / k if k else None)
    out["tr_pre"], out["tr_smax"], out["tr_above"] = tr_pre, tr_smax, tr_above
    out["pre_obs"] = n_pre / runs                                                  # Zahl der Beobachtungen (Lauf, Zeitpunkt) „noch drin“ je Lauf
    out["smax_pre"] = s_sum / n_pre if n_pre else None
    out["above_pre"] = n_above / n_pre if n_pre else None
    out["nmin_pre"] = n_sum / n_pre if n_pre else None
    return out


def checkpoints_of(steps: int, points: int = C.TRACE_POINTS) -> tuple[int, ...]:
    """Gleichmäßig verteilte Stützstellen 0 bis steps (points + 1 Werte, ohne Doppelte)."""
    return tuple(sorted({int(round(steps * i / points)) for i in range(points + 1)}))


def run_block(s: S.Settings, seed: int, trace: bool = False) -> dict:
    """Ein Block: s.runs Läufe mit dem gegebenen Zufalls-Seed; Kennzahlen siehe `block_metrics`."""
    ext = S.ext_of(s)
    snaps = L.population_run(ext, s.beta, s.n_agents, s.steps, s.runs, seed, s.start, checkpoints_of(s.steps) if trace else None)
    return block_metrics(ext, snaps, S.threshold(s), s.n_agents, trace)


def cell(s: S.Settings, blocks: int, label: str, trace: bool = False) -> dict:
    """Eine Zelle: `blocks` Blöcke mit je s.runs Läufen, Seed des Blocks b = seed_of(f"{label}|b{b}")."""
    runs = [run_block(s, seed_of(f"{label}|b{b}"), trace) for b in range(blocks)]
    out = stats(runs)
    out["runs"] = blocks * s.runs
    return out


# --------------------------------------------------------------------------------- exakte Tabellen (E1, E2, E5)
def equilibrium_row(game: str, b: float, t: float, talk: str, K: int) -> dict:
    """Gleichgewichte eines Spiels mit einer Art des Redens: Zahl der reinen Gleichgewichte, ihre Auszahlungspaare, bei einseitigem Reden mit K ≤ 2 (und ohne Reden) zusätzlich die
    Auszahlungspaare ALLER Gleichgewichte (auch gemischte) und ob ein informatives existiert; sonst None. Fehlkoordination: gibt es ein reines Gleichgewicht mit verschiedenen Aktionen?"""
    ext = G.build(game, b, t, talk, K if talk != G.TALK_NONE else 0)
    if ext.S1 * ext.S2 > 1_000_000:                                                  # beidseitig K = 3: nur zählen, ohne Liste
        ne_count, ne = E.pure_ne_count(ext.U1, ext.U2), None
    else:
        ne = E.pure_ne(ext.U1, ext.U2)
        ne_count = len(ne)
    row = {"game": game, "talk": talk, "K": K if talk != G.TALK_NONE else 0, "strategies": (ext.S1, ext.S2), "pure": ne_count,
           "pure_payoffs": E.payoff_pairs(ext, ne) if ne is not None else None, "miscoordination": E.has_miscoordination(ext, ne) if ne is not None else None,
           "all_payoffs": None, "informative": None, "difference": None, "lottery": None}
    if ext.S1 <= C.EXACT_MAX_STRATEGIES and ext.S2 <= C.EXACT_MAX_STRATEGIES:
        eq = E.all_equilibria(ext)
        row["all_boxes"] = [tuple(float(v) for v in box) for box in eq.boxes()]
        row["all_payoffs"] = eq.points() if not eq.degenerate_boxes else None
        row["informative"] = eq.informative if ext.kind == "one" and ext.K == 2 else None
        row["difference"] = eq.max_difference if ext.kind == "one" and ext.K == 2 else None
    if ext.kind == "two" and ext.K >= 2 and ext.S1 <= 40:
        x, y = E.lottery_profile(ext)
        ok, e1, e2 = E.is_ne_mixed(ext.U1, ext.U2, x, y)
        row["lottery"] = {"equilibrium": ok, "payoff1": e1, "payoff2": e2}
    return row


# --------------------------------------------------------------------------------- Mittelfeld-Rechnung des Mechanismus (exakt)
def hearer_share(beta: float, n_agents: int, pstar: float, senders: int, lone_a: int = 1, t: float = 0.0) -> float:
    """Stationärer Anteil der Hörer, die auf eine Nachricht mit A antworten, wenn `senders` Sprecher diese Nachricht senden und davon `lone_a` A spielen (die übrigen B), bei fest gehaltenen Sprechern.
    Der Nutzen der Hörerregel ist je Nachricht additiv, die Logit-Wahl faktorisiert also über die Nachrichten. Ein Hörer vergleicht A (9 gegen einen A-Sprecher, sonst 0) mit B (b + t gegen einen
    A-Sprecher, b gegen einen B-Sprecher); auf Höchstwert 9 skaliert ist der Vorteil von A gleich ((9 − t) · a − b · n) / (9 · N), also P(A) = 1 / (1 + exp(−β · (9 − t)/9 · (a − p* · n) / N))
    mit a A-Sprechern unter n Absendern und p* = b / (9 − t)."""
    return 1.0 / (1.0 + math.exp(-beta * (9.0 - t) / 9.0 * (lone_a - pstar * senders) / n_agents))


def lone_sender_limit(beta: float, n_agents: int, pstar: float, t: float = 0.0) -> int:
    """Größte Zahl n von Sprechern auf einer Nachricht, für die ein einzelner A-Spieler darunter den Hörern einen Anteil über p* verschafft (hearer_share(…, senders=n) > p*); 0, wenn schon n = 1 nicht reicht."""
    n = 0
    while n < 10 * n_agents and hearer_share(beta, n_agents, pstar, n + 1, 1, t) > pstar:
        n += 1
    return n


# --------------------------------------------------------------------------------- Lewis-Zellen
def lewis_cell(n: int, ns: int, runs: int, blocks: int, rounds: int, checks: tuple[int, ...], label: str) -> dict:
    """Lewis-Spiel (n Zustände, ns Signale): je Zeitpunkt Effizienz, Anteil der Läufe mit Effizienz ≥ 95 % des Bestwerts und Fremdspiel (Sprecher aus Lauf r gegen Hörer aus Lauf r + 1, nur für Paare,
    in denen beide Läufe den Bestwert erreicht haben; Zahl dieser Paare als Mittel je Block)."""
    best = LW.best_efficiency(n, ns)
    per_check: dict[int, list[dict]] = {c: [] for c in checks}
    for b in range(blocks):
        snaps = LW.roth_erev(n, ns, rounds, runs, seed_of(f"{label}|b{b}"), checks)
        for snap in snaps:
            eff = snap.efficiency
            ok = eff >= 0.95 * best
            both = ok & np.roll(ok, -1)
            cross = LW.cross_efficiency(snap.ps, snap.pr)
            per_check[snap.round].append({"eff": float(eff.mean()), "share": float(ok.mean()), "cross": float(cross[both].mean()) if both.any() else None, "pairs": float(both.sum())})
    out = {str(c): stats(per_check[c]) for c in checks}
    out.update({"n": n, "ns": ns, "best": best, "runs": blocks * runs})
    return out


# --------------------------------------------------------------------------------- Aufträge der vorgerechneten Reihen
def _spec(family: str, game: str, b: float, t: float, talk: str, K: int, beta: float, n_agents: int, start: str, steps: int, trace: bool) -> dict:
    return {"family": family, "game": game, "b": b, "t": t, "talk": talk, "K": K, "beta": beta, "n_agents": n_agents, "start": start, "steps": steps, "trace": trace}


def cell_specs() -> list[dict]:
    """Alle Lernzellen der vorgerechneten Reihen in fester Reihenfolge."""
    specs: list[dict] = []
    for pstar in C.THRESHOLD_PSTARS:                                                 # E3: Altkonvention verlassen, Schwelle p* × Reden × β
        for talk, K in C.E3_TALKS:
            for beta in C.E3_BETAS:
                specs.append(_spec("e3", G.GAME_HIRSCH, round(9 * pstar, 6), 0.0, talk, K, beta, 20, C.START_OLD, C.E3_STEPS, talk != G.TALK_NONE))
    for n_agents in (10, 40):                                                        # E3: Abhängigkeit von N (einseitig, K = 3, p* = 0,4 und 0,5, β = 20)
        for pstar in (0.4, 0.5):
            specs.append(_spec("e3n", G.GAME_HIRSCH, round(9 * pstar, 6), 0.0, G.TALK_ONE, 3, 20.0, n_agents, C.START_OLD, C.E3_STEPS, True))
    for t in (0.0, 1.0):                                                             # E3: geheimer Handschlag bei p* = 7/9
        for talk, K in ((G.TALK_ONE, 1), (G.TALK_TWO, 2)):
            for beta in C.HANDSHAKE_BETAS:
                specs.append(_spec("hs", G.GAME_HIRSCH, C.HANDSHAKE_B, t, talk, K, beta, 20, C.START_OLD, C.HANDSHAKE_STEPS_PER_AGENT * 20, talk == G.TALK_TWO))
    for n_agents in (10, 40):                                                        # H11: Abhängigkeit des Handschlags von N (Schritte je Akteur gleich)
        for beta in C.HANDSHAKE_BETAS:
            specs.append(_spec("hsn", G.GAME_HIRSCH, C.HANDSHAKE_B, 0.0, G.TALK_TWO, 2, beta, n_agents, C.START_OLD, C.HANDSHAKE_STEPS_PER_AGENT * n_agents, True))
    for start in C.STARTS:                                                           # E4: Kampf der Geschlechter
        for talk, K in C.E3_TALKS:
            for beta in C.BOS_BETAS:
                specs.append(_spec("bos", G.GAME_BOS, C.DEFAULT_B, 0.0, talk, K, beta, 20, start, C.BOS_STEPS, False))
    for game, t in ((G.GAME_COORD, 0.0), (G.GAME_HIRSCH, 0.0), (G.GAME_HIRSCH, 1.0)):    # zufälliger Start in den drei Spielen
        for talk, K in C.E3_TALKS:
            for beta in C.RANDOM_BETAS:
                specs.append(_spec("rand", game, C.DEFAULT_B, t, talk, K, beta, 20, C.START_RANDOM, C.RANDOM_STEPS, False))
    return specs


def spec_label(spec: dict) -> str:
    return "|".join(str(spec[k]) for k in ("family", "game", "b", "t", "talk", "K", "beta", "n_agents", "start", "steps"))


def settings_of_spec(spec: dict, runs: int) -> S.Settings:
    return S.Settings(game=spec["game"], b=spec["b"], t=spec["t"], talk=spec["talk"], K=spec["K"], beta=spec["beta"], n_agents=spec["n_agents"], start=spec["start"], steps=spec["steps"], runs=runs)


def jobs() -> list[tuple]:
    """Alle Aufträge der vorgerechneten Reihen (ein Auftrag = ein Block einer Lernzelle oder eine Lewis-Zelle), fester Reihenfolge. Art, Schlüssel, Argumente."""
    out: list[tuple] = []
    for i, spec in enumerate(cell_specs()):
        for b in range(C.MEAS_BLOCKS):
            out.append(("block", ("cell", i, b), spec, b))
    for n, ns in C.LEWIS_PAIRS:
        out.append(("lewis", ("lewis", n, ns), (n, ns)))
    return out


def run_job(job: tuple):
    """Rechnet einen Auftrag aus `jobs()`; Rückgabe (Schlüssel, Ergebnis)."""
    kind, key = job[0], job[1]
    if kind == "block":
        _, _, spec, b = job
        s = settings_of_spec(spec, C.MEAS_RUNS)
        return key, run_block(s, seed_of(f"{spec_label(spec)}|b{b}"), spec["trace"])
    if kind == "lewis":
        n, ns = job[2]
        return key, lewis_cell(n, ns, C.LEWIS_RUNS, C.LEWIS_BLOCKS, max(C.LEWIS_CHECKS), C.LEWIS_CHECKS, f"lewis|{n}|{ns}")
    raise ValueError(kind)


def assemble(results: dict) -> dict:
    """Ordnet die Ergebnisse der Aufträge (Schlüssel → Ergebnis) zu dem Wörterbuch, das in data/measurements.json steht."""
    out: dict = {"blocks": C.MEAS_BLOCKS, "runs_per_block": C.MEAS_RUNS, "cells": [], "lewis": []}
    for i, spec in enumerate(cell_specs()):                                                    # feste Reihenfolge, unabhängig davon, wann die Prozesse fertig wurden
        blocks = [results[("cell", i, b)] for b in range(C.MEAS_BLOCKS)]
        res = stats(blocks)
        res["runs"] = C.MEAS_BLOCKS * C.MEAS_RUNS
        out["cells"].append({"spec": spec, "result": res})
    for n, ns in C.LEWIS_PAIRS:
        out["lewis"].append(results[("lewis", n, ns)])
    return out


# --------------------------------------------------------------------------------- vorgerechnete Reihen: Datei
def generator_hash() -> str:
    """Hash der Dateien, aus deren Code die Messreihen stammen (Zeilenenden normalisiert, damit CRLF/LF denselben Hash geben)."""
    h = hashlib.sha256()
    for name in HASHED_FILES:
        text = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        h.update(name.encode("utf-8") + b"\0" + text + b"\0")
    return h.hexdigest()[:16]


def to_jsonable(obj):
    """Wörterbücher mit Zahlen- oder Tupelschlüsseln in JSON-fähige (Text-)Schlüssel umwandeln."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def load_measurements() -> dict | None:
    """Die vorgerechneten Reihen, oder None, wenn die Datei fehlt oder zu einem anderen Rechencode gehört (dann zeigt die App sie nicht)."""
    if not MEAS_FILE.exists():
        return None
    data = json.loads(MEAS_FILE.read_text(encoding="utf-8"))
    return data if data.get("generator_hash") == generator_hash() else None


def write_measurements(payload: dict) -> None:
    out = {"generator_hash": generator_hash(), **to_jsonable(payload)}
    MEAS_FILE.parent.mkdir(exist_ok=True)
    MEAS_FILE.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")


def find_cell(meas: dict, **fields) -> dict:
    """Die Zelle der Messdatei, deren Spezifikation die gegebenen Felder hat (genau eine); sonst KeyError."""
    hits = [c for c in meas["cells"] if all(c["spec"].get(k) == v for k, v in fields.items())]
    if len(hits) != 1:
        raise KeyError(f"{len(hits)} Zellen für {fields}")
    return hits[0]["result"]


def find_lewis(meas: dict, n: int, ns: int) -> dict:
    hits = [c for c in meas["lewis"] if c["n"] == n and c["ns"] == ns]
    if len(hits) != 1:
        raise KeyError(f"Lewis-Zelle ({n}, {ns}) nicht gefunden")
    return hits[0]
