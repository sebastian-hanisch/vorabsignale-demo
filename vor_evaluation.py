"""Kennzahlen der Hauptansicht und Verdikte (reine Funktionen: Art und Text, die App ruft `getattr(st, Art)(Text)`; DEMO-PLAYBOOK §12).

Jedes Verdikt zählt über die Zellen, die es beschreibt (k von m), und nennt die Ausnahme im selben Satz (DEMO-PLAYBOOK §11a). „Höher“, „niedriger“ und „gleich“ heißen: der Unterschied zweier Mittel
liegt über +4, unter −4 oder innerhalb von ±4 Standardfehlern (Fehler der beiden Mittel quadratisch addiert, mindestens die feste Untergrenze FLOOR). Alle Aussagen gelten für dieses Lernmodell
(Logit-Lernen zweier Populationen mit N Akteuren, bzw. Roth-Erev-Lernen im Lewis-Spiel), nicht für Menschen, Betriebe oder Sprachmodelle."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import vor_budget as BU
import vor_constants as C
import vor_equilibria as E
import vor_experiments as XP
import vor_games as G
import vor_learning as L
import vor_lewis as LW
import vor_scenario as S

Z = 4.0                       # Marge in Standardfehlern
FLOOR = 0.02                  # feste Untergrenze der Marge für Anteile


# --- Formate ---------------------------------------------------------------------------------------------------------------------------------
def de(x: float, nd: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und echtem Minuszeichen (−)."""
    text = f"{x:.{nd}f}"
    if float(text) == 0.0:
        text = text.lstrip("-")                      # „−0,000“ gibt es nicht
    return text.replace(".", ",").replace("-", "−")


def thousands(n: int | float) -> str:
    """Ganze Zahl mit Punkt als Tausendertrenner."""
    return f"{int(round(n)):,}".replace(",", ".")


def pct(x: float, nd: int = 1) -> str:
    """Anteil als Prozentzahl mit Komma und Leerzeichen vor dem Prozentzeichen."""
    return f"{de(100.0 * x, nd)} %"


def ms_mean(c: dict, key: str, nd: int = 3) -> str:
    return de(c[key]["mean"], nd)


def ms_pm(c: dict, key: str, nd: int = 3) -> str:
    """Mittel ± Standardfehler der Blöcke."""
    return f"{de(c[key]['mean'], nd)} ± {de(c[key]['se'], nd)}"


def differs(a: dict, b: dict, key: str) -> int:
    """+1, wenn das Mittel von a über dem von b liegt (Abstand über der Marge), −1 darunter, sonst 0."""
    tol = max(Z * math.hypot(a[key]["se"], b[key]["se"]), FLOOR)
    d = a[key]["mean"] - b[key]["mean"]
    return 1 if d > tol else -1 if d < -tol else 0


def talk_label(talk: str, K: int) -> str:
    """Kurzname einer Redeart für Tabellen: „kein Reden“, „Zeile spricht, K = 2“, „beide sprechen, K = 2“."""
    return "kein Reden" if talk == G.TALK_NONE else f"{talk}, K = {K}"


# --- Hauptansicht Reden ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MainView:
    settings: S.Settings
    runs_used: int                   # Läufe, mit denen tatsächlich gerechnet wurde (Budget); 0: nicht gelernt
    steps: np.ndarray                # Stützstellen
    snaps: list                      # Momentaufnahmen (L.Snapshot) an den Stützstellen

    @property
    def last(self) -> L.Snapshot:
        return self.snaps[-1]


def analyse(s: S.Settings) -> MainView:
    """Ein Lernlauf mit höchstens BU.used_runs(s) Läufen und die Kennzahlen daraus (bei beidseitigem Reden mit K = 3 nicht gelernt)."""
    used = BU.used_runs(s)
    if used == 0:
        return MainView(s, 0, np.zeros(0, dtype=int), [])
    ext = S.ext_of(s)
    checks = XP.checkpoints_of(s.steps, C.TRACE_POINTS)
    snaps = L.population_run(ext, s.beta, s.n_agents, s.steps, used, s.seed, s.start, checks)
    return MainView(s, used, np.array([x.step for x in snaps]), snaps)


@dataclass(frozen=True)
class LewisView:
    settings: S.Settings
    runs: int
    best: float
    steps: np.ndarray
    snaps: list                      # LW.LewisSnapshot an den Stützstellen

    @property
    def last(self) -> LW.LewisSnapshot:
        return self.snaps[-1]

    @property
    def reached(self) -> np.ndarray:
        """Läufe, deren Effizienz mindestens 95 % des Bestwerts min(1, ns/n) erreicht."""
        return self.last.efficiency >= 0.95 * self.best

    def cross(self) -> tuple[float | None, int]:
        """Fremdspiel (Sprecher aus Lauf r gegen Hörer aus Lauf r + 1) über die Paare, in denen beide Läufe den Bestwert erreicht haben, und Zahl dieser Paare."""
        both = self.reached & np.roll(self.reached, -1)
        if not both.any():
            return None, 0
        return float(LW.cross_efficiency(self.last.ps, self.last.pr)[both].mean()), int(both.sum())


def analyse_lewis(s: S.Settings) -> LewisView:
    checks = XP.checkpoints_of(s.lewis_rounds, C.TRACE_POINTS)
    snaps = LW.roth_erev(s.lewis_n, s.lewis_ns, s.lewis_rounds, s.runs, s.seed, checks)
    return LewisView(s, s.runs, LW.best_efficiency(s.lewis_n, s.lewis_ns), np.array([x.round for x in snaps]), snaps)


# --- Verdikte der Hauptansicht ------------------------------------------------------------------------------------------------------------------
def verdict_main(view: MainView) -> tuple[str, str]:
    """Zahlen des Lernlaufs am Ende: Anteile der Paarungen, Auszahlungen, Information; Zahl der Läufe, die mindestens zur Hälfte in (A, A) stehen."""
    s = view.settings
    if view.runs_used == 0:
        return "info", ("Beidseitiges Reden mit K = 3 hat 1.536 Strategien je Seite (2,4 Millionen Paare): hier werden nur die Gleichgewichte gezählt (Tabelle oben), nicht gelernt. "
                        "Wähle K = 2 oder einseitiges Reden, um zu lernen.")
    last = view.last
    exits = int((last.aa >= XP.EXIT_LEVEL).sum())
    start_text = "Start in der Altkonvention „alle B“" if s.start == C.START_OLD else "zufälliger Start"
    return "info", (f"Nach {thousands(s.steps)} Schritten ({start_text}, β = {de(s.beta, 0)}, N = {s.n_agents}, Mittel über {view.runs_used} Läufe): (A, A) {pct(float(last.aa.mean()), 1)}, "
                    f"(B, B) {pct(float(last.bb.mean()), 1)}, Fehlkoordination {pct(float(last.mis.mean()), 1)}; Auszahlung Zeile {de(float(last.payoff1.mean()), 3)}, Spalte {de(float(last.payoff2.mean()), 3)}; "
                    f"Information der Nachricht {de(float(last.info.mean()), 3)} Bit. In {exits} von {view.runs_used} Läufen sind mindestens {pct(XP.EXIT_LEVEL, 0)} der Paarungen (A, A).")


def verdict_lewis_live(view: LewisView) -> tuple[str, str]:
    """Zahlen des Lewis-Laufs am Ende: mittlere Effizienz, Läufe am Bestwert, Fremdspiel mit der Zahl der Paare, Zufallswert 1/n."""
    s = view.settings
    cross, pairs = view.cross()
    reached = int(view.reached.sum())
    cross_text = "nicht definiert (kein Paar, in dem beide Läufe den Bestwert erreicht haben)" if cross is None else f"{de(cross, 3)} aus {pairs} {'Paar' if pairs == 1 else 'Paaren'}"
    best_text = "" if s.lewis_ns >= s.lewis_n else f" Mit ns = {s.lewis_ns} < n = {s.lewis_n} Signalen können höchstens {s.lewis_ns} Zustände unterschieden werden, der Bestwert ist ns/n."
    return "info", (f"Nach {thousands(s.lewis_rounds)} Runden (n = {s.lewis_n} Zustände, ns = {s.lewis_ns} Signale, {view.runs} Läufe) erreichen {reached} von {view.runs} Läufen mindestens 95 % des Bestwerts "
                    f"{de(view.best, 3)}; mittlere Effizienz {de(float(view.last.efficiency.mean()), 3)} (Start 1/n = {de(1.0 / s.lewis_n, 3)}). Fremdspiel: {cross_text}; Zufallswert 1/n = {de(1.0 / s.lewis_n, 3)}.{best_text}")



# --- Exakte Tabellen: Gleichgewichte (E1, E2, E5) ----------------------------------------------------------------------------------------------
def num(x: float, nd: int = 3) -> str:
    """Zahl ohne überflüssige Nullen: 9 → „9“, 7,875 → „7,875“, 0,6666… → „0,667“."""
    text = de(x, nd)
    return text.rstrip("0").rstrip(",") if "," in text else text


def pairs_text(pairs) -> str:
    """Auszahlungspaare als „(9; 9), (7; 7)“, absteigend nach der Auszahlung der Zeile."""
    if pairs is None:
        return "nicht gerechnet"
    return ", ".join(f"({num(a)}; {num(b)})" for a, b in sorted(pairs, reverse=True))


def informative_text(row: dict) -> str:
    """„ja (Reaktionsdifferenz 0,778)“, „nein“ oder warum nicht gerechnet."""
    if row["talk"] == G.TALK_NONE:
        return "–"
    if row["talk"] == G.TALK_ONE and row["K"] == 1:
        return "nein (eine Nachricht)"
    if row["informative"] is None:
        return "nicht gerechnet"
    return f"ja (Reaktionsdifferenz {num(row['difference'])})" if row["informative"] else "nein"


def equilibrium_table(rows: list[dict]) -> list[dict]:
    """Tabelle der Gleichgewichte: Strategien, Zahl der reinen Gleichgewichte, Auszahlungspaare (reine; alle, auch gemischte), informativ ja/nein."""
    out = []
    for r in rows:
        out.append({"Reden": talk_label(r["talk"], r["K"]), "Strategien (Zeile × Spalte)": f"{thousands(r['strategies'][0])} × {thousands(r['strategies'][1])}", "Reine Gleichgewichte": thousands(r["pure"]),
                    "Auszahlungspaare der reinen": pairs_text(r["pure_payoffs"]), "Auszahlungspaare aller Gleichgewichte": pairs_text(r["all_payoffs"]) if r["all_payoffs"] is not None else "nicht gerechnet",
                    "Informatives Gleichgewicht": informative_text(r)})
    return out


def same_points(a, b) -> bool:
    """Gleiche Mengen von Auszahlungspaaren (auf neun Stellen)?"""
    return a is not None and b is not None and {(round(x, 9), round(y, 9)) for x, y in a} == {(round(x, 9), round(y, 9)) for x, y in b}


def verdict_equilibria(row_none: dict, row_talk: dict) -> tuple[str, str]:
    """Vergleich des gewählten erweiterten Spiels mit dem Spiel ohne Reden: reine Gleichgewichte, Auszahlungspaare, informativ."""
    if row_talk["talk"] == G.TALK_NONE:
        return "info", (f"Ohne Reden hat das Spiel {row_none['pure']} reine Gleichgewichte mit den Auszahlungspaaren {pairs_text(row_none['pure_payoffs'])}; "
                        f"alle Gleichgewichte, auch gemischte: {pairs_text(row_none['all_payoffs'])}. Wähle in der Seitenleiste eine Art des Redens, um sie zu vergleichen.")
    same_pure = same_points(row_none["pure_payoffs"], row_talk["pure_payoffs"])
    text = (f"Im erweiterten Spiel ({talk_label(row_talk['talk'], row_talk['K'])}) gibt es {thousands(row_talk['pure'])} statt {row_none['pure']} reine Gleichgewichte; die Auszahlungspaare der reinen Gleichgewichte sind "
            f"{'dieselben' if same_pure else 'andere'}: {pairs_text(row_talk['pure_payoffs'])}")
    if row_talk["all_payoffs"] is not None:
        text += f". Alle Gleichgewichte (auch gemischte) haben {'dieselben' if same_points(row_none['all_payoffs'], row_talk['all_payoffs']) else 'andere'} Auszahlungspaare wie ohne Reden"
    text += f". Fehlkoordination in einem reinen Gleichgewicht: {'ja' if row_talk['miscoordination'] else 'nein'}. Informatives Gleichgewicht: {informative_text(row_talk)}."
    if row_talk["lottery"] is not None:
        lot = row_talk["lottery"]
        text += f" Lotterie (gleiche Nachricht → A, sonst B): {'ein Gleichgewicht' if lot['equilibrium'] else 'kein Gleichgewicht'} mit den Auszahlungen ({num(lot['payoff1'])}; {num(lot['payoff2'])})."
    return "info", text


E1_GAMES = (("Reine Koordination", G.GAME_COORD, 0.0), ("Hirschjagd", G.GAME_HIRSCH, 0.0), ("Hirschjagd mit Verlockung", G.GAME_HIRSCH, 1.0), ("Kampf der Geschlechter", G.GAME_BOS, 0.0))
E1_TALKS = ((G.TALK_NONE, 0), (G.TALK_ONE, 1), (G.TALK_ONE, 2), (G.TALK_ONE, 3), (G.TALK_TWO, 2))


def e1_rows(b: float, t: float) -> list[dict]:
    """Zeilen für E1: vier Spiele (reine Koordination, Hirschjagd mit t = 0, Hirschjagd mit Verlockung t, Kampf der Geschlechter) × fünf Arten des Redens; die Verlockung ist t, wenn t > 0, sonst 1."""
    lure = t if t > 0 else 1.0
    out = []
    for label, game, tt in E1_GAMES:
        tt = lure if tt > 0 else 0.0
        for talk, K in E1_TALKS:
            row = XP.equilibrium_row(game, b, tt, talk, K)
            row["label"] = label if game != G.GAME_HIRSCH or tt == 0 else f"{label} t = {num(tt)}"
            row["t"] = tt
            out.append(row)
    return out


def e1_table(rows: list[dict], only_game: str | None = None) -> list[dict]:
    """Tabelle zu E1: je Zeile Spiel und Art des Redens."""
    out = []
    for r, t in zip(rows, equilibrium_table(rows)):
        if only_game is not None and r["game"] != only_game:
            continue
        out.append({"Spiel": r["label"], **t})
    return out


def two_sided_three(game: str, b: float, t: float) -> dict:
    """Beidseitiges Reden mit K = 3 (1 536 Strategien je Seite): Zahl der reinen Gleichgewichte und ihre Auszahlungspaare."""
    ext = G.build(game, b, t, G.TALK_TWO, 3)
    count, pairs = E.pure_ne_summary(ext)
    return {"strategies": ext.S1, "pure": count, "pure_payoffs": pairs}


def verdict_e1(rows: list[dict]) -> tuple[str, str]:
    """Zählt über die Zellen: in wie vielen Zellen die Auszahlungspaare aller Gleichgewichte mit Reden (einseitig K ≤ 2) gleich sind, die der reinen in allen Arten des Redens, und wo ein informatives existiert."""
    base = {r["label"]: r for r in rows if r["talk"] == G.TALK_NONE}
    exact = [r for r in rows if r["all_payoffs"] is not None and r["talk"] != G.TALK_NONE]
    same_all = [r for r in exact if same_points(base[r["label"]]["all_payoffs"], r["all_payoffs"])]
    pure_rows = [r for r in rows if r["talk"] != G.TALK_NONE]
    same_pure = [r for r in pure_rows if same_points(base[r["label"]]["pure_payoffs"], r["pure_payoffs"])]
    informative = sorted({r["label"] for r in rows if r["informative"]})
    none_inf = sorted({r["label"] for r in rows if r["talk"] == G.TALK_ONE and r["K"] == 2 and not r["informative"]})
    miscoord = sum(1 for r in rows if r["miscoordination"])
    return "success", (f"Die Auszahlungspaare aller Gleichgewichte (auch gemischter) sind mit Reden dieselben wie ohne in {len(same_all)} von {len(exact)} Zellen (Spiele × einseitig K = 1 und 2), "
                       f"die der reinen Gleichgewichte in {len(same_pure)} von {len(pure_rows)} Zellen (alle Arten des Redens); reine Gleichgewichte mit Fehlkoordination: {miscoord}. "
                       f"Die Zahl der reinen Gleichgewichte ist dagegen mit einseitigem K = 2 und 3 und mit beidseitigem Reden größer als ohne Reden. Ein informatives Gleichgewicht (einseitig, K = 2) gibt es bei: {', '.join(informative) or 'keinem Spiel'}; "
                       f"keines bei: {', '.join(none_inf) or 'keinem Spiel'}.")


def e2_rows(b: float, talk_k: tuple[str, int]) -> list[dict]:
    """E2: die Hirschjagd mit Hase-Auszahlung b und Verlockung t aus C.E2_TS bei der gegebenen Redeart (einseitig, K = 2)."""
    out = []
    for t in C.E2_TS:
        row = XP.equilibrium_row(G.GAME_HIRSCH, b, t, talk_k[0], talk_k[1])
        row["t"] = t
        row["b"] = b
        row["pstar"] = G.threshold(b, t)
        out.append(row)
    return out


def e2_table(rows: list[dict]) -> list[dict]:
    return [{"Verlockung t": num(r["t"], 2), "Schwelle p* = b / (9 − t)": num(r["pstar"]), "Auszahlungspaare aller Gleichgewichte": pairs_text(r["all_payoffs"]),
             "Informatives Gleichgewicht": informative_text(r)} for r in rows]


def verdict_e2(rows: list[dict]) -> tuple[str, str]:
    """Zählt die Verlockungen mit informativem Gleichgewicht; nennt die Verlockung 0 und die mit t > 0 im selben Satz."""
    inf = [r["t"] for r in rows if r["informative"]]
    non = [r["t"] for r in rows if not r["informative"]]
    base = rows[0]
    return "success", (f"Bei b = {num(base['b'], 1)} gibt es ein informatives Gleichgewicht in {len(inf)} von {len(rows)} Verlockungen (t = {', '.join(num(t, 2) for t in inf) or '–'}); "
                       f"keines bei t = {', '.join(num(t, 2) for t in non) or '–'}. Ohne Verlockung ist die Reaktionsdifferenz der Spalte {num(base['difference'])} (die Spalte spielt nach der einen Nachricht A mit "
                       f"Wahrscheinlichkeit p* = {num(G.threshold(base['b']))}, nach der anderen B); mit t > 0 hätte die Zeile mehr davon, die Nachricht zu senden, auf die A folgt, und B zu spielen.")


def e5_rows(b: float, t: float) -> list[dict]:
    """E5: je Spiel (reine Koordination, Hirschjagd t = 0 und t = 1, Kampf der Geschlechter) die Lotterie bei beidseitigem Reden mit K = 2 und die Gleichgewichte ohne Reden (b, t wie bei E1)."""
    lure = t if t > 0 else 1.0
    out = []
    for label, game, tt in E1_GAMES:
        tt = lure if tt > 0 else 0.0
        two = XP.equilibrium_row(game, b, tt, G.TALK_TWO, 2)
        none = XP.equilibrium_row(game, b, tt, G.TALK_NONE, 0)
        out.append({"label": label, "game": game, "t": tt, "lottery": two["lottery"], "none_all": none["all_payoffs"], "none_pure": none["pure_payoffs"]})
    return out


def e5_table(rows: list[dict]) -> list[dict]:
    return [{"Spiel": r["label"], "Lotterie ist Gleichgewicht": "ja" if r["lottery"]["equilibrium"] else "nein", "Auszahlung der Lotterie (Zeile; Spalte)": f"({num(r['lottery']['payoff1'])}; {num(r['lottery']['payoff2'])})",
             "reine Gleichgewichte ohne Reden": pairs_text(r["none_pure"]), "alle Gleichgewichte ohne Reden": pairs_text(r["none_all"])} for r in rows]


def verdict_e5(rows: list[dict], meas: dict | None, mech: dict | None = None) -> tuple[str, str]:
    """Zählt über die vier Spiele: Lotterie ist Gleichgewicht; Summe beider Auszahlungen größer als beim schlechtesten, gleich dem besten Gleichgewicht ohne Reden."""
    ok = [r for r in rows if r["lottery"]["equilibrium"]]
    above, equal = [], []
    for r in ok:
        lot = r["lottery"]["payoff1"] + r["lottery"]["payoff2"]
        sums = [a + c for a, c in r["none_all"]]
        if lot > min(sums) + 1e-9:
            above.append(r["label"])
        if abs(lot - max(sums)) < 1e-9:
            equal.append(r["label"])
    text = (f"Die Lotterie ist in {len(ok)} von {len(rows)} Spielen ein Gleichgewicht. Die Summe beider Auszahlungen liegt über der des schlechtesten Gleichgewichts ohne Reden in {len(above)} von {len(rows)} Spielen "
            f"und ist so groß wie die des besten in {len(equal)} von {len(rows)} ({', '.join(equal) or '–'}).")
    if mech is not None:
        lot = mech["lottery"]
        text += (f" Gelernt wird sie nicht: im Kampf der Geschlechter (beidseitig K = 2, zufälliger Start, β = {de(lot['beta'], 0)}, N = {lot['n_agents']}, {thousands(lot['steps'])} Schritte) spielen am Ende im Mittel "
                 f"{pct(lot['mean_share'], 1)} der Akteure die Lotterie-Regel, bei gleichverteilter Wahl wären es {pct(lot['uniform_share'], 1)}; in {lot['runs_at_least_half']} von {lot['runs']} Läufen ist es die Hälfte oder mehr.")
    return "info", text



# --- Live-Experimente mit den Reglern der Seitenleiste ------------------------------------------------------------------------------------------
def _live_runs(s: S.Settings) -> int:
    """Läufe der Live-Zellen: so viele, wie das Budget für das teuerste Reden (beidseitig, K = 2) zulässt."""
    return BU.used_runs(S.with_(s, talk=G.TALK_TWO, K=2))


def e3_live(s: S.Settings) -> dict:
    """E3 mit den Reglern: die fünf Arten des Redens in der Hirschjagd (b, t, β, N, Schritte aus den Einstellungen), Start in der Altkonvention, ein Block."""
    runs = _live_runs(s)
    out = []
    for talk, K in C.E3_TALKS:
        si = S.with_(s, game=G.GAME_HIRSCH, talk=talk, K=K, start=C.START_OLD, runs=runs)
        out.append({"talk": talk, "K": K, "result": XP.cell(si, 1, f"e3live|{s.seed}|{talk}|{K}")})
    return {"runs": runs, "rows": out}


def e3_live_table(live: dict) -> list[dict]:
    return [{"Reden": talk_label(r["talk"], r["K"]), "(A, A)": pct(r["result"]["aa"]["mean"], 1), "(B, B)": pct(r["result"]["bb"]["mean"], 1),
             "Läufe mit mindestens der Hälfte (A, A)": pct(r["result"]["exit"]["mean"], 1), "Information (Bit)": de(r["result"]["info"]["mean"], 3)} for r in live["rows"]]


def e4_live(s: S.Settings) -> dict:
    """E4 mit den Reglern: die fünf Arten des Redens im Kampf der Geschlechter (β, N, Schritte, Start aus den Einstellungen), ein Block."""
    runs = _live_runs(s)
    out = []
    for talk, K in C.E3_TALKS:
        si = S.with_(s, game=G.GAME_BOS, talk=talk, K=K, runs=runs)
        out.append({"talk": talk, "K": K, "result": XP.cell(si, 1, f"e4live|{s.seed}|{talk}|{K}")})
    return {"runs": runs, "rows": out}


def e4_live_table(live: dict) -> list[dict]:
    return [{"Reden": talk_label(r["talk"], r["K"]), "Auszahlung Zeile": de(r["result"]["p1"]["mean"], 3), "Auszahlung Spalte": de(r["result"]["p2"]["mean"], 3),
             "Fehlkoordination": pct(r["result"]["mis"]["mean"], 1), "Information (Bit)": de(r["result"]["info"]["mean"], 3)} for r in live["rows"]]


def e5_live(s: S.Settings) -> dict:
    """E5 mit den Reglern: Lotterie im Spiel der Einstellungen exakt prüfen und ein Lernlauf mit beidseitigem Reden, K = 2."""
    b, t = S.effective_b(s), S.effective_t(s)
    row = XP.equilibrium_row(s.game, b, t, G.TALK_TWO, 2)
    none = XP.equilibrium_row(s.game, b, t, G.TALK_NONE, 0)
    si = S.with_(s, talk=G.TALK_TWO, K=2, runs=_live_runs(s))
    return {"lottery": row["lottery"], "none_all": none["all_payoffs"], "runs": si.runs, "learned": XP.cell(si, 1, f"e5live|{s.seed}")}


def e5_live_table(live: dict) -> list[dict]:
    lot, c = live["lottery"], live["learned"]
    return [{"Lotterie ist Gleichgewicht": "ja" if lot["equilibrium"] else "nein", "Auszahlung der Lotterie (Zeile; Spalte)": f"({num(lot['payoff1'])}; {num(lot['payoff2'])})",
             "alle Gleichgewichte ohne Reden": pairs_text(live["none_all"]), "gelernt: Auszahlung (Zeile; Spalte)": f"({de(c['p1']['mean'], 3)}; {de(c['p2']['mean'], 3)})",
             "gelernt: Information (Bit)": de(c["info"]["mean"], 3)}]


def e6_live(s: S.Settings) -> dict:
    """E6 mit den Reglern: Lewis-Läufe für n = 2 bis 6 mit ns = n (Läufe, Runden, Zufalls-Seed aus den Einstellungen), ein Block."""
    rows = [XP.lewis_cell(n, n, s.runs, 1, s.lewis_rounds, (s.lewis_rounds,), f"e6live|{s.seed}|{n}") for n in range(C.LEWIS_N_MIN, C.LEWIS_N_MAX + 1)]
    return {"rounds": s.lewis_rounds, "rows": rows}


def e6_live_table(live: dict) -> list[dict]:
    key = str(live["rounds"])
    return [{"Zustände n": r["n"], "Signale": r["ns"], "Effizienz": de(r[key]["eff"]["mean"], 3), "Läufe am Bestwert": pct(r[key]["share"]["mean"], 1),
             "Fremdspiel": "–" if r[key]["cross"]["mean"] is None else de(r[key]["cross"]["mean"], 3), "Zufallswert 1/n": de(1.0 / r["n"], 3)} for r in live["rows"]]


_LEWIS_NE_CACHE: dict = {}


def lewis_ne_table() -> list[dict]:
    """Gleichgewichte des Lewis-Spiels mit n = ns = 2, 3, 4 durch Vollaufzählung; die Zahl der Signalsysteme ist n!."""
    out = []
    for n in (2, 3, 4):
        if n not in _LEWIS_NE_CACHE:
            _LEWIS_NE_CACHE[n] = LW.pure_ne_lewis(n)
        r = _LEWIS_NE_CACHE[n]
        by = "; ".join(f"{thousands(c)} mit {num(k)}" for k, c in sorted(r["by_efficiency"].items(), reverse=True))
        out.append({"Zustände n = Signale": n, "reine Gleichgewichte": thousands(r["count"]), "Signalsysteme (Effizienz 1)": thousands(r["signalling_systems"]), "n!": thousands(math.factorial(n)), "nach Effizienz": by})
    return out



# --- Tabellen der vorgerechneten Messreihen -----------------------------------------------------------------------------------------------------
def pm_pct(c: dict, key: str, nd: int = 1) -> str:
    """Anteil als Prozentzahl ± Standardfehler der Blöcke."""
    return f"{pct(c[key]['mean'], nd)} ± {pct(c[key]['se'], nd)}"


def e3_cell(meas: dict, pstar: float, talk: str, K: int, beta: float) -> dict:
    return XP.find_cell(meas, family="e3", b=round(9 * pstar, 6), talk=talk, K=K, beta=beta)


def e3_table(meas: dict, beta: float) -> list[dict]:
    """E3: Anteil (A, A) nach 40.000 Schritten je Art des Redens (Zeilen) und Schwelle p* (Spalten); Start in der Altkonvention, N = 20."""
    return [{"Reden": talk_label(tk, k), **{f"p* = {de(p, 1)}": pm_pct(e3_cell(meas, p, tk, k, beta), "aa") for p in C.THRESHOLD_PSTARS}} for tk, k in C.E3_TALKS]


def mech_cells(meas: dict) -> list[tuple]:
    """Die einseitigen Zellen von E3 mit Mechanismus-Kennzahlen: (p*, K, β, Zelle)."""
    return [(p, k, beta, e3_cell(meas, p, G.TALK_ONE, k, beta)) for p in C.THRESHOLD_PSTARS for k in (2, 3) for beta in C.E3_BETAS]


def mech_table(meas: dict) -> list[dict]:
    """Mechanismus: bei den Läufen, die noch in der Altkonvention stehen, der größte Hörer-Anteil s_max, der Anteil der Zeitpunkte mit s_max > p*, die Zahl der Sprecher auf der seltensten Nachricht
    n_min, und aus der Formel die größte Zahl n von Sprechern auf einer Nachricht, bei der ein einzelner A-Sprecher den Hörern einen Anteil über p* verschafft."""
    out = []
    for p, k, beta, c in mech_cells(meas):
        out.append({"Schwelle p*": de(p, 1), "Nachrichten K": k, "Schärfe β": de(beta, 0), "s_max (noch drin)": ms_pm(c, "smax_pre"), "Zeitpunkte mit s_max > p*": pm_pct(c, "above_pre"),
                    "n_min (Sprecher auf der seltensten Nachricht)": de(c["nmin_pre"]["mean"], 1), "Formel: größtes n": XP.lone_sender_limit(beta, 20, p), "am Ende (A, A)": pm_pct(c, "aa")})
    return out


def hearer_table(beta: float, n_agents: int, pstar: float) -> list[dict]:
    """Hörer-Anteil nach der Formel, wenn n Sprecher eine Nachricht senden und genau einer von ihnen A spielt (die übrigen B)."""
    return [{"Sprecher auf der Nachricht n": n, "Hörer-Anteil mit A": de(XP.hearer_share(beta, n_agents, pstar, n), 3), "über p*": "ja" if XP.hearer_share(beta, n_agents, pstar, n) > pstar else "nein"} for n in (1, 2, 3, 4, 5, 8)]


def hs_cell(meas: dict, t: float, talk: str, K: int, beta: float, n_agents: int = 20) -> dict:
    fam = "hsn" if n_agents != 20 else "hs"
    return XP.find_cell(meas, family=fam, t=t, talk=talk, K=K, beta=beta, n_agents=n_agents)


def hs_table(meas: dict) -> list[dict]:
    """E3, Handschlag: Hirschjagd mit b = 7 (p* = 7/9), Start in der Altkonvention, 10.000 Schritte je Akteur; (A, A) am Ende und Anteil der Akteure mit Handschlag."""
    out = []
    for t in (0.0, 1.0):
        for tk, k in ((G.TALK_ONE, 1), (G.TALK_TWO, 2)):
            for beta in C.HANDSHAKE_BETAS:
                c = hs_cell(meas, t, tk, k, beta)
                out.append({"Verlockung t": num(t, 1), "Reden": talk_label(tk, k), "β": de(beta, 0), "N": 20, "Schritte": thousands(C.HANDSHAKE_STEPS_PER_AGENT * 20), "(A, A)": pm_pct(c, "aa"),
                            "Läufe mit mindestens der Hälfte (A, A)": pm_pct(c, "exit"), "Anteil Handschlag": pm_pct(c, "hs")})
    for n_agents in (10, 40):
        for beta in C.HANDSHAKE_BETAS:
            c = hs_cell(meas, 0.0, G.TALK_TWO, 2, beta, n_agents)
            out.append({"Verlockung t": num(0.0, 1), "Reden": talk_label(G.TALK_TWO, 2), "β": de(beta, 0), "N": n_agents, "Schritte": thousands(C.HANDSHAKE_STEPS_PER_AGENT * n_agents), "(A, A)": pm_pct(c, "aa"),
                        "Läufe mit mindestens der Hälfte (A, A)": pm_pct(c, "exit"), "Anteil Handschlag": pm_pct(c, "hs")})
    return out


def e4_table(meas: dict, start: str) -> list[dict]:
    """E4: Kampf der Geschlechter, Auszahlung der Zeile und der Spalte, Summe, Fehlkoordination, Information je Art des Redens."""
    out = []
    for tk, k in C.E3_TALKS:
        c = XP.find_cell(meas, family="bos", talk=tk, K=k, start=start)
        out.append({"Reden": talk_label(tk, k), "Auszahlung Zeile": ms_pm(c, "p1"), "Auszahlung Spalte": ms_pm(c, "p2"), "Summe": de(c["p1"]["mean"] + c["p2"]["mean"], 3), "Fehlkoordination": pm_pct(c, "mis"),
                    "Information (Bit)": ms_pm(c, "info")})
    return out


def rand_cell(meas: dict, game: str, t: float, talk: str, K: int, beta: float) -> dict:
    return XP.find_cell(meas, family="rand", game=game, t=t, talk=talk, K=K, beta=beta)


def rand_table(meas: dict, game: str, t: float) -> list[dict]:
    """Zufälliger Start in Hirschjagd oder reiner Koordination: (A, A), (B, B), Fehlkoordination, Effizienz (Summe der Auszahlungen durch den Bestwert) je Art des Redens und Schärfe."""
    out = []
    for beta in C.RANDOM_BETAS:
        for tk, k in C.E3_TALKS:
            c = rand_cell(meas, game, t, tk, k, beta)
            out.append({"β": de(beta, 0), "Reden": talk_label(tk, k), "(A, A)": pm_pct(c, "aa"), "(B, B)": pm_pct(c, "bb"), "Fehlkoordination": pm_pct(c, "mis"), "Effizienz": ms_pm(c, "eff")})
    return out


def e5_learned_table(meas: dict) -> list[dict]:
    """E5: gelernt im Kampf der Geschlechter bei beidseitigem Reden mit K = 2, zufälliger Start und Altkonvention, gegen das Spiel ohne Reden: Auszahlungen und Information."""
    out = []
    for start in C.STARTS:
        for tk, k in ((G.TALK_NONE, 0), (G.TALK_TWO, 2)):
            c = XP.find_cell(meas, family="bos", talk=tk, K=k, start=start)
            out.append({"Start": start, "Reden": talk_label(tk, k), "Auszahlung Zeile": ms_pm(c, "p1"), "Auszahlung Spalte": ms_pm(c, "p2"), "Fehlkoordination": pm_pct(c, "mis"), "Information (Bit)": ms_pm(c, "info")})
    return out


def e6_table(meas: dict, rounds: int) -> list[dict]:
    """E6: je (n, ns) Effizienz, Anteil der Läufe mit mindestens 95 % des Bestwerts, Bestwert, Fremdspiel (Läufe, in denen beide am Bestwert) und Zufallswert 1/n."""
    out = []
    for n, ns in C.LEWIS_PAIRS:
        c = XP.find_lewis(meas, n, ns)
        r = c[str(rounds)]
        out.append({"Zustände n": n, "Signale ns": ns, "Bestwert min(1, ns/n)": de(c["best"], 3), "Effizienz": ms_pm(r, "eff"), "Läufe am Bestwert": pm_pct(r, "share"),
                    "Fremdspiel": ms_pm(r, "cross") if r["cross"]["mean"] is not None else "–", "Zufallswert 1/n": de(1.0 / n, 3)})
    return out


# --- Ereignisstudie (H10) -----------------------------------------------------------------------------------------------------------------------
def mech_event_table(mech: dict) -> list[dict]:
    """Rückblick auf den Austritt (Ereignisstudie, jeder Schritt aufgezeichnet): mittlere Zahl der Sprecher n und mittlerer Hörer-Anteil s der Austrittsnachricht 200 / 100 / 50 / 20 Schritte vorher."""
    import vor_mechanism as M
    out = []
    for c in mech["cells"]:
        sm = M.summary(c)
        out.append({"Schwelle p*": de(c["pstar"], 1), "K": c["K"], "β": de(c["beta"], 0), "Austritte": f"{c['exits']} von {c['runs']}", "Sprecher je Nachricht, vor dem Austritt": de(c["mean_n_before"], 1),
                    "n: 200 / 100 / 50 / 20 Schritte vorher": " / ".join(de(sm[k]["n"], 1) for k in (200, 100, 50, 20)),
                    "Hörer-Anteil s: 200 / 100 / 50 / 20 Schritte vorher": " / ".join(de(sm[k]["s"], 2) for k in (200, 100, 50, 20)), "Formel: größtes n": c["limit"]})
    return out


def verdict_mechanism(meas: dict, mech: dict | None) -> tuple[str, str]:
    """H10: Die Vermutung „ungenutzte Nachrichten, Hörer-Anteil um ½“ gegen die Momentaufnahmen der 12 einseitigen Zellen und, wenn vorhanden, gegen die Ereignisstudie."""
    cells = mech_cells(meas)
    smax = [c["smax_pre"]["mean"] for _, _, _, c in cells if c["smax_pre"]["mean"] is not None]
    above = [c["above_pre"]["mean"] for _, _, _, c in cells if c["above_pre"]["mean"] is not None]
    nmin = [c["nmin_pre"]["mean"] for _, _, _, c in cells if c["nmin_pre"]["mean"] is not None]
    text = (f"Die Vermutung des Plans (ungenutzte Nachrichten lassen den Hörer-Anteil auf etwa ½ treiben) trifft nicht zu. Bei den noch nicht ausgetretenen Läufen liegt der größte Hörer-Anteil "
            f"in den {len(smax)} einseitigen Zellen im Mittel zwischen {de(min(smax), 3)} und {de(max(smax), 3)}, nie bei ½; an den 41 Stützstellen je Lauf liegt er in {sum(1 for a in above if a == 0.0)} von {len(above)} Zellen "
            f"nie über p* (größter Anteil der Zeitpunkte {pct(max(above), 1)}); die seltenste Nachricht hat im Mittel {de(min(nmin), 1)} bis {de(max(nmin), 1)} Sprecher, ungenutzt ist sie nicht.")
    if mech is not None:
        import vor_mechanism as M
        sums = [M.summary(c) for c in mech["cells"]]
        n100 = [s[100]["n"] for s in sums]
        n_base = [c["mean_n_before"] for c in mech["cells"]]
        s200 = [s[200]["s"] for s in sums]
        s50 = [s[50]["s"] for s in sums]
        s20 = [s[20]["s"] for s in sums]
        a20 = [s[20]["a"] for s in sums]
        text += (f" Die Ereignisstudie (jeder Schritt aufgezeichnet, {sum(c['exits'] for c in mech['cells'])} Austritte in {len(mech['cells'])} Zellen) zeigt im Rückblick: Die Nachricht, über die der Austritt läuft, wird vorher selten gesendet "
                 f"(100 Schritte vor dem Austritt von im Mittel {de(min(n100), 1)} bis {de(max(n100), 1)} Sprechern, gegenüber {de(min(n_base), 1)} bis {de(max(n_base), 1)} im Mittel über alle Nachrichten), "
                 f"der Hörer-Anteil auf sie steigt von {de(min(s200), 2)} bis {de(max(s200), 2)} (200 Schritte vorher) über {de(min(s50), 2)} bis {de(max(s50), 2)} (50 Schritte) auf {de(min(s20), 2)} bis {de(max(s20), 2)} (20 Schritte vorher), "
                 f"zusammen mit den ersten A-Sprechern auf ihr (20 Schritte vorher im Mittel {de(min(a20), 1)} bis {de(max(a20), 1)}).")
    return "warning", text


def verdict_e3(meas: dict, beta: float) -> tuple[str, str]:
    """Zählt über die 12 Zellen (vier Arten des Redens × drei Schwellen) für die gewählte Schärfe, wo (A, A) über dem Spiel ohne Reden liegt, und über die Schärfen, wo höheres β den Austritt senkt oder hebt."""
    higher, lines = 0, []
    for tk, k in C.E3_TALKS[1:]:
        ps = [p for p in C.THRESHOLD_PSTARS if differs(e3_cell(meas, p, tk, k, beta), e3_cell(meas, p, G.TALK_NONE, 0, beta), "aa") == 1]
        higher += len(ps)
        lines.append(f"{talk_label(tk, k)}: {', '.join(de(p, 1) for p in ps) or 'keine'}")
    none_zero = sum(1 for p in C.THRESHOLD_PSTARS if e3_cell(meas, p, G.TALK_NONE, 0, beta)["aa"]["mean"] == 0.0)
    cmp = {1: [], -1: [], 0: []}
    for tk, k in C.E3_TALKS[1:]:
        for p in C.THRESHOLD_PSTARS:
            lo, hi = (e3_cell(meas, p, tk, k, x) for x in C.E3_BETAS)
            cmp[differs(hi, lo, "aa")].append(f"{talk_label(tk, k)} bei p* = {de(p, 1)}")
    return "success", (f"Ohne Reden bleibt die Altkonvention in {none_zero} von {len(C.THRESHOLD_PSTARS)} Schwellen vollständig stehen ((A, A) = 0,0 % bei β = {de(beta, 0)}). Mit Reden liegt (A, A) über dem Spiel ohne Reden "
                       f"in {higher} von {4 * len(C.THRESHOLD_PSTARS)} Zellen (Art des Redens × Schwelle; über vier Standardfehlern und {pct(FLOOR, 0)}-Punkten); nach Art des Redens und Schwellen p*: {'; '.join(lines)}. "
                       f"Von β = {de(C.E3_BETAS[0], 0)} auf β = {de(C.E3_BETAS[1], 0)} sinkt (A, A) in {len(cmp[-1])}, steigt in {len(cmp[1])} ({', '.join(cmp[1]) or 'keine'}) und bleibt gleich in {len(cmp[0])} dieser 12 Zellen.")


def verdict_handshake(meas: dict) -> tuple[str, str]:
    """H11 und Handschlag: (A, A) bei p* = 7/9 mit und ohne Verlockung, die Kontrolle ohne Inhalt, und die Abhängigkeit von N bei gleichen Schritten je Akteur."""
    two = {(t, beta): hs_cell(meas, t, G.TALK_TWO, 2, beta) for t in (0.0, 1.0) for beta in C.HANDSHAKE_BETAS}
    ctrl = [hs_cell(meas, t, G.TALK_ONE, 1, beta) for t in (0.0, 1.0) for beta in C.HANDSHAKE_BETAS]
    ctrl_zero = sum(1 for c in ctrl if c["aa"]["mean"] == 0.0)
    byn = {beta: [hs_cell(meas, 0.0, G.TALK_TWO, 2, beta, n)["aa"]["mean"] for n in C.HANDSHAKE_NS] for beta in C.HANDSHAKE_BETAS}
    falling = sum(1 for v in byn.values() if all(a > b for a, b in zip(v, v[1:])))
    return "success", (f"Der Handschlag tritt auf: bei p* = 7/9, N = 20 und beidseitigem Reden liegt (A, A) nach 200.000 Schritten bei {pct(two[(0.0, 40.0)]['aa']['mean'], 1)} (β = 40) und {pct(two[(0.0, 100.0)]['aa']['mean'], 1)} (β = 100); "
                       f"mit Verlockung t = 1 bei {pct(two[(1.0, 40.0)]['aa']['mean'], 1)} und {pct(two[(1.0, 100.0)]['aa']['mean'], 1)}. Die Kontrolle (einseitig, K = 1) bleibt in {ctrl_zero} von {len(ctrl)} Zellen bei 0,0 %. "
                       f"Mit der Zahl der Akteure N = {' / '.join(str(n) for n in C.HANDSHAKE_NS)} (Schritte je Akteur gleich) fällt (A, A) bei β = 40 von {pct(byn[40.0][0], 1)} über {pct(byn[40.0][1], 1)} auf {pct(byn[40.0][2], 1)} "
                       f"und bei β = 100 von {pct(byn[100.0][0], 1)} über {pct(byn[100.0][1], 1)} auf {pct(byn[100.0][2], 1)} (in {falling} von {len(byn)} Schärfen streng fallend). Eine eigene Mutationsrate gibt es im Modell nicht: das Rauschen ist 1/β.")


def verdict_e4(meas: dict) -> tuple[str, str]:
    """Kampf der Geschlechter: wer spricht, gewinnt; die Summe der Auszahlungen bleibt. Zufälliger Start und Altkonvention, β = 20."""
    out = []
    for start in C.STARTS:
        none = XP.find_cell(meas, family="bos", talk=G.TALK_NONE, K=0, start=start)
        parts = []
        for tk, k in C.E3_TALKS[1:]:
            c = XP.find_cell(meas, family="bos", talk=tk, K=k, start=start)
            parts.append(f"{talk_label(tk, k)} {de(c['p1']['mean'], 3)} / {de(c['p2']['mean'], 3)}")
        out.append(f"Start {start}: ohne Reden {de(none['p1']['mean'], 3)} / {de(none['p2']['mean'], 3)} (Zeile / Spalte); {'; '.join(parts)}")
    sums = [XP.find_cell(meas, family="bos", talk=tk, K=k, start=s) for tk, k in C.E3_TALKS for s in C.STARTS]
    spread = max(abs(c["p1"]["mean"] + c["p2"]["mean"] - 3.0) for c in sums)
    gain = sum(1 for s in C.STARTS for tk, k in C.E3_TALKS[2:4]
               if differs(XP.find_cell(meas, family="bos", talk=tk, K=k, start=s), XP.find_cell(meas, family="bos", talk=G.TALK_NONE, K=0, start=s), "p1") == 1)
    return "success", (f"Die Zeile gewinnt, wenn sie spricht: Auszahlung der Zeile höher als ohne Reden in {gain} von 4 Zellen (einseitig K = 2 und 3 × zwei Starts). {' | '.join(out)}. "
                       f"Die Summe beider Auszahlungen weicht in allen {len(sums)} Zellen höchstens {de(spread, 3)} von 3 ab; die Kontrolle K = 1 gleicht dem Spiel ohne Reden.")


def verdict_random(meas: dict) -> tuple[str, str]:
    """Zufälliger Start: Hirsch/Hirsch löst sich durch Reden nicht, Fehlkoordination wächst bei kleinem β mit der Zahl der Strategien."""
    aa = [(g, t, tk, k, beta, rand_cell(meas, g, t, tk, k, beta)["aa"]["mean"]) for g, t in ((G.GAME_HIRSCH, 0.0), (G.GAME_HIRSCH, 1.0)) for tk, k in C.E3_TALKS for beta in C.RANDOM_BETAS]
    worst = max(aa, key=lambda x: x[5])
    low = sum(1 for x in aa if x[5] <= 0.03)
    mis = [rand_cell(meas, G.GAME_HIRSCH, 0.0, tk, k, 6.0)["mis"]["mean"] for tk, k in C.E3_TALKS]
    eff = [rand_cell(meas, G.GAME_COORD, 0.0, tk, k, 6.0)["eff"]["mean"] for tk, k in C.E3_TALKS]
    return "info", (f"Reden löst die Hirschjagd bei zufälligem Start nicht: (A, A) liegt in {low} von {len(aa)} Zellen (Hirschjagd mit t = 0 und t = 1 × fünf Arten des Redens × zwei Schärfen) bei höchstens 3 %, "
                    f"der größte Wert ist {pct(worst[5], 1)} ({talk_label(worst[2], worst[3])}, β = {de(worst[4], 0)}). Bei β = 6 wächst die Fehlkoordination in der Hirschjagd mit der Zahl der Strategien: kein Reden / K = 1 / einseitig K = 2 / "
                    f"einseitig K = 3 / beidseitig K = 2 {' / '.join(pct(x, 1) for x in mis)}; in der reinen Koordination sinkt die Effizienz entsprechend {' / '.join(de(x, 3) for x in eff)}. "
                    f"Der Vergleich bei gleichem β benachteiligt das Spiel mit Reden, weil es mehr gleichwertige Strategien hat.")


def verdict_e6(meas: dict) -> tuple[str, str]:
    """Lewis-Spiel: Anteil der Läufe am Bestwert nach 100.000 Runden für n = ns = 2, 3, 4, 6; Überschuss an Signalen; Fremdspiel gegen 1/n."""
    cp = str(max(C.LEWIS_CHECKS))
    share = {n: XP.find_lewis(meas, n, n)[cp]["share"]["mean"] for n in (2, 3, 4, 6)}
    falling = all(a > b for a, b in zip(list(share.values()), list(share.values())[1:]))
    heal = [(3, 4), (3, 6), (4, 8), (2, 3)]
    healed = {f"n = {n}, ns = {ns}": XP.find_lewis(meas, n, ns)[cp]["share"]["mean"] for n, ns in heal}
    cross_ok, cross_all, exceptions = 0, 0, []
    for n, ns in C.LEWIS_PAIRS:
        r = XP.find_lewis(meas, n, ns)[cp]["cross"]
        if r["mean"] is None:
            continue
        cross_all += 1
        if abs(r["mean"] - 1.0 / n) <= 2 * r["se"] + 0.005:
            cross_ok += 1
        else:
            exceptions.append(f"n = {n}, ns = {ns}: {de(r['mean'], 3)} gegen {de(1.0 / n, 3)}")
    return "success", (f"Nach {thousands(int(cp))} Runden erreichen bei n = ns = 2 / 3 / 4 / 6 {' / '.join(pct(v, 1) for v in share.values())} der Läufe den Bestwert (streng fallend: {'ja' if falling else 'nein'}). "
                       f"Ein Überschuss an Signalen hilft: {', '.join(f'{k} {pct(v, 1)}' for k, v in healed.items())}. Das Fremdspiel liegt in {cross_ok} von {cross_all} Zellen innerhalb von zwei Standardfehlern "
                       f"(plus 0,005) um 1/n; Ausnahmen: {'; '.join(exceptions) or 'keine'}.")
