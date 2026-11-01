"""Gleichgewichte des (erweiterten) Spiels: reine per Aufzählung, alle (auch gemischte) bei höchstens zwei Nachrichten und einseitigem Reden per Trägerpaar-Rechnung in exakter Bruchrechnung
(ein kleines Simplex-Verfahren mit Bland-Regel, kein scipy zur Laufzeit), dazu die Lotterie des beidseitigen Redens (DEMO-PLAYBOOK §1b: eine Einheit je Regel).

**Trägerpaare.** Für Teilmengen S_x der Zeilenstrategien und S_y der Spaltenstrategien, alle nichtleer, gilt: Das Paar (x, y) mit Träger von x in S_x und von y in S_y, in dem alle Strategien aus S_x
gegen y gleich gut und mindestens so gut wie jede andere Zeilenstrategie sind (ebenso für S_y gegen x), ist ein Gleichgewicht, und jedes Gleichgewicht entsteht so aus seinem Trägerpaar. Die Bedingungen für
y (Zeile ist zufrieden) und für x (Spalte ist zufrieden) hängen bei festem Trägerpaar nicht voneinander ab; beide sind lineare Systeme. Je Trägerpaar werden Wertebereich der Auszahlungen beider Seiten
(Minimum und Maximum je Seite) bestimmt; die Gleichgewichtsauszahlungen sind die Vereinigung dieser Rechtecke über alle Trägerpaare.

**Informatives Gleichgewicht** (einseitig, zwei Nachrichten): ein Gleichgewicht, in dem die Zeile beide Nachrichten mit positiver Wahrscheinlichkeit sendet und die Spalte auf sie verschieden reagiert;
die Reaktionsdifferenz ist |P(Spalte spielt A | Nachricht 1) − P(Spalte spielt A | Nachricht 2)|, ihr größter Wert über alle solchen Gleichgewichte."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

import vor_games as G

ZERO, ONE = Fraction(0), Fraction(1)


# --------------------------------------------------------------------------------- reine Gleichgewichte
def pure_ne(U1: np.ndarray, U2: np.ndarray, tol: float = 1e-12) -> list[tuple[int, int]]:
    """Alle reinen Gleichgewichte (i, j): Zeile i ist beste Antwort auf Spalte j und Spalte j beste Antwort auf Zeile i (Gleichstände zählen mit)."""
    best_row = U1.max(axis=0, keepdims=True)
    best_col = U2.max(axis=1, keepdims=True)
    I, J = np.nonzero((U1 >= best_row - tol) & (U2 >= best_col - tol))
    return list(zip(I.tolist(), J.tolist()))


def pure_ne_count(U1: np.ndarray, U2: np.ndarray, tol: float = 1e-12) -> int:
    """Zahl der reinen Gleichgewichte ohne die Liste (für große Matrizen)."""
    best_row = U1.max(axis=0, keepdims=True)
    best_col = U2.max(axis=1, keepdims=True)
    return int(np.count_nonzero((U1 >= best_row - tol) & (U2 >= best_col - tol)))


def is_ne_mixed(U1: np.ndarray, U2: np.ndarray, x: np.ndarray, y: np.ndarray, tol: float = 1e-9) -> tuple[bool, float, float]:
    """Ist (x, y) ein Gleichgewicht? Gibt (ja/nein, Auszahlung Zeile, Auszahlung Spalte); keine reine Abweichung bringt mehr als `tol`."""
    e1, e2 = float(x @ U1 @ y), float(x @ U2 @ y)
    return bool((U1 @ y).max() <= e1 + tol and (x @ U2).max() <= e2 + tol), e1, e2


def payoff_pairs(ext: G.Ext, equilibria: list[tuple[int, int]]) -> list[tuple[float, float]]:
    """Verschiedene Auszahlungspaare (Zeile, Spalte) der reinen Gleichgewichte, auf neun Stellen gerundet, aufsteigend."""
    return sorted({(round(float(ext.U1[i, j]), 9), round(float(ext.U2[i, j]), 9)) for i, j in equilibria})


def action_pairs(ext: G.Ext, equilibria: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    """Zahl der reinen Gleichgewichte je Aktionspaar (a1, a2): gleiche Aktionen, aber verschiedene Nachrichten und Regeln zählen einzeln."""
    out: dict[tuple[int, int], int] = {}
    for i, j in equilibria:
        key = (int(ext.act1[i, j]), int(ext.act2[i, j]))
        out[key] = out.get(key, 0) + 1
    return out


def has_miscoordination(ext: G.Ext, equilibria: list[tuple[int, int]]) -> bool:
    """Gibt es ein reines Gleichgewicht mit verschiedenen Aktionen (A, B) oder (B, A)? Im Kampf der Geschlechter und bei Hirschjagd ist das auszuschließen (Auszahlung 0 bei Abweichung möglich)."""
    return any(a != b for (a, b) in action_pairs(ext, equilibria))


# --------------------------------------------------------------------------------- Lotterie
def lottery_profile(ext: G.Ext) -> tuple[np.ndarray, np.ndarray]:
    """Die Lotterie des beidseitigen Redens: beide senden gleichverteilt eine der K Nachrichten; gleiche Nachrichten → beide spielen A, verschiedene → beide spielen B.
    Gemischtes Profil (x, y) über die Strategien des erweiterten Spiels (K ≥ 2 für ein sinnvolles Zufallsprotokoll, K = 1 ergibt „immer A“)."""
    if ext.kind != "two":
        raise ValueError("die Lotterie braucht beidseitiges Reden")
    K = ext.K
    x = np.zeros(ext.S1)
    for i, (m, r) in enumerate(ext.s1):
        if all(r[own * K + fremd] == (0 if own == fremd else 1) for own in range(K) for fremd in range(K)):
            x[i] = 1.0
    x /= x.sum()
    return x, x.copy()


# --------------------------------------------------------------------------------- exaktes Simplex-Verfahren
def _pivot(T: list[list[Fraction]], r: int, c: int) -> None:
    pv = T[r][c]
    T[r] = [v / pv for v in T[r]]
    for i in range(len(T)):
        if i != r and T[i][c] != 0:
            f = T[i][c]
            T[i] = [a - f * b for a, b in zip(T[i], T[r])]


class Polytope:
    """{z ≥ 0 : A z = b} mit exakter Bruchrechnung. `feasible` nach dem Anlegen, `minimize(c)` / `maximize(c)` liefern den optimalen Wert (None, wenn leer; die hier gebauten Mengen sind beschränkt)."""

    def __init__(self, A: list[list[Fraction]], b: list[Fraction]):
        m, n = len(A), len(A[0]) if A else 0
        self.n = n
        T = []
        for i in range(m):
            row = [Fraction(v) for v in A[i]]
            rhs = Fraction(b[i])
            if rhs < 0:
                row, rhs = [-v for v in row], -rhs
            T.append(row + [ONE if k == i else ZERO for k in range(m)] + [rhs])
        basis = [n + i for i in range(m)]
        obj = [ZERO] * (n + m + 1)
        for i in range(m):
            obj = [o - v for o, v in zip(obj, T[i])]
        for k in range(n, n + m):
            obj[k] = ZERO
        T.append(obj)
        self._run(T, basis, n)
        self.feasible = (-T[-1][-1]) == 0
        self.T, self.basis = None, None
        if self.feasible:
            keep = []
            for i in range(m):                                                       # künstliche Variablen aus der Basis entfernen
                if basis[i] >= n:
                    col = next((k for k in range(n) if T[i][k] != 0), None)
                    if col is None:
                        continue                                                     # redundante Zeile
                    _pivot(T, i, col)
                    basis[i] = col
                keep.append(i)
            rows = [T[i][:n] + [T[i][-1]] for i in keep]
            self.T, self.basis = rows, [basis[i] for i in keep]

    @staticmethod
    def _run(T, basis, limit):
        m = len(basis)
        while True:
            col = next((k for k in range(limit) if T[-1][k] < 0), None)               # Bland: kleinster Index mit negativen reduzierten Kosten
            if col is None:
                return
            best, row = None, None
            for i in range(m):
                if T[i][col] > 0:
                    ratio = T[i][-1] / T[i][col]
                    if best is None or ratio < best or (ratio == best and basis[i] < basis[row]):
                        best, row = ratio, i
            if row is None:
                raise ArithmeticError("unbeschränkte Richtung")
            _pivot(T, row, col)
            basis[row] = col

    def minimize(self, c: list[Fraction]) -> Fraction | None:
        if not self.feasible:
            return None
        T = [r[:] for r in self.T]
        basis = self.basis[:]
        obj = [Fraction(v) for v in c] + [ZERO]
        for i, bcol in enumerate(basis):
            if obj[bcol] != 0:
                f = obj[bcol]
                obj = [o - f * v for o, v in zip(obj, T[i])]
        T.append(obj)
        self._run(T, basis, self.n)
        return -T[-1][-1]

    def maximize(self, c: list[Fraction]) -> Fraction | None:
        v = self.minimize([-Fraction(x) for x in c])
        return None if v is None else -v


def _exact(U: np.ndarray) -> list[list[Fraction]]:
    return [[Fraction(f"{x:.6f}") for x in row] for row in U]


def _best_response_polytope(U: list[list[Fraction]], own: tuple[int, ...], other: tuple[int, ...], extra_rows: int = 0):
    """System für die Mischung z über `other` (Spalten von U): Σ z = 1; für jede Zeile i aus `own` gilt (U z)_i = v, für die übrigen (U z)_i + s_i = v, s_i ≥ 0.
    Variablen: z_j (j in other), v, dann die Schlupfvariablen. Gibt (A, b, Zahl der Zeilen)."""
    n_rows = len(U)
    rest = [i for i in range(n_rows) if i not in own]
    nvar = len(other) + 1 + len(rest)
    A, b = [], []
    A.append([ONE] * len(other) + [ZERO] * (1 + len(rest)))
    b.append(ONE)
    for i in range(n_rows):
        row = [U[i][j] for j in other] + [-ONE] + [ZERO] * len(rest)
        if i in rest:
            row[len(other) + 1 + rest.index(i)] = ONE
        A.append(row)
        b.append(ZERO)
    return A, b, nvar


@dataclass(frozen=True)
class SupportPair:
    rows: tuple[int, ...]
    cols: tuple[int, ...]
    v1: tuple[Fraction, Fraction]        # kleinste und größte Auszahlung der Zeile in den Gleichgewichten dieses Trägerpaares
    v2: tuple[Fraction, Fraction]
    two_messages: bool                   # Zeile kann beide Nachrichten mit positiver Wahrscheinlichkeit senden (nur einseitig, K = 2)
    difference: Fraction                 # größte Reaktionsdifferenz der Spalte (nur bei two_messages sonst 0)


@dataclass(frozen=True)
class EquilibriumSet:
    pairs: tuple[SupportPair, ...]

    def boxes(self) -> list[tuple]:
        """Verschiedene Auszahlungsrechtecke (v1_min, v1_max, v2_min, v2_max) als Brüche, aufsteigend; ein Punkt hat v_min = v_max."""
        return sorted({(p.v1[0], p.v1[1], p.v2[0], p.v2[1]) for p in self.pairs})

    def points(self) -> list[tuple[float, float]]:
        """Die Auszahlungspaare, wenn alle Rechtecke Punkte sind (sonst ValueError)."""
        out = []
        for lo1, hi1, lo2, hi2 in self.boxes():
            if lo1 != hi1 or lo2 != hi2:
                raise ValueError("Gleichgewichte mit verschiedenen Auszahlungen im selben Trägerpaar")
            out.append((float(lo1), float(lo2)))
        return out

    @property
    def degenerate_boxes(self) -> bool:
        return any(lo1 != hi1 or lo2 != hi2 for lo1, hi1, lo2, hi2 in self.boxes())

    @property
    def informative(self) -> bool:
        return any(p.two_messages and p.difference > 0 for p in self.pairs)

    @property
    def max_difference(self) -> float:
        return float(max((p.difference for p in self.pairs if p.two_messages), default=ZERO))


def all_equilibria(ext: G.Ext) -> EquilibriumSet:
    """Alle Gleichgewichte (auch gemischte) des Spiels `ext` über alle Trägerpaare; gerechnet wird nur bei höchstens 6 Strategien je Seite (alle Spiele ohne Reden und einseitiges Reden mit K ≤ 2)."""
    if ext.S1 > 6 or ext.S2 > 6:
        raise ValueError("die Trägerpaar-Rechnung ist auf höchstens 6 Strategien je Seite ausgelegt")
    U1, U2 = _exact(ext.U1), _exact(ext.U2)
    U2T = [[U2[i][j] for i in range(ext.S1)] for j in range(ext.S2)]                  # Zeilen = Spaltenstrategien
    rows_all, cols_all = range(ext.S1), range(ext.S2)
    row_sets = [s for k in range(1, ext.S1 + 1) for s in itertools.combinations(rows_all, k)]
    col_sets = [s for k in range(1, ext.S2 + 1) for s in itertools.combinations(cols_all, k)]
    pairs = []
    for rows in row_sets:
        for cols in col_sets:
            A, b, nvar = _best_response_polytope(U1, rows, cols)                                         # Bedingungen an y (Zeile zufrieden)
            Py = Polytope(A, b)
            if not Py.feasible:
                continue
            A2, b2, nvar2 = _best_response_polytope(U2T, cols, rows)                                     # Bedingungen an x (Spalte zufrieden)
            Px = Polytope(A2, b2)
            if not Px.feasible:
                continue
            cv1 = [ZERO] * len(cols) + [ONE] + [ZERO] * (nvar - len(cols) - 1)
            cv2 = [ZERO] * len(rows) + [ONE] + [ZERO] * (nvar2 - len(rows) - 1)
            v1 = (Py.minimize(cv1), Py.maximize(cv1))
            v2 = (Px.minimize(cv2), Px.maximize(cv2))
            two, diff = False, ZERO
            if ext.kind == "one" and ext.K == 2:
                two, diff = _informative(ext, rows, cols, Py, Px, nvar, nvar2)
            pairs.append(SupportPair(rows, cols, v1, v2, two, diff))
    return EquilibriumSet(tuple(pairs))


def _informative(ext: G.Ext, rows, cols, Py: Polytope, Px: Polytope, nvar: int, nvar2: int) -> tuple[bool, Fraction]:
    """Kann die Zeile in diesem Trägerpaar beide Nachrichten mit positiver Wahrscheinlichkeit senden, und wie verschieden kann die Spalte reagieren?
    Positives Gewicht beider Nachrichten: t > 0 ist erreichbar, wenn max t mit Σ_{Zeilen der Nachricht m} x_i ≥ t für beide m positiv ist (eigenes System mit t und Überschussvariablen)."""
    msg_rows = [[i for i in rows if ext.msg1[i] == m] for m in (0, 1)]
    if not msg_rows[0] or not msg_rows[1]:
        return False, ZERO
    U2T = [[Fraction(f"{ext.U2[i, j]:.6f}") for i in range(ext.S1)] for j in range(ext.S2)]
    A2, b2, _ = _best_response_polytope(U2T, cols, rows)
    width = len(A2[0])
    A3 = [r + [ZERO, ZERO, ZERO] for r in A2]                       # neue Spalten: t, Überschuss e0, e1
    b3 = list(b2)
    for m in (0, 1):
        row = [ZERO] * (width + 3)
        for i in msg_rows[m]:
            row[rows.index(i)] = ONE
        row[width] = -ONE                                           # − t
        row[width + 1 + m] = -ONE                                   # − e_m
        A3.append(row)
        b3.append(ZERO)
    P3 = Polytope(A3, b3)
    tmax = P3.maximize([ZERO] * width + [ONE, ZERO, ZERO])
    if tmax is None or tmax <= 0:
        return False, ZERO
    best = ZERO
    for sign in (1, -1):
        c = []
        for j in cols:
            rule = ext.s2[j][1]
            c.append(Fraction(sign * ((1 if rule[0] == 0 else 0) - (1 if rule[1] == 0 else 0))))
        c += [ZERO] * (nvar - len(cols))
        best = max(best, Py.maximize(c))
    return True, best


def reaction_probability(ext: G.Ext, y: np.ndarray, m: int) -> float:
    """P(Spalte spielt A | Nachricht m) bei der gemischten Spaltenstrategie y über die Regeln (einseitiges Reden)."""
    return float(sum(y[j] for j, (_, rule) in enumerate(ext.s2) if rule[m] == 0))


def pure_ne_summary(ext: G.Ext, tol: float = 1e-12) -> tuple[int, list[tuple[float, float]]]:
    """Zahl der reinen Gleichgewichte und ihre verschiedenen Auszahlungspaare (auf neun Stellen gerundet), ohne die Liste der Index-Paare (für große Matrizen)."""
    mask = (ext.U1 >= ext.U1.max(axis=0, keepdims=True) - tol) & (ext.U2 >= ext.U2.max(axis=1, keepdims=True) - tol)
    pairs = np.unique(np.round(np.stack([ext.U1[mask], ext.U2[mask]], axis=1), 9), axis=0)
    return int(np.count_nonzero(mask)), [(float(a), float(b)) for a, b in pairs]
