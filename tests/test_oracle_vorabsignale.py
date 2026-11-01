"""Orakel-Tests (ORACLE-PLAYBOOK, DEMO-PLAYBOOK §12): jede Kernroutine gegen einen anderen Rechenweg.

* erweitertes Spiel: Auszahlungsmatrix gegen den Spielablauf `play_direct` in ALLEN Paarungen und gegen die bilineare Form bei Zufallspaarungen;
* reine Gleichgewichte: Aufzählung gegen eine Schleife über alle Abweichungen;
* alle Gleichgewichte (Trägerpaare, exakte Brüche) gegen scipy.optimize.linprog auf denselben Trägerpaaren (Zahl der Trägerpaare, Wertebereiche, Gleichgewichtsprobe der Lösung);
* Lotterie gegen alle reinen Abweichungen; Lewis-Effizienz gegen den Spielablauf; Gleichgewichtszahlen n!, 87, 2056 als Handzahlen;
* Lernen: der Lauf gegen eine Schleifenfassung mit denselben Zufallszahlen (Zahl für Zahl) und gegen die exakte stationäre Verteilung der Markov-Kette bei N = 2;
* Information, Ausgangsverteilung und Hörer-Anteil gegen Schleifenfassungen.
Alles ohne Zeit-Assertion; die Lernläufe sind klein (Sekundenbruchteile bis wenige Sekunden)."""

import itertools
import math

import numpy as np
import pytest

import vor_equilibria as E
import vor_games as G
import vor_learning as L
import vor_lewis as LW
from vor_rng import SplitMix64

SPIELE = [(G.GAME_COORD, 7.0, 0.0), (G.GAME_HIRSCH, 7.0, 0.0), (G.GAME_HIRSCH, 7.0, 1.0), (G.GAME_HIRSCH, 3.6, 0.5), (G.GAME_BOS, 7.0, 0.0)]
REDEN = [(G.TALK_NONE, 0), (G.TALK_ONE, 1), (G.TALK_ONE, 2), (G.TALK_ONE, 3), (G.TALK_TWO, 1), (G.TALK_TWO, 2)]


# --------------------------------------------------------------------------------- erweitertes Spiel
@pytest.mark.parametrize("game,b,t", SPIELE)
@pytest.mark.parametrize("talk,K", REDEN)
def test_matrix_equals_the_game_played_out_for_every_pair(game, b, t, talk, K):
    ext = G.build(game, b, t, talk, K)
    for i in range(ext.S1):
        for j in range(ext.S2):
            a1, a2, u1, u2 = G.play_direct(ext, i, j)
            assert (a1, a2) == (ext.act1[i, j], ext.act2[i, j])
            assert u1 == ext.U1[i, j] and u2 == ext.U2[i, j]


def test_strategy_counts_follow_the_formulas():
    """keins 2 × 2; einseitig 2K Zeilenstrategien und 2^K Regeln; beidseitig K · 2^(K²) je Seite (K = 1: 2, K = 2: 32, K = 3: 1 536)."""
    for K in (1, 2, 3):
        e1 = G.build(G.GAME_HIRSCH, 7.0, 0.0, G.TALK_ONE, K)
        assert (e1.S1, e1.S2) == (2 * K, 2 ** K)
    for K, expected in ((1, 2), (2, 32), (3, 1536)):
        assert K * 2 ** (K * K) == expected
        if K < 3:
            e2 = G.build(G.GAME_HIRSCH, 7.0, 0.0, G.TALK_TWO, K)
            assert (e2.S1, e2.S2) == (expected, expected)


def test_random_pairings_equal_the_bilinear_form():
    """Gemischte Strategien: die erwartete Auszahlung aus U1 gleicht der Summe über alle Paarungen des Spielablaufs, gewichtet mit x_i y_j."""
    rng = np.random.default_rng(3)
    for game, b, t in SPIELE:
        for talk, K in ((G.TALK_ONE, 2), (G.TALK_TWO, 2)):
            ext = G.build(game, b, t, talk, K)
            x, y = rng.dirichlet(np.ones(ext.S1)), rng.dirichlet(np.ones(ext.S2))
            direct = sum(x[i] * y[j] * G.play_direct(ext, i, j)[2] for i in range(ext.S1) for j in range(ext.S2))
            assert x @ ext.U1 @ y == pytest.approx(direct, rel=1e-12)


# --------------------------------------------------------------------------------- reine Gleichgewichte
@pytest.mark.parametrize("game,b,t", SPIELE)
@pytest.mark.parametrize("talk,K", [(G.TALK_NONE, 0), (G.TALK_ONE, 1), (G.TALK_ONE, 2), (G.TALK_TWO, 2)])
def test_pure_equilibria_equal_a_loop_over_all_deviations(game, b, t, talk, K):
    ext = G.build(game, b, t, talk, K)
    loop = []
    for i in range(ext.S1):
        for j in range(ext.S2):
            ok1 = all(ext.U1[k, j] <= ext.U1[i, j] + 1e-12 for k in range(ext.S1))
            ok2 = all(ext.U2[i, k] <= ext.U2[i, j] + 1e-12 for k in range(ext.S2))
            if ok1 and ok2:
                loop.append((i, j))
    assert E.pure_ne(ext.U1, ext.U2) == loop
    assert E.pure_ne_count(ext.U1, ext.U2) == len(loop)


def test_pure_equilibria_never_miscoordinate_and_keep_the_payoff_pairs():
    """Zählung über die 5 Spiele × 6 Redearten: kein reines Gleichgewicht mit verschiedenen Aktionen; die Auszahlungspaare der reinen Gleichgewichte sind die ohne Reden."""
    for game, b, t in SPIELE:
        base = None
        for talk, K in REDEN:
            ext = G.build(game, b, t, talk, K)
            ne = E.pure_ne(ext.U1, ext.U2)
            assert not E.has_miscoordination(ext, ne), (game, talk, K)
            pairs = E.payoff_pairs(ext, ne)
            base = pairs if base is None else base
            assert pairs == base, (game, talk, K)


@pytest.mark.parametrize("game,expected", [(G.GAME_COORD, (2, 8, 24, 512)), (G.GAME_HIRSCH, (2, 6, 15, 320)), (G.GAME_BOS, (2, 6, 15, 256))])
def test_numbers_of_pure_equilibria_by_hand_formulas(game, expected):
    """Von Hand (einseitiges Reden, K Nachrichten): ein reines Gleichgewicht ist (Zeile sendet m und spielt a; Spalte folgt auf m mit der passenden Antwort, auf die K − 1 anderen Nachrichten
    mit beliebigen Antworten, soweit keine Abweichung der Zeile auf eine ungenutzte Nachricht lohnt). Reine Koordination: (A, A) K · 2^(K−1), (B, B) ebenso (ein Wechsel auf eine Nachricht mit
    Antwort A bringt gleich viel): 2 · K · 2^(K−1) = 8 bei K = 2 und 24 bei K = 3. Hirschjagd: (A, A) K · 2^(K−1), (B, B) nur K (jede ungenutzte Nachricht muss mit B beantwortet werden, sonst lohnt
    Hirsch für die Zeile): 4 + 2 = 6 bei K = 2, 12 + 3 = 15 bei K = 3. Kampf der Geschlechter: ebenso 6 und 15. Ohne Reden 2. Beidseitig K = 2: Vollaufzählung (512 / 320 / 256), keine Handformel."""
    counts = []
    for talk, K in ((G.TALK_NONE, 0), (G.TALK_ONE, 2), (G.TALK_ONE, 3), (G.TALK_TWO, 2)):
        ext = G.build(game, 7.0, 0.0, talk, K)
        counts.append(E.pure_ne_count(ext.U1, ext.U2))
    assert tuple(counts) == expected


# --------------------------------------------------------------------------------- alle Gleichgewichte gegen scipy
def _linprog_range(U, own, other, direction):
    """Wertebereich v im Trägerpaar mit scipy: Variablen (z über `other`, v), Gleichungen Σz = 1, (U z)_i = v (i in own), Ungleichungen (U z)_i ≤ v (i nicht in own), z ≥ 0, v frei."""
    linprog = pytest.importorskip("scipy.optimize").linprog
    n = len(other)
    rest = [i for i in range(U.shape[0]) if i not in own]
    A_eq = [[1.0] * n + [0.0]] + [[U[i, j] for j in other] + [-1.0] for i in own]
    b_eq = [1.0] + [0.0] * len(own)
    A_ub = [[U[i, j] for j in other] + [-1.0] for i in rest] or None
    b_ub = [0.0] * len(rest) if rest else None
    c = [0.0] * n + [direction]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * n + [(None, None)], method="highs")
    return res


@pytest.mark.parametrize("game,b,t", [(G.GAME_COORD, 7.0, 0.0), (G.GAME_HIRSCH, 7.0, 0.0), (G.GAME_HIRSCH, 7.0, 1.0), (G.GAME_BOS, 7.0, 0.0)])
@pytest.mark.parametrize("talk,K", [(G.TALK_NONE, 0), (G.TALK_ONE, 1), (G.TALK_ONE, 2)])
def test_support_pairs_equal_scipy_linprog_and_every_solution_is_an_equilibrium(game, b, t, talk, K):
    ext = G.build(game, b, t, talk, K)
    eq = E.all_equilibria(ext)
    found = {(p.rows, p.cols): p for p in eq.pairs}
    rows_all, cols_all = range(ext.S1), range(ext.S2)
    row_sets = [s for k in range(1, ext.S1 + 1) for s in itertools.combinations(rows_all, k)]
    col_sets = [s for k in range(1, ext.S2 + 1) for s in itertools.combinations(cols_all, k)]
    feasible = 0
    for rows in row_sets:
        for cols in col_sets:
            ry = _linprog_range(ext.U1, rows, cols, 1.0)                         # y mit Wertebereich v1
            rx = _linprog_range(ext.U2.T, cols, rows, 1.0)
            ok = ry.status == 0 and rx.status == 0
            assert ((rows, cols) in found) == ok, (rows, cols)
            if not ok:
                continue
            feasible += 1
            p = found[(rows, cols)]
            hi_y = _linprog_range(ext.U1, rows, cols, -1.0)
            hi_x = _linprog_range(ext.U2.T, cols, rows, -1.0)
            assert float(p.v1[0]) == pytest.approx(ry.fun, abs=1e-9) and float(p.v1[1]) == pytest.approx(-hi_y.fun, abs=1e-9)
            assert float(p.v2[0]) == pytest.approx(rx.fun, abs=1e-9) and float(p.v2[1]) == pytest.approx(-hi_x.fun, abs=1e-9)
            y = np.zeros(ext.S2)
            y[list(cols)] = ry.x[:-1]
            x = np.zeros(ext.S1)
            x[list(rows)] = rx.x[:-1]
            ok_ne, e1, e2 = E.is_ne_mixed(ext.U1, ext.U2, x, y)
            assert ok_ne and e1 == pytest.approx(ry.fun, abs=1e-9) and e2 == pytest.approx(rx.fun, abs=1e-9)
    assert feasible == len(found)


def test_support_pair_counts_and_payoff_sets_by_hand():
    """Von Hand: 2×2 ohne Reden hat 3 Gleichgewichte (zwei reine, ein gemischtes). Reine Koordination: Auszahlungen 1 und 1/2; Hirschjagd 9, 0, 7, 7: 9 und 7 (gemischt 7);
    mit Verlockung t = 1 (Hase allein gegen Hirsch 8): gemischt p = 7/(9 − 1) = 7/8 mit Auszahlung 9 · 7/8 = 63/8 = 7,875; Kampf der Geschlechter: (2; 1), (1; 2) und 2/3 für beide."""
    expected = {
        (G.GAME_COORD, 0.0): [(0.5, 0.5), (1.0, 1.0)],
        (G.GAME_HIRSCH, 0.0): [(7.0, 7.0), (9.0, 9.0)],
        (G.GAME_HIRSCH, 1.0): [(7.0, 7.0), (7.875, 7.875), (9.0, 9.0)],
        (G.GAME_BOS, 0.0): [(2 / 3, 2 / 3), (1.0, 2.0), (2.0, 1.0)],
    }
    for (game, t), points in expected.items():
        eq = E.all_equilibria(G.build(game, 7.0, t, G.TALK_NONE, 0))
        assert len(eq.pairs) == 3
        for got, want in zip(eq.points(), points):
            assert got == pytest.approx(want, abs=1e-12)
        assert len(eq.points()) == len(points)


def test_exact_simplex_equals_scipy_on_random_systems():
    """Das kleine Simplex-Verfahren in Brüchen gegen linprog auf Zufallssystemen {z ≥ 0: A z = b} (zulässig und unzulässig, Minimum und Maximum)."""
    linprog = pytest.importorskip("scipy.optimize").linprog
    from fractions import Fraction
    rng = np.random.default_rng(11)
    counts = {"optimal": 0, "infeasible": 0, "unbounded": 0}
    for _ in range(120):
        m = int(rng.integers(1, 4))
        n = int(rng.integers(m + 1, 6))
        A = rng.integers(-3, 4, size=(m, n))
        z0 = rng.integers(0, 3, size=n)
        b = A @ z0 if rng.random() < 0.7 else rng.integers(-3, 4, size=m)
        c = rng.integers(-3, 4, size=n)
        poly = E.Polytope([[Fraction(int(v)) for v in row] for row in A], [Fraction(int(v)) for v in b])
        ref = linprog(c, A_eq=A, b_eq=b, bounds=[(0, None)] * n, method="highs")
        assert poly.feasible == (ref.status in (0, 3))
        if not poly.feasible:
            counts["infeasible"] += 1
        elif ref.status == 0:
            counts["optimal"] += 1
            assert float(poly.minimize([Fraction(int(v)) for v in c])) == pytest.approx(ref.fun, abs=1e-9)
        else:
            counts["unbounded"] += 1
            with pytest.raises(ArithmeticError):
                poly.minimize([Fraction(int(v)) for v in c])
    assert counts["optimal"] >= 20 and counts["infeasible"] >= 5


# --------------------------------------------------------------------------------- Lotterie
@pytest.mark.parametrize("game,b,t", [(G.GAME_COORD, 7.0, 0.0), (G.GAME_HIRSCH, 7.0, 0.0), (G.GAME_HIRSCH, 7.0, 1.0), (G.GAME_BOS, 7.0, 0.0)])
def test_lottery_is_an_equilibrium_against_every_pure_deviation(game, b, t):
    K = 2
    ext = G.build(game, b, t, G.TALK_TWO, K)
    x, y = E.lottery_profile(ext)
    assert x.sum() == pytest.approx(1.0) and np.count_nonzero(x) == K
    e1, e2 = x @ ext.U1 @ y, x @ ext.U2 @ y
    assert max(ext.U1[i] @ y for i in range(ext.S1)) <= e1 + 1e-12
    assert max(x @ ext.U2[:, j] for j in range(ext.S2)) <= e2 + 1e-12


def test_lottery_payoffs_by_hand(bos_hand, lottery_hirsch_hand):
    """Kampf der Geschlechter 1,5 / 1,5; Hirschjagd 8; reine Koordination 1 (gleiche Nachricht → A, sonst B: immer gleiche Aktionen)."""
    x, y = E.lottery_profile(G.build(G.GAME_BOS, 7.0, 0.0, G.TALK_TWO, 2))
    ext = G.build(G.GAME_BOS, 7.0, 0.0, G.TALK_TWO, 2)
    assert (x @ ext.U1 @ y, x @ ext.U2 @ y) == (pytest.approx(bos_hand["lottery"]), pytest.approx(bos_hand["lottery"]))
    exh = G.build(G.GAME_HIRSCH, 7.0, 0.0, G.TALK_TWO, 2)
    xh, yh = E.lottery_profile(exh)
    assert xh @ exh.U1 @ yh == pytest.approx(lottery_hirsch_hand)
    exc = G.build(G.GAME_COORD, 7.0, 0.0, G.TALK_TWO, 2)
    xc, yc = E.lottery_profile(exc)
    assert xc @ exc.U1 @ yc == pytest.approx(1.0)


def test_the_message_alone_carries_no_information_in_the_lottery():
    """Einzeln trägt die Nachricht nichts: I(Nachricht der Zeile; Aktion der Spalte) = 0 bei der Lotterie (die Aktion der Spalte hängt am Paar der Nachrichten)."""
    ext = G.build(G.GAME_BOS, 7.0, 0.0, G.TALK_TWO, 2)
    x, y = E.lottery_profile(ext)
    cx, cy = (x * 1000)[None, :], (y * 1000)[None, :]
    assert L.mutual_information(ext, cx, cy, 1000)[0] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------------- Lewis
def test_lewis_equilibrium_counts_are_the_hand_numbers(lewis_hand):
    """n = 2: 6 (davon 2 Signalsysteme); n = 3: 87 (6 Signalsysteme, 72 mit Effizienz 2/3, 9 mit 1/3); n = 4: 2056 (24, 864 mit 3/4, 1152 mit 1/2, 16 mit 1/4)."""
    r2 = LW.pure_ne_lewis(2)
    assert (r2["count"], r2["signalling_systems"], r2["by_efficiency"][0.5]) == (lewis_hand["count"], lewis_hand["signalling_systems"], lewis_hand["pooling"])
    r3 = LW.pure_ne_lewis(3)
    assert r3["count"] == 87 and r3["signalling_systems"] == math.factorial(3)
    assert {round(k, 6): v for k, v in r3["by_efficiency"].items()} == {round(1 / 3, 6): 9, round(2 / 3, 6): 72, 1.0: 6}
    r4 = LW.pure_ne_lewis(4)
    assert r4["count"] == 2056 and r4["signalling_systems"] == math.factorial(4)
    assert {round(k, 6): v for k, v in r4["by_efficiency"].items()} == {0.25: 16, 0.5: 1152, 0.75: 864, 1.0: 24}


@pytest.mark.parametrize("n,ns", [(3, 3), (4, 3), (3, 5)])
def test_lewis_efficiency_formula_equals_the_game_played_out(n, ns):
    rng = np.random.default_rng(7 + n + ns)
    ps = rng.dirichlet(np.ones(ns), size=(1, n))
    pr = rng.dirichlet(np.ones(n), size=(1, ns))
    trials = 400_000
    state = rng.integers(0, n, trials)
    sig = np.array([rng.choice(ns, p=ps[0, k]) for k in range(n)])                  # Platzhalter, ersetzt unten durch Ziehungen je Versuch
    cum_s, cum_r = np.cumsum(ps[0], axis=1), np.cumsum(pr[0], axis=1)
    sig = (rng.random(trials)[:, None] > cum_s[state]).sum(axis=1)
    act = (rng.random(trials)[:, None] > cum_r[sig]).sum(axis=1)
    wins = (act == state)
    exact = LW.efficiency(ps, pr)[0]
    assert abs(wins.mean() - exact) < 4 * math.sqrt(exact * (1 - exact) / trials)


def test_lewis_start_efficiency_is_one_over_n_and_signalling_system_is_one():
    for n, ns in ((3, 3), (4, 6)):
        assert LW.efficiency(np.full((1, n, ns), 1 / ns), np.full((1, ns, n), 1 / n))[0] == pytest.approx(1 / n)
    ps = np.eye(3)[None]
    pr = np.eye(3)[None]
    assert LW.efficiency(ps, pr)[0] == pytest.approx(1.0)
    assert LW.efficiency(ps, np.roll(pr, 1, axis=1))[0] == pytest.approx(0.0)


def test_roth_erev_equals_a_loop_with_the_same_random_numbers():
    """Die vektorisierte Fassung gegen eine Schleifenfassung, die dieselben SplitMix64-Zahlen Stück für Stück verbraucht (Form (Runden, Läufe, 3))."""
    n, ns, rounds, runs, seed = 3, 4, 60, 3, 91
    snap = LW.roth_erev(n, ns, rounds, runs, seed)[-1]
    gen = SplitMix64(seed)
    ws = [[[1.0] * ns for _ in range(n)] for _ in range(runs)]
    wr = [[[1.0] * n for _ in range(ns)] for _ in range(runs)]
    for _ in range(rounds):
        for r in range(runs):
            u0, u1, u2 = gen.uniform(), gen.uniform(), gen.uniform()
            state = min(int(u0 * n), n - 1)
            row = ws[r][state]
            tot, acc, sig = sum(row), 0.0, None
            for s, w in enumerate(row):
                acc += w
                if acc > u1 * tot:
                    sig = s
                    break
            row = wr[r][sig]
            tot, acc, act = sum(row), 0.0, None
            for a, w in enumerate(row):
                acc += w
                if acc > u2 * tot:
                    act = a
                    break
            if act == state:
                ws[r][state][sig] += 1.0
                wr[r][sig][act] += 1.0
    for r in range(runs):
        for k in range(n):
            assert snap.ps[r, k] == pytest.approx(np.array(ws[r][k]) / sum(ws[r][k]), rel=1e-12)
        for s in range(ns):
            assert snap.pr[r, s] == pytest.approx(np.array(wr[r][s]) / sum(wr[r][s]), rel=1e-12)


# --------------------------------------------------------------------------------- Lernen
def _reference_run(ext, beta, n_agents, steps, runs, seed, start):
    """Schleifenfassung des Lernens: dieselben Zahlen in derselben Reihenfolge, Strategien als Listen, Logit per Gumbel-Max in reinem Python."""
    gen = SplitMix64(seed)
    S1, S2 = ext.S1, ext.S2
    scale = max(float(ext.U1.max()), float(ext.U2.max()))
    U1 = [[float(ext.U1[i, j]) / scale for j in range(S2)] for i in range(S1)]
    U2 = [[float(ext.U2[i, j]) / scale for j in range(S2)] for i in range(S1)]
    init = [[[gen.uniform() for _ in range(n_agents)] for _ in range(2)] for _ in range(runs)]
    cx, cy = [], []
    for r in range(runs):
        if start == L.START_OLD:
            a, b = [0] * S1, [0] * S2
            a[G.old_convention(ext, 0)] = n_agents
            b[G.old_convention(ext, 1)] = n_agents
        else:
            a, b = [0] * S1, [0] * S2
            for u in init[r][0]:
                a[min(int(u * S1), S1 - 1)] += 1
            for u in init[r][1]:
                b[min(int(u * S2), S2 - 1)] += 1
        cx.append(a)
        cy.append(b)
    smax = max(S1, S2)
    for _ in range(steps):
        for r in range(runs):
            pick = [gen.uniform() for _ in range(2 + smax)]
            row_side = pick[0] < 0.5
            who = min(int(pick[1] * n_agents), n_agents - 1)
            g = [-math.log(-math.log(u + 0.5 * 2.0 ** -53)) for u in pick[2:]]
            if row_side:
                score = [beta * sum(cy[r][j] / n_agents * U1[i][j] for j in range(S2)) + g[i] for i in range(S1)]
                new = max(range(S1), key=lambda i: (score[i], -i))
                cum, old = 0, None
                for i in range(S1):
                    cum += cx[r][i]
                    if cum > who:
                        old = i
                        break
                cx[r][old] -= 1
                cx[r][new] += 1
            else:
                score = [beta * sum(cx[r][i] / n_agents * U2[i][j] for i in range(S1)) + g[j] for j in range(S2)]
                new = max(range(S2), key=lambda j: (score[j], -j))
                cum, old = 0, None
                for j in range(S2):
                    cum += cy[r][j]
                    if cum > who:
                        old = j
                        break
                cy[r][old] -= 1
                cy[r][new] += 1
    return np.array(cx), np.array(cy)


@pytest.mark.parametrize("game,b,t,talk,K,start", [
    (G.GAME_HIRSCH, 7.0, 0.0, G.TALK_NONE, 0, L.START_RANDOM), (G.GAME_BOS, 7.0, 0.0, G.TALK_ONE, 2, L.START_RANDOM), (G.GAME_HIRSCH, 3.6, 0.0, G.TALK_ONE, 3, L.START_OLD),
    (G.GAME_HIRSCH, 7.0, 1.0, G.TALK_TWO, 2, L.START_OLD), (G.GAME_COORD, 7.0, 0.0, G.TALK_TWO, 1, L.START_RANDOM),
])
def test_population_run_equals_the_loop_version_with_the_same_random_numbers(game, b, t, talk, K, start):
    ext = G.build(game, b, t, talk, K)
    n_agents, steps, runs, seed, beta = 6, 150, 3, 4711, 12.0
    snaps = L.population_run(ext, beta, n_agents, steps, runs, seed, start, (0, 75, steps))
    cx, cy = _reference_run(ext, beta, n_agents, steps, runs, seed, start)
    # die Besetzungen selbst sind nicht Teil der Momentaufnahme: Ausgangsverteilung und Information daraus nachrechnen
    q = L.outcome_shares(ext, cx, cy, n_agents)
    last = snaps[-1]
    assert last.aa == pytest.approx(q[:, 0, 0], abs=1e-12) and last.bb == pytest.approx(q[:, 1, 1], abs=1e-12)
    assert last.mis == pytest.approx(q[:, 0, 1] + q[:, 1, 0], abs=1e-12)
    assert last.info == pytest.approx(L.mutual_information(ext, cx, cy, n_agents), abs=1e-12)


def _stationary_distribution(ext, beta, n_agents):
    """Exakte stationäre Verteilung der Markov-Kette über (Zahl der Zeilenakteure mit Strategie 0, Zahl der Spaltenakteure mit Strategie 0) für ein Spiel mit je 2 Strategien."""
    scale = max(float(ext.U1.max()), float(ext.U2.max()))
    U1, U2 = ext.U1 / scale, ext.U2 / scale
    N = n_agents
    states = [(i, j) for i in range(N + 1) for j in range(N + 1)]
    index = {s: k for k, s in enumerate(states)}
    P = np.zeros((len(states), len(states)))
    for (i, j), k in index.items():
        x = np.array([i, N - i]) / N
        y = np.array([j, N - j]) / N
        w1 = np.exp(beta * (U1 @ y))
        w1 /= w1.sum()
        w2 = np.exp(beta * (x @ U2))
        w2 /= w2.sum()
        for old, count in ((0, i), (1, N - i)):                                  # Zeile überprüft (Wahrscheinlichkeit 1/2)
            if count == 0:
                continue
            for new in (0, 1):
                ni = i - (old == 0) + (new == 0)
                P[k, index[(ni, j)]] += 0.5 * count / N * w1[new]
        for old, count in ((0, j), (1, N - j)):                                  # Spalte überprüft
            if count == 0:
                continue
            for new in (0, 1):
                nj = j - (old == 0) + (new == 0)
                P[k, index[(i, nj)]] += 0.5 * count / N * w2[new]
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1))])
    return states, pi / pi.sum()


@pytest.mark.parametrize("game,b,beta", [(G.GAME_COORD, 7.0, 2.0), (G.GAME_BOS, 7.0, 3.0), (G.GAME_HIRSCH, 4.0, 4.0)])
def test_population_run_matches_the_exact_stationary_distribution_for_n_equal_two(game, b, beta):
    """Ohne Reden und mit N = 2 Akteuren je Population ist die Kette klein (9 Zustände); die Verteilung der Läufe nach vielen Schritten gleicht der exakten stationären Verteilung."""
    ext = G.build(game, b, 0.0, G.TALK_NONE, 0)
    states, pi = _stationary_distribution(ext, beta, 2)
    runs = 3000
    snaps = L.population_run(ext, beta, 2, 300, runs, 5, L.START_RANDOM)
    # Zustand aus der Ausgangsverteilung nicht rekonstruierbar: stattdessen die Verteilung von (A, A), (B, B) und Fehlkoordination vergleichen
    aa = sum(p * (i / 2) * (j / 2) for (i, j), p in zip(states, pi))
    bb = sum(p * ((2 - i) / 2) * ((2 - j) / 2) for (i, j), p in zip(states, pi))
    last = snaps[-1]
    se = lambda arr: arr.std() / math.sqrt(len(arr))                               # noqa: E731
    assert abs(last.aa.mean() - aa) < 4 * se(last.aa) + 1e-3
    assert abs(last.bb.mean() - bb) < 4 * se(last.bb) + 1e-3


def test_gumbel_max_reproduces_the_softmax():
    """argmax(β·u + Gumbel) wählt die Strategie s mit Wahrscheinlichkeit ∝ exp(β·u_s)."""
    rng = np.random.default_rng(2)
    u = np.array([0.9, 0.5, 0.2, 0.9])
    beta = 3.0
    draws = L.gumbel(rng.random((200_000, 4)))
    pick = np.argmax(beta * u + draws, axis=1)
    freq = np.bincount(pick, minlength=4) / len(pick)
    expect = np.exp(beta * u) / np.exp(beta * u).sum()
    assert np.all(np.abs(freq - expect) < 4 * np.sqrt(expect * (1 - expect) / len(pick)))


def test_mutual_information_equals_a_loop_over_all_pairings(rng):
    """Information aus der Verbundverteilung (Nachricht, fremde Aktion) per Schleife über alle Paarungen."""
    def loop_info(ext, x, y):
        total = 0.0
        legs = [(ext.msg1, ext.act2, x, y, False)]
        if ext.kind == "two":
            legs.append((ext.msg2, ext.act1, y, x, True))
        for msg, act, wm, wa, swapped in legs:
            joint = {}
            for i in range(ext.S1):
                for j in range(ext.S2):
                    m = msg[j] if swapped else msg[i]
                    a = act[i, j]
                    joint[(m, a)] = joint.get((m, a), 0.0) + x[i] * y[j]
            pm, pa = {}, {}
            for (m, a), p in joint.items():
                pm[m] = pm.get(m, 0.0) + p
                pa[a] = pa.get(a, 0.0) + p
            total += sum(p * math.log2(p / (pm[m] * pa[a])) for (m, a), p in joint.items() if p > 0)
        return total
    for talk, K in ((G.TALK_ONE, 2), (G.TALK_ONE, 3), (G.TALK_TWO, 2)):
        ext = G.build(G.GAME_BOS, 7.0, 0.0, talk, K)
        x, y = rng.dirichlet(np.ones(ext.S1)), rng.dirichlet(np.ones(ext.S2))
        got = L.mutual_information(ext, (x * 1000)[None, :], (y * 1000)[None, :], 1000)[0]
        assert got == pytest.approx(loop_info(ext, x, y), abs=1e-12)


def test_information_of_a_message_that_the_listener_follows_is_one_bit(gleich_gross):
    """Hälfte der Sprecher sendet 1 und spielt A, die andere 2 und B; alle Hörer folgen (Regel: 1 → A, 2 → B): I = 1 Bit."""
    ext = G.build(G.GAME_COORD, 7.0, 0.0, G.TALK_ONE, 2)
    cx = np.zeros((1, ext.S1))
    cx[0, ext.s1.index((0, 0))] = 10
    cx[0, ext.s1.index((1, 1))] = 10
    cy = np.zeros((1, ext.S2))
    cy[0, ext.s2.index((None, (0, 1)))] = 20
    assert L.mutual_information(ext, cx, cy, 20)[0] == pytest.approx(gleich_gross)


# --------------------------------------------------------------------------------- Hörer-Anteil bei festen Sprechern (Formel gegen Simulation)
@pytest.mark.parametrize("n_senders,a_senders", [(0, 0), (1, 1), (4, 1), (10, 1), (5, 3), (10, 10)])
def test_hearer_share_formula_equals_simulated_hearers_at_fixed_speakers(n_senders, a_senders):
    """Die Sprecher bleiben fest (n senden Nachricht 1, davon a mit A, die übrigen 20 − n senden Nachricht 2 mit B); nur die Hörer überprüfen (Spalte revidiert in jedem Schritt).
    Der stationäre Anteil der Hörer mit „Nachricht 1 → A“ gleicht der Formel 1 / (1 + exp(−β (a − p* n) / N)) (Mittel über 3000 Läufe, vier Standardfehler)."""
    import vor_experiments as XP
    b, beta, N, runs = 3.6, 20.0, 20, 3000
    ext = G.build(G.GAME_HIRSCH, b, 0.0, G.TALK_ONE, 2)
    cx = np.zeros((runs, ext.S1), dtype=np.int64)
    cx[:, ext.s1.index((0, 0))] = a_senders
    cx[:, ext.s1.index((0, 1))] = n_senders - a_senders
    cx[:, ext.s1.index((1, 1))] = N - n_senders
    cy = np.zeros((runs, ext.S2), dtype=np.int64)
    cy[:, ext.s2.index((None, (1, 1)))] = N
    U1n, U2n = L.scaled_payoffs(ext)
    stream = L.Stream(17)
    width = 2 + max(ext.S1, ext.S2)
    rows = np.arange(runs)
    for _ in range(400):
        u = stream.take(runs * width).reshape(runs, width)
        u[:, 0] = 0.9                                                              # u ≥ ½: die Spalte überprüft
        L.revise_step(cx, cy, u, rows, U1n, U2n, beta, N)
    assert (cx.sum(axis=1) == N).all() and (cy.sum(axis=1) == N).all()
    share = L.listener_share_hirsch(ext, cy, N)[:, 0]
    expect = XP.hearer_share(beta, N, b / 9, n_senders, a_senders)
    assert abs(share.mean() - expect) < 4 * share.std() / math.sqrt(runs) + 1e-3, (share.mean(), expect)
