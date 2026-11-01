"""Rechenbudget der App (DEMO-PLAYBOOK §1c): wie viele Läufe die App rechnet und wie lange das ungefähr dauert. Reine Anzeige- und Begrenzungshilfen, ohne Einfluss auf die Messzahlen."""

from __future__ import annotations

import vor_constants as C
import vor_games as G
from vor_scenario import Settings, ext_of


def estimated_seconds(ext: G.Ext, steps: int, runs: int) -> float:
    """Grobe Zeitschätzung (Sekunden) eines Lernlaufs für die Anzeige am Spinner und die Begrenzung der Läufe: Schritte · (50 µs fest + je Lauf 0,24 µs plus 1,7 ns je Strategienpaar),
    auf dem Entwicklungsrechner gemessen (200 Läufe, 40 000 Schritte: 4,0 s ohne Reden, 4,6 s einseitig mit K = 2, 17,8 s beidseitig mit K = 2); nur Größenordnung."""
    return steps * (5.0e-5 + runs * (2.4e-7 + 1.7e-9 * ext.S1 * ext.S2))


def learnable(s: Settings) -> bool:
    """Beidseitiges Reden mit K = 3 hat 1 536 Strategien je Seite: dort werden nur Gleichgewichte gezählt, nicht gelernt."""
    return not (s.talk == G.TALK_TWO and s.K >= 3)


def used_runs(s: Settings) -> int:
    """Läufe, mit denen die App tatsächlich rechnet: so viele, dass die geschätzte Zeit C.MAX_SECONDS nicht übersteigt (mindestens 1, höchstens s.runs)."""
    if not learnable(s):
        return 0
    ext = ext_of(s)
    lo, hi = 1, max(1, s.runs)
    while lo < hi:                                   # größtes r mit geschätzter Zeit ≤ Grenze
        mid = (lo + hi + 1) // 2
        if estimated_seconds(ext, s.steps, mid) <= C.MAX_SECONDS:
            lo = mid
        else:
            hi = mid - 1
    return lo


def lewis_estimated_seconds(s: Settings) -> float:
    """Grobe Zeit des Lewis-Teils für die Anzeige am Spinner: Runden · (33 µs fest + 0,22 µs je Lauf), auf dem Entwicklungsrechner gemessen, nur Größenordnung."""
    return s.lewis_rounds * (33e-6 + 0.22e-6 * s.runs)
