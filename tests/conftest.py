"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b). Alle Erwartungswerte stehen als Rechnung dabei."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def hirschjagd_hand():
    """Hirschjagd 9, 0, 7, 7 (A/A 9, A allein 0, B gegen A 7, B gegen B 7) ohne Reden. Von Hand:
    reine Gleichgewichte (A, A) mit (9; 9) und (B, B) mit (7; 7); (A, B) bringt der Zeile 0 und der Spalte 7, die Zeile wechselt also zu B: kein Gleichgewicht.
    Gemischt: die Spalte spielt A mit p, die Zeile ist indifferent bei 9p = 7, also p* = 7/9; Auszahlung 9 · 7/9 = 7."""
    return {"p_star": 7 / 9, "mixed_payoff": 7.0, "pure": {(0, 0): (9.0, 9.0), (1, 1): (7.0, 7.0)}}


@pytest.fixture
def bos_hand():
    """Kampf der Geschlechter (Zeile 2/1, Spalte 1/2). Von Hand: gemischtes Gleichgewicht: die Spalte spielt A mit q, die Zeile ist indifferent bei 2q = 1 − q, also q = 1/3;
    die Zeile spielt A mit p, die Spalte ist indifferent bei p = 2(1 − p), also p = 2/3. Auszahlung beider 2/3 (2 · 1/3 = 2/3 für die Zeile, 2/3 für die Spalte).
    Lotterie des beidseitigen Redens (gleiche Nachricht → A, sonst B): halb (A, A) mit (2; 1), halb (B, B) mit (1; 2): (2 + 1)/2 = 1,5 für beide."""
    return {"row_a": 2 / 3, "col_a": 1 / 3, "mixed_payoff": 2 / 3, "lottery": 1.5}


@pytest.fixture
def lottery_hirsch_hand():
    """Lotterie in der Hirschjagd 9, 0, 7, 7 mit zwei Nachrichten: gleiche Nachricht (Wahrscheinlichkeit 1/2) → (A, A) mit 9, verschiedene → (B, B) mit 7: (9 + 7)/2 = 8."""
    return 8.0


@pytest.fixture
def handshake_hand():
    """Geheimer Handschlag in der Hirschjagd 9, 0, 7, 7, beidseitiges Reden mit K = 2: Bewohner: Nachricht 1, immer B. Mutant: Nachricht 2, A, wenn der andere auch 2 sendet, sonst B.
    Mutant gegen Bewohner: Nachrichten verschieden, Mutant spielt B, Bewohner B: 7. Bewohner gegen Mutant: 7. Bewohner gegen Bewohner: 7. Mutant gegen Mutant: gleiche Nachricht, beide A: 9.
    Bei einem Mutantenanteil 0,1 in der anderen Population: Mutant 0,1 · 9 + 0,9 · 7 = 7,2, Bewohner 7."""
    return {"mutant_vs_mutant": 9.0, "mutant_vs_resident": 7.0, "resident_vs_mutant": 7.0, "resident_vs_resident": 7.0, "mutant_at_01": 7.2, "resident_at_01": 7.0}


@pytest.fixture
def lewis_hand():
    """Lewis-Signalspiel mit n = 2 Zuständen und 2 Signalen. Sprecherabbildungen Zustand → Signal: 4, Hörerabbildungen Signal → Handlung: 4, also 16 Paare.
    Reine Gleichgewichte: die 2 Signalsysteme (Sprecher bijektiv, Hörer die Umkehrung; Effizienz 1) und 4 Pooling-Gleichgewichte (Sprecher sendet immer dasselbe Signal, der Hörer
    antwortet auf das gesendete Signal mit einer festen Handlung; die Handlung auf das ungesendete Signal ist beliebig, 2 Möglichkeiten; Effizienz 1/2). Zusammen 6."""
    return {"count": 6, "signalling_systems": 2, "pooling": 4, "pooling_efficiency": 0.5}


@pytest.fixture
def gleich_gross():
    """Besetzungen für die Informationsrechnung: eine Zeile, die Nachricht 1 mit A und Nachricht 2 mit B (bei K = 2: Sprecher je zur Hälfte), und eine Spalte, die der Nachricht folgt,
    hat I = 1 Bit (die Aktion verrät die Nachricht vollständig)."""
    return 1.0


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
