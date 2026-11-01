"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Rechenaufwand am Slider-Maximum (DEMO-PLAYBOOK §1c): Ein Lernlauf kostet O(Schritte · Läufe · S1 · S2) mit S1 · S2 Strategienpaaren (keins 4, einseitig K = 3: 6 · 8 = 48,
beidseitig K = 2: 32 · 32 = 1 024); die Pythonschleife über die Schritte kostet je Schritt rund 50 µs fest, dazu je Lauf rund 0,25 µs plus 1,7 ns je Strategienpaar (auf dem Entwicklungsrechner
gemessen, nur Größenordnung). Die App begrenzt die Läufe so, dass die geschätzte Rechenzeit MAX_SECONDS nicht übersteigt, und meldet die Kürzung. Beidseitig mit K = 3 hat 1 536 Strategien je
Seite (2,4 Mio. Paare): dort werden nur Gleichgewichte gezählt, nicht gelernt. Trägerpaar-Rechnung in Brüchen: 225 Trägerpaare bei K = 2 (rund 0,2 s je Spiel), nur bis 6 Strategien je Seite.
Lewis-Spiel: O(Runden · Läufe) mit rund 33 µs je Runde und 0,22 µs je Lauf (100 000 Runden mit 200 Läufen: rund 8 s)."""

import vor_games as G

# --- Regler: Spiel und Reden ----------------------------------------------------------------------------------------------------------------
B_MIN, B_MAX, DEFAULT_B = 1.0, 7.5, 7.0              # Auszahlung des Hasen b (Hirschjagd); Schwelle p* = b / (9 − t). b + t < 9 bleibt: 7,5 + 1 < 9
T_MIN, T_MAX, DEFAULT_T = 0.0, 1.0, 0.0              # Verlockung t: Hase allein gegen Hirsch bringt b + t
K_MIN, K_MAX, DEFAULT_K = 1, 3, 2                    # Zahl der Nachrichten
# --- Regler: Lernen -------------------------------------------------------------------------------------------------------------------------
BETA_MIN, BETA_MAX, DEFAULT_BETA = 2.0, 100.0, 20.0  # Schärfe β der Logit-Wahl bei Auszahlungen, die auf den Höchstwert 1 skaliert sind
N_MIN, N_MAX, DEFAULT_N = 10, 40, 20                 # Akteure je Population
STEPS_MIN, STEPS_MAX, DEFAULT_STEPS = 2_000, 200_000, 40_000
RUNS_MIN, RUNS_MAX, DEFAULT_RUNS = 4, 200, 40
SEED_MAX, DEFAULT_SEED = 999_999, 35
# --- Regler: Lewis-Signalspiel --------------------------------------------------------------------------------------------------------------
LEWIS_N_MIN, LEWIS_N_MAX, DEFAULT_LEWIS_N = 2, 6, 3  # Zahl der Zustände
LEWIS_NS_MIN, LEWIS_NS_MAX, DEFAULT_LEWIS_NS = 2, 12, 3   # Zahl der Signale (2n bei n = 6)
LEWIS_ROUNDS_MIN, LEWIS_ROUNDS_MAX, DEFAULT_LEWIS_ROUNDS = 1_000, 100_000, 20_000

MAX_SECONDS = 25.0                                   # so lange darf ein Lernlauf der App nach Schätzung höchstens dauern; darüber rechnet die App mit weniger Läufen
TRACE_POINTS = 40                                    # Stützstellen der Verlaufskurven
EXACT_MAX_STRATEGIES = 6                             # Trägerpaar-Rechnung bis zu so vielen Strategien je Seite

# --- Teile und Auswahllisten ------------------------------------------------------------------------------------------------------------------
PART_TALK = "Kostenloses Reden vor dem Spiel"
PART_LEWIS = "Lewis-Signalspiel"
PARTS = (PART_TALK, PART_LEWIS)
START_RANDOM = "zufällig"
START_OLD = "Altkonvention (alle B)"
STARTS = (START_RANDOM, START_OLD)

# --- feste Annahmen der Messreihen ----------------------------------------------------------------------------------------------------------
MEAS_BLOCKS = 4                                       # unabhängige Blöcke (Zufallsströme) je Zelle
MEAS_RUNS = 200                                       # Läufe je Block
LEWIS_BLOCKS, LEWIS_RUNS = 4, 200
EXP_SEED = 20261101                                   # Grundlage der Seeds der Live-Experimente
THRESHOLD_PSTARS = (0.4, 0.5, 0.6)                    # E3: Schwellen p* = b / 9 mit b = 3,6 / 4,5 / 5,4
E3_BETAS = (20.0, 40.0)
E3_STEPS = 40_000
E3_TALKS = ((G.TALK_NONE, 0), (G.TALK_ONE, 1), (G.TALK_ONE, 2), (G.TALK_ONE, 3), (G.TALK_TWO, 2))
HANDSHAKE_B = 7.0                                     # p* = 7/9 für den geheimen Handschlag
HANDSHAKE_BETAS = (40.0, 100.0)
HANDSHAKE_STEPS_PER_AGENT = 10_000                    # Schritte = 10 000 · N (N = 20: 200 000)
HANDSHAKE_NS = (10, 20, 40)
BOS_BETAS = (20.0,)
BOS_STEPS = 40_000
RANDOM_BETAS = (6.0, 20.0)
RANDOM_STEPS = 40_000
LEWIS_PAIRS = ((2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (3, 6), (4, 2), (4, 3), (4, 4), (4, 8), (6, 6))
LEWIS_CHECKS = (1_000, 10_000, 100_000)
E2_TS = (0.0, 0.25, 0.5, 1.0)

# --- Farben ---------------------------------------------------------------------------------------------------------------------------------
COLOR_A = "#2F6B65"            # A = Hirsch / Treffpunkt 1
COLOR_B = "#C0762B"            # B = Hase / Treffpunkt 2
COLOR_MIS = "#d62728"          # Fehlkoordination
COLOR_LINE = "#4c78a8"
COLOR_GREY = "rgba(150,150,150,0.65)"

# --- Presets --------------------------------------------------------------------------------------------------------------------------------
BASE = {"part": PART_TALK, "game": G.GAME_HIRSCH, "b": DEFAULT_B, "t": DEFAULT_T, "talk": G.TALK_ONE, "K": DEFAULT_K, "beta": DEFAULT_BETA, "n_agents": DEFAULT_N, "start": START_RANDOM,
        "steps": DEFAULT_STEPS, "runs": DEFAULT_RUNS, "seed": DEFAULT_SEED, "lewis_n": DEFAULT_LEWIS_N, "lewis_ns": DEFAULT_LEWIS_NS, "lewis_rounds": DEFAULT_LEWIS_ROUNDS}
PRESETS = {
    "Standard (Hirschjagd, Zeile spricht)": dict(BASE),
    "Beispiel in der Art von Aumann": {**BASE, "t": 1.0},
    "Kampf der Geschlechter: wer spricht?": {**BASE, "game": G.GAME_BOS, "K": 3},
    "Altkonvention verlassen": {**BASE, "b": 3.6, "K": 3, "start": START_OLD},
    "Geheimer Handschlag": {**BASE, "talk": G.TALK_TWO, "beta": 100.0, "start": START_OLD, "steps": 200_000},
    "Lewis: drei Zustände": {**BASE, "part": PART_LEWIS, "runs": 100, "lewis_rounds": 100_000},
}
PRESET_HELP = {
    "Standard (Hirschjagd, Zeile spricht)": "Hirschjagd mit Hase-Auszahlung 7 (Schwelle p* = 0,778), die Zeile schickt eine von zwei Nachrichten. Die Auszahlungspaare der Gleichgewichte bleiben (9; 9) und (7; 7), "
                                            "ein informatives Gleichgewicht gibt es (Reaktionsdifferenz 0,778); gelernt wird bei zufälligem Start trotzdem (B, B): 99,8 % bei β = 20 (vorgerechnet: 4 Blöcke zu je 200 Läufen, 40.000 Schritte).",
    "Beispiel in der Art von Aumann": "Wie der Standard, aber Hase allein gegen Hirsch bringt 8 (Zahlen 9, 0, 8, 7; kein Zitat, die Quelle ist nicht im Volltext geprüft). Die Zeile hätte Grund zu lügen: es gibt kein "
                                      "informatives Gleichgewicht mehr, die Auszahlungspaare der Gleichgewichte bleiben dieselben.",
    "Kampf der Geschlechter: wer spricht?": "Die Zeile (bevorzugt A) schickt eine von drei Nachrichten. Bei zufälligem Start und β = 20 steigt ihre Auszahlung von 1,545 ohne Reden auf 1,999, die der Spalte fällt von 1,455 auf 1,000; "
                                            "die Summe bleibt 3 (vorgerechnet: 4 Blöcke zu je 200 Läufen, 40.000 Schritte).",
    "Altkonvention verlassen": "Alle starten bei B, die Schwelle der Hirschjagd ist p* = 0,4. Nach 40.000 Schritten ist (A, A) bei β = 20 und drei Nachrichten 100,0 %, bei zwei Nachrichten 24,4 %, "
                               "mit einer Nachricht und ohne Reden 0,0 % (vorgerechnet: 4 Blöcke zu je 200 Läufen, N = 20).",
    "Geheimer Handschlag": "Beide sprechen mit zwei Nachrichten, die Schwelle ist hoch (p* = 0,778), Start bei B, β = 100, 200.000 Schritte. Vorgerechnet liegt (A, A) bei N = 20 bei 18,4 % (β = 40: 3,8 %), "
                           "bei N = 10 bei 61,5 % und bei N = 40 bei 6,5 % (gleiche Schritte je Akteur); ohne Reden 0,0 %.",
    "Lewis: drei Zustände": "Drei Zustände, drei Signale, Roth-Erev-Lernen. Nach 100.000 Runden erreichen 87,6 % der Läufe den Bestwert (vorgerechnet: 4 Blöcke zu je 200 Läufen); "
                            "ein Fremder trifft mit 0,341, der Zufallswert 1/3 ist 0,333.",
}
