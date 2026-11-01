# Signale und Konventionen: ändert Reden, was sich durchsetzt? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-vorabsignale-demo.streamlit.app/)** -->

Siebtes Stück der Reihe **Koordinationsspiele** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Vehikel: **zwei Betriebe, die gleichzeitig ein Format wählen** (A: neu, am Hub bündeln; B: alt, direkt fahren), nachdem sie sich eine **kostenlose, unverbindliche Nachricht** schicken durften (Cheap Talk). Auszahlung ist Zeitersparnis je Sendung in willkürlichen Einheiten (Skala 0 bis 9). Gespielt werden reine Koordination, Hirschjagd (Hirsch gegen Hirsch 9, Hirsch allein 0, Hase 7, mit einstellbarer Verlockung t) und Kampf der Geschlechter; dazu das Lewis-Signalspiel, in dem eine Sprache erst aus dem Lernen entsteht.

Abgrenzung: `korreliert-demo` lässt einen **Vermittler** Handlungen empfehlen und prüft Gehorsam als Gleichgewichtsbedingung; hier gibt es keinen Vermittler, die Beteiligten reden selbst und jeder darf lügen (Kontrast, nicht Wiederholung). `nash-demo` rechnet Gleichgewichte kleiner Spiele ohne Reden. In der Reihe rechnet `gleichgewichtsauswahl-demo` (Stück 2) Logit-Lernen derselben Art ohne Reden und `fokalpunkt-demo` (Stück 6) lässt Fahrer ohne jedes Signal zusammenfinden. Lernregeln und Eigennutz sind **Modellannahmen**, keine Aussage über Menschen, Betriebe oder Sprachmodelle.

## Kernfrage

*Ändert ein kostenloses Wort vor dem Spiel, welches Gleichgewicht sich durchsetzt, und wann bekommt ein Wort eine Bedeutung?* Die Antwort dieser Demo hat drei Teile. (1) **Die Auszahlungen der Gleichgewichte ändert Reden in diesen vier Spielen nicht**, die Zahl der Gleichgewichte schon: die Auszahlungspaare aller Gleichgewichte sind mit einseitigem Reden dieselben wie ohne in 8 von 8 Zellen, die der reinen Gleichgewichte in 16 von 16, während die Zahl der reinen Gleichgewichte von 2 auf 6 bis 512 wächst (Hirschjagd: 2 / 2 / 6 / 15 / 320). (2) **Den Auswahlprozess ändert es**: eine Altkonvention (alle B, Schwelle p* = 0,4) verlassen nach 40.000 Schritten bei β = 20 und N = 20 100,0 % der Paarungen mit drei Nachrichten, 24,4 % mit zwei, ohne Reden und mit einer Nachricht 0,0 %; im Kampf der Geschlechter gewinnt der Sprecher (1,999 gegen 1,000 bei drei Nachrichten, 1,545 gegen 1,455 ohne Reden, Summe 3). (3) **Eine Sprache kann entstehen, mit Fallen**: im Lewis-Spiel erreichen nach 100.000 Runden 98,3 / 87,6 / 72,1 / 49,9 % der Läufe den Bestwert bei n = 2 / 3 / 4 / 6 Zuständen; ein Überschuss an Signalen heilt.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen sind in dieser Demo neu gemessen (`tests/test_claims.py`); Voraussetzungen stehen in der Tabelle. Die Messreihen (Zellen aus 4 Blöcken zu je 200 Läufen, N = 20 Akteure je Population, Logit-Lernen) sind vorgerechnet (`data/measurements.json` mit Erzeuger-Hash, `tools/make_measurements.py`), die Ereignisstudie zu H10 in `data/mechanism.json` (`tools/make_mechanism.py`); die Live-Experimente rechnen mit den Reglern der Seitenleiste. „Höher“, „niedriger“ und „gleich“ heißen: Unterschied zweier Mittel über 4 Standardfehlern der Blöcke und über 2 Prozentpunkten (`vor_evaluation.differs`).

| Nr. | Hypothese | Antwort | Beleg und Voraussetzungen |
|---|---|---|---|
| H1 | Einseitiges Reden ändert die Menge der Gleichgewichtsauszahlungen nicht (reine Koordination, Hirschjagd mit t = 0 und 1, Kampf der Geschlechter) | **Ja** | Auszahlungspaare **aller** Gleichgewichte (auch gemischter) sind mit einseitigem Reden (K = 1, 2) dieselben wie ohne Reden in 8 von 8 Zellen, die der reinen Gleichgewichte in 16 von 16 Zellen (alle vier Spiele × einseitig K = 1, 2, 3 und beidseitig K = 2); kein reines Gleichgewicht mit Fehlkoordination. Zahl der reinen Gleichgewichte (kein Reden / einseitig K = 1 / 2 / 3 / beidseitig K = 2): reine Koordination 2 / 2 / 8 / 24 / 512, Hirschjagd 2 / 2 / 6 / 15 / 320, Kampf der Geschlechter 2 / 2 / 6 / 15 / 256; beidseitig K = 3 (1.536 Strategien je Seite, nur gezählt): 1.179.648 / 626.688 / 294.912 reine Gleichgewichte mit denselben Auszahlungspaaren (`test_h1_*`) |
| H2 | Ein informatives Gleichgewicht gibt es bei reiner Koordination und Hirschjagd ohne Verlockung, nicht mit Verlockung und nicht im Kampf der Geschlechter | **Ja** | Einseitig, K = 2, exakt: Reaktionsdifferenz der Spalte 1,000 (reine Koordination) und 0,778 (Hirschjagd, b = 7); keines bei Hirschjagd mit t = 1 und im Kampf der Geschlechter. Bei b = 7 und t = 0 / 0,25 / 0,5 / 1 gibt es nur bei t = 0 eines; dasselbe bei b = 3,6 / 4,5 / 5,4 (`test_h2_*`) |
| H3 | Das beidseitige Zufallsprotokoll („gleiche Nachricht → A, sonst B“) ist ein Gleichgewicht, zahlt aber nur im Kampf der Geschlechter 1,5 / 1,5 und wird nicht gelernt | **Ja** | Gleichgewicht in 4 von 4 Spielen; Auszahlung (1; 1) / (8; 8) / (8; 8) / (1,5; 1,5) in reiner Koordination, Hirschjagd (t = 0 und 1), Kampf der Geschlechter; Summe beider Auszahlungen über der des schlechtesten Gleichgewichts ohne Reden in 4 von 4 Spielen, gleich der des besten in 2 von 4 (reine Koordination, Kampf der Geschlechter). Gelernt (Kampf der Geschlechter, beidseitig K = 2, zufälliger Start, β = 20, N = 20, 40.000 Schritte, 800 Läufe): im Mittel 5,5 % der Akteure spielen die Lotterie-Regel (gleichverteilte Wahl: 6,25 %), in 0 von 800 Läufen die Hälfte oder mehr (`test_h3_*`) |
| H4 | Reden löst die Hirschjagd bei zufälligem Start im Lernen nicht | **Ja** | (A, A) höchstens 3 % in 20 von 20 Zellen (Hirschjagd mit t = 0 und 1 × fünf Arten des Redens × β = 6 und 20, 40.000 Schritte); größter Wert 2,5 % (beidseitig K = 2, β = 6). Bei β = 20 (B, B) 100,0 / 100,0 / 99,8 / 98,6 / 99,7 % (kein Reden / einseitig K = 1 / 2 / 3 / beidseitig K = 2) (`test_h4_*`) |
| H5 | Reden öffnet eine Altkonvention einseitig nur bis etwa p* = 1/2, beidseitig auch darüber, und die Wirkung hängt an der Schärfe β | **Ja, mit Einschränkungen** | Tabelle unten. Über dem Spiel ohne Reden liegt (A, A) bei β = 20 in 6 von 12 Zellen (4 Arten des Redens × 3 Schwellen), bei β = 40 in 5 von 12. Einseitig K = 2 nur bei p* = 0,4 (24,4 %), einseitig K = 3 bei p* = 0,4 und 0,5 (100,0 / 40,6 %), beidseitig K = 2 bei allen drei (99,9 / 81,3 / 24,0 %). Von β = 20 auf 40 sinkt (A, A) in 5 Zellen, steigt in 1 (beidseitig bei p* = 0,6: 24,0 → 32,6 %) und bleibt gleich in 6. Einschränkungen: ein Lernmodell, N = 20, 40.000 Schritte (`test_h5_*`) |
| H6 | Der Sprecher gewinnt im Kampf der Geschlechter, die Summe bleibt gleich | **Ja** | Auszahlung Zeile / Spalte, zufälliger Start, β = 20: ohne Reden 1,545 / 1,455; einseitig K = 1 1,506 / 1,494; K = 2 1,898 / 1,101; K = 3 1,999 / 1,000; beidseitig 1,491 / 1,490. Start in der Altkonvention (Spalte bevorzugt): 1,000 / 2,000; 1,000 / 2,000; 1,245 / 1,753; 1,999 / 1,000; 1,054 / 1,938. Summe in allen 10 Zellen höchstens 0,018 neben 3; Kontrolle K = 1 gleicht dem Spiel ohne Reden (`test_h6_*`) |
| H7 | Mehr Strategien erhöhen bei festem β die Fehlkoordination | **Ja, gemessen; Ursache ungeprüft** | Hirschjagd, zufälliger Start, β = 6: Fehlkoordination 2,0 / 2,1 / 11,0 / 20,3 / 22,9 % (kein Reden / einseitig K = 1 / 2 / 3 / beidseitig K = 2); reine Koordination, Effizienz 0,995 / 0,995 / 0,945 / 0,906 / 0,870; bei β = 20 in der reinen Koordination 1,000 / 1,000 / 1,000 / 1,000 / 0,999. Ob die zusätzlichen gleichwertigen Strategien die Ursache sind, ist nicht getrennt geprüft; der Vergleich bei gleichem β benachteiligt das Spiel mit Reden (`test_h7_*`) |
| H8 | Lewis-Spiel: der Anteil der Läufe mit Signalsystem fällt mit n, ein Überschuss an Signalen heilt | **Ja** | Tabelle unten: nach 100.000 Runden 98,3 / 87,6 / 72,1 / 49,9 % bei n = ns = 2 / 3 / 4 / 6 (nach 10.000 Runden 95,5 / 80,6 / 58,1 / 29,5 %); n = 3 mit ns = 4 / 6: 98,4 / 100,0 %, n = 4 mit ns = 8: 99,9 %, n = 2 mit ns = 3: 100,0 %. Reine Gleichgewichte durch Vollaufzählung: n = 2: 6, n = 3: 87, n = 4: 2.056, davon n! Signalsysteme (`test_h8_*`, `test_lewis_*`) |
| H9 | Fremdspiel getrennt entstandener Signalsprachen ≈ 1/n | **Ja, mit einer Ausnahme zu benennen** | Fremdspiel nach 100.000 Runden bei n = ns = 2 / 3 / 4 / 6: 0,498 / 0,341 / 0,238 / 0,187 (1/n: 0,500 / 0,333 / 0,250 / 0,167); in 10 von 11 Zellen innerhalb von zwei Standardfehlern (plus 0,005) um 1/n, Ausnahme n = 4, ns = 3: 0,232 gegen 0,250. Bei n = 6 liegt es 2,1 Standardfehler über 1/n; diese Abweichung bleibt ungedeutet (`test_h9_*`) |
| H10 | Mechanismus: ungenutzte Nachrichten lassen den Hörer-Anteil auf etwa ½ treiben, ein Sprecher probiert A, sobald er über p* liegt | **Nein, in dieser Form nicht** | Die Momentaufnahmen widersprechen: bei den noch nicht ausgetretenen Läufen liegt der größte Hörer-Anteil in den 12 einseitigen Zellen im Mittel zwischen 0,000 und 0,107, die seltenste Nachricht hat im Mittel 2,2 bis 8,1 Sprecher. Die Ereignisstudie (jeder Schritt aufgezeichnet, 4 Zellen zu je 800 Läufen, 1.668 Austritte) zeigt im Rückblick: Die Nachricht, über die der Austritt läuft, wird vorher **selten** gesendet (100 Schritte vor dem Austritt von im Mittel 3,5 bis 5,9 Sprechern gegenüber 6,7 bis 10,0 im Mittel über alle Nachrichten), der Hörer-Anteil auf sie steigt von 0,02 bis 0,07 (200 Schritte vorher) über 0,22 bis 0,40 (50) auf 0,47 bis 0,59 (20 Schritte vorher), zusammen mit den ersten A-Sprechern. Die Formel für den Hörer-Anteil bei festen Sprechern stimmt mit der Simulation überein (Orakel, 6 Fälle) (`test_h10_*`) |
| H11 | Der Handschlag hängt an N und am Rauschen | **Ja für N und β, Mutationsrate nicht getrennt** | Hirschjagd mit p* = 7/9, beidseitig K = 2, Start in der Altkonvention, 10.000 Schritte je Akteur: (A, A) bei N = 10 / 20 / 40: 38,8 / 3,8 / 0,4 % (β = 40) und 61,5 / 18,4 / 6,5 % (β = 100); ohne Reden und mit einer Nachricht 0,0 % in 4 von 4 Zellen; mit Verlockung t = 1 bei N = 20: 0,0 % (β = 40) und 2,5 % (β = 100). Einseitig K = 3 bei 40.000 Schritten und β = 20: p* = 0,4: 99,9 / 100,0 / 48,1 % für N = 10 / 20 / 40, p* = 0,5: 99,8 / 40,6 / 0,0 % (hier sind die Schritte je Akteur nicht gleich). Das Modell hat keine eigene Mutationsrate, das Rauschen ist 1/β (`test_h11_*`) |

### E3: Anteil (A, A) nach 40.000 Schritten bei Start in der Altkonvention (β = 20 und β = 40)

| Reden | p* = 0,4 | p* = 0,5 | p* = 0,6 |
|---|---|---|---|
| kein Reden | 0,0 % ± 0,0 % | 0,0 % ± 0,0 % | 0,0 % ± 0,0 % |
| Zeile spricht, K = 1 | 0,0 % ± 0,0 % | 0,0 % ± 0,0 % | 0,0 % ± 0,0 % |
| Zeile spricht, K = 2 | 24,4 % ± 1,0 % | 0,1 % ± 0,1 % | 0,0 % ± 0,0 % |
| Zeile spricht, K = 3 | 100,0 % ± 0,0 % | 40,6 % ± 1,2 % | 1,0 % ± 0,4 % |
| beide sprechen, K = 2 | 99,9 % ± 0,0 % | 81,3 % ± 1,2 % | 24,0 % ± 1,5 % |

| Reden | p* = 0,4 | p* = 0,5 | p* = 0,6 |
|---|---|---|---|
| kein Reden | 0,0 % ± 0,0 % | 0,0 % ± 0,0 % | 0,0 % ± 0,0 % |
| Zeile spricht, K = 1 | 0,0 % ± 0,0 % | 0,0 % ± 0,0 % | 0,0 % ± 0,0 % |
| Zeile spricht, K = 2 | 0,5 % ± 0,0 % | 0,0 % ± 0,0 % | 0,0 % ± 0,0 % |
| Zeile spricht, K = 3 | 39,9 % ± 1,7 % | 3,4 % ± 0,6 % | 0,0 % ± 0,0 % |
| beide sprechen, K = 2 | 77,9 % ± 1,5 % | 54,4 % ± 2,3 % | 32,6 % ± 1,1 % |

### E3: Handschlag (Hirschjagd, b = 7, p* = 7/9, Start in der Altkonvention, 10.000 Schritte je Akteur)

| Verlockung t | Reden | β | N | Schritte | (A, A) | Läufe mit mindestens der Hälfte (A, A) | Anteil Handschlag |
|---|---|---|---|---|---|---|---|
| 0 | Zeile spricht, K = 1 | 40 | 20 | 200.000 | 0,0 % ± 0,0 % | 0,0 % ± 0,0 % | 0,0 % ± 0,0 % |
| 0 | Zeile spricht, K = 1 | 100 | 20 | 200.000 | 0,0 % ± 0,0 % | 0,0 % ± 0,0 % | 0,0 % ± 0,0 % |
| 0 | beide sprechen, K = 2 | 40 | 20 | 200.000 | 3,8 % ± 0,3 % | 3,8 % ± 0,3 % | 3,8 % ± 0,3 % |
| 0 | beide sprechen, K = 2 | 100 | 20 | 200.000 | 18,4 % ± 1,0 % | 18,4 % ± 1,0 % | 18,3 % ± 1,0 % |
| 1 | Zeile spricht, K = 1 | 40 | 20 | 200.000 | 0,0 % ± 0,0 % | 0,0 % ± 0,0 % | 0,0 % ± 0,0 % |
| 1 | Zeile spricht, K = 1 | 100 | 20 | 200.000 | 0,0 % ± 0,0 % | 0,0 % ± 0,0 % | 0,0 % ± 0,0 % |
| 1 | beide sprechen, K = 2 | 40 | 20 | 200.000 | 0,0 % ± 0,0 % | 0,0 % ± 0,0 % | 0,0 % ± 0,0 % |
| 1 | beide sprechen, K = 2 | 100 | 20 | 200.000 | 2,5 % ± 0,3 % | 2,5 % ± 0,3 % | 2,4 % ± 0,4 % |
| 0 | beide sprechen, K = 2 | 40 | 10 | 100.000 | 38,8 % ± 1,0 % | 38,9 % ± 1,0 % | 14,9 % ± 0,7 % |
| 0 | beide sprechen, K = 2 | 100 | 10 | 100.000 | 61,5 % ± 1,7 % | 61,5 % ± 1,7 % | 25,2 % ± 1,2 % |
| 0 | beide sprechen, K = 2 | 40 | 40 | 400.000 | 0,4 % ± 0,1 % | 0,4 % ± 0,1 % | 0,4 % ± 0,1 % |
| 0 | beide sprechen, K = 2 | 100 | 40 | 400.000 | 6,5 % ± 1,0 % | 6,5 % ± 1,0 % | 6,5 % ± 1,0 % |

### E4: Kampf der Geschlechter, zufälliger Start, β = 20, N = 20, 40.000 Schritte

| Reden | Auszahlung Zeile | Auszahlung Spalte | Summe | Fehlkoordination | Information (Bit) |
|---|---|---|---|---|---|
| kein Reden | 1,545 ± 0,016 | 1,455 ± 0,016 | 3,000 | 0,0 % ± 0,0 % | 0,000 ± 0,000 |
| Zeile spricht, K = 1 | 1,506 ± 0,020 | 1,494 ± 0,020 | 3,000 | 0,0 % ± 0,0 % | 0,000 ± 0,000 |
| Zeile spricht, K = 2 | 1,898 ± 0,007 | 1,101 ± 0,006 | 2,999 | 0,0 % ± 0,0 % | 0,001 ± 0,000 |
| Zeile spricht, K = 3 | 1,999 ± 0,000 | 1,000 ± 0,000 | 2,999 | 0,0 % ± 0,0 % | 0,001 ± 0,001 |
| beide sprechen, K = 2 | 1,491 ± 0,014 | 1,490 ± 0,014 | 2,982 | 0,6 % ± 0,1 % | 0,024 ± 0,003 |

### E6: Lewis-Spiel nach 100.000 Runden (4 Blöcke zu je 200 Läufen)

| Zustände n | Signale ns | Bestwert min(1, ns/n) | Effizienz | Läufe am Bestwert | Fremdspiel | Zufallswert 1/n |
|---|---|---|---|---|---|---|
| 2 | 2 | 1,000 | 0,995 ± 0,002 | 98,2 % ± 1,1 % | 0,498 ± 0,008 | 0,500 |
| 2 | 3 | 1,000 | 1,000 ± 0,000 | 100,0 % ± 0,0 % | 0,498 ± 0,013 | 0,500 |
| 3 | 2 | 0,667 | 0,666 ± 0,000 | 100,0 % ± 0,0 % | 0,337 ± 0,021 | 0,333 |
| 3 | 3 | 1,000 | 0,963 ± 0,002 | 87,6 % ± 0,7 % | 0,341 ± 0,010 | 0,333 |
| 3 | 4 | 1,000 | 0,994 ± 0,002 | 98,4 % ± 0,7 % | 0,331 ± 0,004 | 0,333 |
| 3 | 6 | 1,000 | 0,999 ± 0,000 | 100,0 % ± 0,0 % | 0,321 ± 0,008 | 0,333 |
| 4 | 2 | 0,500 | 0,500 ± 0,000 | 100,0 % ± 0,0 % | 0,256 ± 0,006 | 0,250 |
| 4 | 3 | 0,750 | 0,746 ± 0,000 | 98,1 % ± 0,4 % | 0,232 ± 0,005 | 0,250 |
| 4 | 4 | 1,000 | 0,934 ± 0,003 | 72,1 % ± 1,2 % | 0,238 ± 0,011 | 0,250 |
| 4 | 8 | 1,000 | 0,998 ± 0,000 | 99,9 % ± 0,1 % | 0,259 ± 0,006 | 0,250 |
| 6 | 6 | 1,000 | 0,913 ± 0,002 | 49,9 % ± 1,0 % | 0,187 ± 0,010 | 0,167 |

### E2: Verlockung bei b = 7, einseitiges Reden mit K = 2 (exakt)

| Verlockung t | Schwelle p* = b / (9 − t) | Auszahlungspaare aller Gleichgewichte | Informatives Gleichgewicht |
|---|---|---|---|
| 0 | 0,778 | (9; 9), (7; 7) | ja (Reaktionsdifferenz 0,778) |
| 0,25 | 0,8 | (9; 9), (7,2; 7,2), (7; 7) | nein |
| 0,5 | 0,824 | (9; 9), (7,412; 7,412), (7; 7) | nein |
| 1 | 0,875 | (9; 9), (7,875; 7,875), (7; 7) | nein |

### Gleichgewichte des Lewis-Spiels (Vollaufzählung)

| Zustände n = Signale | reine Gleichgewichte | Signalsysteme (Effizienz 1) | n! | nach Effizienz |
|---|---|---|---|---|
| 2 | 6 | 2 | 2 | 2 mit 1; 4 mit 0,5 |
| 3 | 87 | 6 | 6 | 6 mit 1; 72 mit 0,667; 9 mit 0,333 |
| 4 | 2.056 | 24 | 24 | 24 mit 1; 864 mit 0,75; 1.152 mit 0,5; 16 mit 0,25 |

## Befunde und Korrekturen gegenüber dem Plan

- **H10 gilt nicht wie geplant.** Der Plan (Stand 2026-10-27) vermutete, ungenutzte Nachrichten ließen die Antwortregeln neutral auf einen Hörer-Anteil von etwa ½ treiben. Gemessen ist das Gegenteil: die seltenste Nachricht hat bei den noch nicht ausgetretenen Läufen im Mittel 2,2 bis 8,1 Sprecher, und der Hörer-Anteil steht weit unter p*. Im Rückblick auf den Austritt ist die Austrittsnachricht vorher **selten gesendet** (H10). Die Momentaufnahmen allein hätten den Weg nicht gezeigt: an den 41 Stützstellen je Lauf liegt der Hörer-Anteil nie über p* (größter Anteil der Zeitpunkte 0,3 %), erst die Aufzeichnung jedes Schritts zeigt die kurze Episode.
- **Zahlen mit anderem Zufallsstrom:** die Wegwerfmessung des Plans nannte für den Austritt bei p* = 0,4 und β = 20 einseitig K = 2 / K = 3 / beidseitig 0,253 / 1,000 / 1,000 (hier 0,244 / 1,000 / 0,999), bei β = 40 0,003 / 0,422 / 0,828 (hier 0,005 / 0,399 / 0,779), für den Handschlag bei β = 40 / 100 0,042 / 0,177 (hier 0,038 / 0,184), im Lewis-Spiel 0,986 / 0,874 / 0,740 / 0,504 (hier 0,983 / 0,876 / 0,721 / 0,499) und das Fremdspiel bei n = 6 0,199 (hier 0,187). Die Unterschiede liegen in der Größenordnung der Blockstreuung; die Wegwerfzahlen (vier Seeds) wurden nicht übernommen.
- **Lotterie wird nicht gelernt, aber das Mittel täuscht.** Der Plan schloss das aus 0,024 Bit Information; die Information einer Lotterie ist einzeln null und sagt darüber nichts. Gemessen ist der Anteil der Akteure mit der Lotterie-Regel (H3). Die mittlere Auszahlung 1,491 / 1,490 sieht der Lotterie (1,5 / 1,5) ähnlich, ist aber ein Mittel über Läufe mit verschiedenen Konventionen und sagt über die Lotterie nichts.
- **Aumanns Zahlen:** das Beispiel mit Hase allein gegen Hirsch 8 (Zahlen 9, 0, 8, 7) heißt in der App „in der Art von Aumann“ und ist **kein Zitat**; die Quelle ist nicht im Volltext gelesen.
- **Neu gegenüber dem Plan:** die Schwelle ist b / (9 − t) statt b / 9, sobald t > 0 (bei t = 1 und b = 7: 0,875, bei t = 0: 0,778); der Mechanismus-Teil (Formel für den Hörer-Anteil bei festen Sprechern, Ereignisstudie); die Abhängigkeit von N.
- **Ampel:** gelb-grün des Plans bleibt; die offenen Punkte H10 und H11 sind beantwortet (H10 verneint, H11 für N und β bejaht), offen bleiben die Literatur im Volltext und ein zweites Lernmodell.
- **Port und Name:** Port 9216 (TCP-Prüfung am 2026-11-01 für 9216 frei; 9208 und 9209 sind in `.claude/launch.json` belegt), Präfix `vor`, Linie `koordination`.

## Modell

**Spiele** (Aktion A = Hirsch bzw. Treffpunkt 1, B = Hase bzw. Treffpunkt 2): reine Koordination (gleiche Wahl 1); Hirschjagd (A/A 9, A allein 0, B gegen B `b`, B gegen A `b + t`), Schwelle p* = b / (9 − t); Kampf der Geschlechter (Zeile 2/1, Spalte 1/2, sonst 0). **Erweitertes Spiel:** erst Nachrichten, dann Aktionen; die Nachricht ändert keine Auszahlung. Einseitig sendet die Zeile eine von K Nachrichten und wählt eine Aktion (2K Strategien), die Spalte hat eine Regel „Nachricht → Aktion“ (2^K Strategien); beidseitig sendet jede Seite eine Nachricht und wählt eine Regel „(eigene, fremde Nachricht) → Aktion“ (K · 2^(K²) Strategien je Seite: 2 / 32 / 1.536 für K = 1 / 2 / 3).

**Gleichgewichte:** reine durch Aufzählung (`vor_equilibria.pure_ne`); alle, auch gemischte, bei kein Reden und einseitigem Reden mit K ≤ 2 über **Trägerpaare** in exakter Bruchrechnung (ein kleines Simplex-Verfahren mit Bland-Regel statt scipy, damit die App ohne scipy läuft): bei festen Trägern sind die Bedingungen an die Mischung der Spalte und der Zeile voneinander unabhängig, je Trägerpaar werden Wertebereich der Auszahlungen und, bei zwei Nachrichten, die größte Reaktionsdifferenz bestimmt. **Informativ** heißt: ein Gleichgewicht, in dem die Zeile beide Nachrichten mit positiver Wahrscheinlichkeit sendet und die Spalte verschieden reagiert. **Lotterie:** beidseitig gleichverteilte Nachricht, A genau bei gleichen Nachrichten.

**Lernen** (`vor_learning.py`, Logit-Wahl nach dem Muster von `gleichgewichtsauswahl-demo`, auf zwei Populationen übertragen, eigener Code): zwei Populationen (Zeile, Spalte) zu je N Akteuren, jeder hält eine reine Strategie des erweiterten Spiels. Je Schritt überprüft ein Akteur einer zufällig gewählten Population seine Strategie und wählt per Logit mit Schärfe β nach der erwarteten Auszahlung gegen die aktuelle Verteilung der anderen Population (Auszahlungen auf Höchstwert 1 skaliert; Gumbel-Max). Kennzahlen exakt aus den Besetzungen: Anteile der Paarungen (A, A), (B, B) und Fehlkoordination, mittlere Auszahlungen, Information der Nachricht in Bit. **Lewis-Spiel** (`vor_lewis.py`): n gleichwahrscheinliche Zustände, ns Signale, Erfolg zahlt 1, Roth-Erev-Urnen (Anfangsgewicht 1, Verstärkung 1 bei Erfolg); Effizienz = Σ P_s(s|k) · P_r(k|s) / n, Bestwert min(1, ns/n), Fremdspiel = Sprecher aus Lauf r gegen Hörer aus Lauf r + 1 (nur Paare, in denen beide den Bestwert erreicht haben). Alle Herleitungen sind eigene Rechnung, durch Handrechnung, zweite Rechenwege und Simulation gestützt, nicht gegen Literatur geprüft; Formeln im 📐-Expander der App.

## Methodik

- **Läufe:** alle Läufe eines Aufrufs laufen gleichzeitig (numpy über Läufe, Strategien); die Gleichverteilten kommen blockweise aus SplitMix64 (`vor_rng.py`, wörtliche Kopie aus `ab-test-demo`, Kopie-Test `tests/test_copies.py`). Die Reihenfolge der Zufallszahlen steht im Kopf von `vor_learning.py` und `vor_lewis.py`; die Blockgröße ändert die Zahlen nicht.
- **Zeitbudget:** die App schätzt die Rechenzeit und rechnet mit weniger Läufen, wenn sie 25 Sekunden überschreitet (z. B. beidseitig K = 2, 200.000 Schritte: 37 statt 200 Läufe); beidseitiges Reden mit K = 3 wird nur gezählt. Die Schätzung ist eine Größenordnung vom Entwicklungsrechner.
- **Zellen:** vier unabhängige Blöcke zu je 200 Läufen, Zufalls-Seeds aus der Zellenbezeichnung (FNV-1a); die Reihen sind bei gleichem Code Zahl für Zahl dieselben, auch bei anderer Zahl von Prozessen (die Zusammenstellung hängt nicht von der Reihenfolge der Ergebnisse ab, Test). Der Erzeuger-Hash deckt `vor_games`, `vor_learning`, `vor_lewis`, `vor_scenario`, `vor_experiments` und `vor_rng` ab.
- **Orakel:** Matrix des erweiterten Spiels gegen den Spielablauf in allen Paarungen; reine Gleichgewichte gegen eine Schleife über alle Abweichungen; Trägerpaare gegen `scipy.optimize.linprog` (Zahl der Trägerpaare, Wertebereiche, jede Lösung als Gleichgewicht geprüft); das Simplex-Verfahren in Brüchen gegen `linprog` auf 120 Zufallssystemen; Lotterie gegen alle 64 reinen Abweichungen (32 je Seite); Lewis-Gleichgewichtszahlen 6 / 87 / 2.056; Lewis-Effizienz gegen den Spielablauf; Lernen gegen eine Schleifenfassung mit denselben Zufallszahlen und gegen die exakte stationäre Verteilung der Markov-Kette bei N = 2; Hörer-Anteil gegen die Formel.
- **Keine Mutationsprüfung** (seit 2026-10-28 abgeschafft); die Fehler fängt der Bau selbst, die Orakel- und die Claims-Tests.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Lernregeln sind Modellannahmen** (Logit-Lernen zweier Populationen mit N Akteuren; Roth-Erev-Urnen im Lewis-Spiel) | Gemessen sind diese Regeln. Andere Lernregeln, Populationsgrößen oder Revisionsfolgen können andere Zahlen geben; über Menschen, Betriebe oder Sprachmodelle sagt die Demo nichts. | `gleichgewichtsauswahl-demo` (Logit ohne Reden) |
| **Zwei Populationen, ein Spiel** | Mehr als zwei Beteiligte, Netze und wiederholte Begegnungen mit wechselnden Partnern sind nicht gerechnet. | Stück 3 der Reihe (Netze) |
| **Keine Kosten, keine private Information** | Teure Signale (Signalisieren im engeren Sinn) und Information, die nur einer hat, sind nicht Teil des erweiterten Spiels; das Lewis-Spiel hat private Information (den Zustand), aber keine Interessenkonflikte. | – |
| **Nur eine Nachrichtenrunde** | Mehrere Runden Hin und Her, die den Kreis der Gleichgewichte weiter öffnen können, sind nicht gerechnet. | – |
| **Kein Vermittler** | Ein Vermittler, der jedem seine Handlung zuteilt, führt auf korrelierte Gleichgewichte; die Lotterie ist ein solches Gleichgewicht ohne Vermittler. | `korreliert-demo` |
| **Auszahlungen willkürlich skaliert** | Die Skala 0 bis 9 ist erfunden; das Beispiel „in der Art von Aumann“ verwendet die Zahlen 9, 0, 8, 7 nicht als Zitat, die Quelle ist nicht im Volltext geprüft. | – |
| **Stichproben sind endlich** | Zellen der vorgerechneten Reihen sind Mittel über 4 Blöcke zu je 200 Läufen mit N = 20 Akteuren; Live-Läufe haben weniger. Andere N verschieben die Zahlen (E3). | – |
| **Literatur nur nach Abstract** | Farrell (1987), Aumann (1990), Robson (1990), Kim und Sobel (1995), Wärneryd (1991), Lewis (1969), Skyrms (2010) und Huttegger u. a. (2010) sind nicht im Volltext gelesen; die Demo behauptet keine ihrer Aussagen, sondern rechnet das hier beschriebene Modell. | – |

Dazu: ein Lernmodell je Teil; Nachrichten sind in der Messung gleichwertig (kein Kontext, keine Kultur); die Ereignisstudie sieht nur Läufe, die austreten, und ihre Rückblicke sind beschreibend, kein Beweis der Ursache (ein Eingriffsversuch, der die Seltenheit der Nachricht erzwingt, fehlt); „Austritt“ heißt hier: erstmals spielt die Hälfte der Sprecher A. Der Hörer-Anteil an den Stützstellen ist eine Momentaufnahme; die kurze Episode vor dem Austritt zeigt erst die Aufzeichnung jedes Schritts.

## Literatur (nicht im Volltext gelesen)

Die Einordnung stützt sich auf Abstracts bzw. Suchauszüge des Einzelplans, **nichts davon ist im Volltext geprüft**; Seiten, Zeitschriften und genaue Aussagen sind deshalb nicht angegeben, und die Demo behauptet keine Aussage dieser Arbeiten.

- Farrell (1987): kostenloses Reden vor Koordinationsspielen. Nur Abstract.
- Aumann (1990): warum kostenloses Reden bei der Hirschjagd nicht überzeugt (Beispiel mit Verlockung; die Zahlen 9, 0, 8, 7 stammen aus dem Gedächtnis, nicht aus der Quelle). Nur Abstract.
- Robson (1990): der „geheime Handschlag“ in evolutionären Modellen. Nur Abstract.
- Kim und Sobel (1995), Wärneryd (1991): Cheap Talk in evolutionären Modellen. Nur Abstract.
- Lewis (1969), Skyrms (2010), Huttegger u. a. (2010): Signalspiele, Signalsysteme gegen Teil-Pooling. Nur Abstract bzw. Suchauszug.

## Tests

**257 Tests**, Laufzeit rund 8,5 Minuten seriell auf Windows mit Python 3.14 (den größten Teil brauchen die AppTest-Dateien `test_app.py` und `test_layout.py`; die Linux-CI mit Python 3.12 ist noch nicht gelaufen). Zusammensetzung:

- **Orakel (99, `test_oracle_vorabsignale.py`):** Matrix gegen Spielablauf, reine Gleichgewichte gegen Schleife, Trägerpaare gegen `scipy.optimize.linprog`, Simplex in Brüchen gegen `linprog`, Lotterie gegen alle 64 reinen Abweichungen, Lewis-Gleichgewichtszahlen und -Effizienz, Lernen gegen Schleifenfassung und exakte Markov-Kette, Information, Hörer-Anteil gegen Formel.
- **Zweig-Tests (12, `test_branches.py`):** K = 1 gleicht dem Spiel ohne Reden Zahl für Zahl, jede Art des Redens ändert eine Kennzahl, der Handschlag tritt bei β = 100 auf, jede Verlockung ändert die Existenz des informativen Gleichgewichts.
- **Claims (28, `test_claims.py`):** jede Zahl der README und der Preset-Hilfen; die README-Tabellen werden mit denselben Zeilenbauern der App erzeugt und im Test als Text in der README gesucht.
- **Messdateien (12 + 7, `test_measurements.py`, `test_mechanism.py`):** Erzeuger-Hash, Achsen, Gestalt, Nullspalten, Reproduktion einzelner Zellen mit anderen Seeds innerhalb einer Marge aus den Standardfehlern.
- **Einheiten:** Randfälle (12), Verdikte und Formate an Handwerten (13), Presets (6), Kopie des Zufallsgenerators (5), Laufzeit-Importe (1).
- **Oberfläche (35 + 25 + 2):** AppTest-Rauchtest (jeder Knopf, jeder Regler an beiden Rändern, Eckfall-Matrix über Spiel × Reden × Rand, jeder ▶️-Knopf einzeln, jedes Preset), Layout (fester Kopf, Schnellstart, Hervorhebung, jeder Regler hebt sie auf), Footer.
- Die Lernzahlen stehen nirgends exakt in einem Test der Rechnung, sondern als Bänder (Standardfehler, feste Marge) oder aus den gespeicherten Messdateien.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Hauptansicht, sechs Experimente, Grenzen, Formeln, PDF |
| `data/measurements.json` | vorgerechnete Messreihen (Erzeuger-Hash) |
| `data/mechanism.json` | Ereignisstudie zu H10 und Lotterie-Studie (Erzeuger-Hash) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (nur Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tools/make_measurements.py` | rechnet `data/measurements.json` neu |
| `tools/make_mechanism.py` | rechnet `data/mechanism.json` neu |
| `vor_budget.py` | Rechenzeit-Schätzung und Begrenzung der Läufe |
| `vor_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets mit Hilfetexten |
| `vor_equilibria.py` | reine Gleichgewichte, Trägerpaare in Brüchen (Simplex), informativ, Lotterie |
| `vor_evaluation.py` | Kennzahlen, Verdikte als reine Funktionen, Zeilenbauer der Tabellen |
| `vor_experiments.py` | Zellen, Aufträge, Messdatei, Formel für den Hörer-Anteil |
| `vor_games.py` | Spiele, erweitertes Spiel, Spielablauf, Altkonvention |
| `vor_layout.py` | fester Kopf (Stylesheet) |
| `vor_learning.py` | Logit-Lernen zweier Populationen, Information der Nachricht |
| `vor_lewis.py` | Lewis-Spiel: Roth-Erev, Effizienz, Gleichgewichte, Fremdspiel |
| `vor_mechanism.py` | Ereignisstudie zum Austritt aus der Altkonvention |
| `vor_pdf_export.py` | PDF-Export (fpdf2), erst nach Klick |
| `vor_presets.py` | Permalink, Presets (Schnellstart oben in der Seitenleiste, aktives hervorgehoben), Zufalls-Seed-Button |
| `vor_rng.py` | SplitMix64 (wörtliche Kopie aus `ab-test-demo`) |
| `vor_scenario.py` | Einstellungen und abgeleitete Größen |
| `vor_texts.py` | feste Texte der App (auch in der README) |
| `vor_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tests/` | Orakel, Zweig-Tests, Claims, Messdateien, Presets, Layout, AppTest-Smoke-Test, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Vermittler:** ein Dritter, der Handlungen zuteilt, führt auf korrelierte Gleichgewichte (`korreliert-demo`); hier reden die Beteiligten selbst.
- **Mehrere Nachrichtenrunden:** nur eine Runde (gleichzeitig oder einseitig); längere Gespräche öffnen den Kreis der Gleichgewichte weiter.
- **Private Information und Signalisieren mit Kosten:** das erweiterte Spiel hat keine private Information und keine kostenpflichtigen Signale; das Lewis-Spiel hat private Information, aber keinen Interessenkonflikt.
- **Netze und komplexe Ansteckung:** Reden in Netzen mit lokalen Nachbarn gehört in Stück 3 der Reihe.
- **Menschenexperimente und Sprachmodelle:** die Demo sagt nichts über Menschen oder Sprachmodelle.
- **Beidseitig mit K = 3 lernen:** 1.536 Strategien je Seite (2,4 Millionen Paare); dort wird nur gezählt.
- **Alle Gleichgewichte bei K = 3 und beidseitig:** die Trägerpaar-Rechnung ist auf höchstens 6 Strategien je Seite ausgelegt; dort sind nur die reinen Gleichgewichte gerechnet.
- **Ein zweites Lernmodell** (Replikator, Q-Lernen) und **eine eigene Mutationsrate** neben β.
- **Eingriffsversuch zu H10:** die Seltenheit einer Nachricht erzwingen und messen, ob der Austritt dann häufiger wird.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Python 3.12 oder neuer; die in der Streamlit Community Cloud eingestellte Python-Version ist beim Deploy zu notieren.

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
