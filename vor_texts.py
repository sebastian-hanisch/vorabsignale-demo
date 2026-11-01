"""Feste Texte der App, die auch in der README stehen (ein Test vergleicht beide Fassungen, DEMO-PLAYBOOK §5). Zahlen aus Messungen stehen hier nicht, sie kommen aus der Rechnung (vor_evaluation)."""

INTRO = """
**Siebtes Stück der Reihe Koordinationsspiele.** Zwei Betriebe wählen gleichzeitig ein Format: **A** (neu, am Hub bündeln) oder **B** (alt, direkt fahren). Vorher dürfen sie sich eine **kostenlose
Nachricht** schicken: sie kostet nichts, bindet nicht und darf gelogen sein (**Cheap Talk**). Die Frage: **Ändert ein Wort vor dem Spiel, welches Gleichgewicht sich durchsetzt, und wann bekommt ein Wort
überhaupt eine Bedeutung?** Gezeigt werden (a) das um Nachrichten erweiterte Spiel samt seinen Gleichgewichten, (b) warum eine Verlockung ein Wort unglaubwürdig macht, (c) ob Reden eine
**Altkonvention** öffnen kann, (d) wer im Kampf der Geschlechter vom Reden profitiert und (e) das **Lewis-Signalspiel**, in dem Wörter erst aus dem Lernen entstehen.

Abgrenzung: `korreliert-demo` lässt einen **Vermittler** Handlungen empfehlen; hier gibt es keinen, die Beteiligten reden selbst und jeder darf lügen. `nash-demo` rechnet Gleichgewichte ohne Reden.
In der Reihe rechnet `gleichgewichtsauswahl-demo` (Stück 2) Logit-Lernen derselben Art ohne Reden, `fokalpunkt-demo` (Stück 6) lässt Fahrer ohne jedes Signal zusammenfinden.
**Lernregeln und Eigennutz sind Modellannahmen**, keine Aussage über Menschen, Betriebe oder Sprachmodelle; die Auszahlungen sind willkürlich skaliert.

Weiter unten: sechs Tabs mit je einem **🔬 Experiment**, die Abschnitte **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""

TALK_EXPLAINER = r"""
1. **Zwei Stufen.** Erst werden Nachrichten gesendet, dann wählen beide gleichzeitig A oder B. Die Nachricht ändert keine Auszahlung (**kostenlos**) und bindet nicht (**Lügen erlaubt**).
2. **Strategien im erweiterten Spiel.** Was ein Beteiligter tut, ist eine ganze Vorschrift, nicht eine einzelne Aktion:

| | Einseitig (die Zeile spricht) | Beidseitig (beide sprechen gleichzeitig) |
|---|---|---|
| **Zeile** | (Nachricht $m$, Aktion $a$) | (eigene Nachricht, **Regel**: (eigene, fremde Nachricht) → Aktion) |
| **Spalte** | **Regel**: Nachricht → Aktion | wie die Zeile |
| **Strategien je Seite** (K = 2) | 4 und 4 | 32 und 32 |

3. **Gleichgewichte.** Ignorieren alle die Nachrichten, bleibt jedes Gleichgewicht des Spiels ohne Reden eines. Neu sind Gleichgewichte, in denen die Nachricht etwas **bedeutet**: die Spalte antwortet auf
   Nachricht 1 anders als auf Nachricht 2 (**informatives Gleichgewicht**). Ob es so eines gibt, zeigen E1 und E2 für vier Spiele; eine Verlockung (E2) ist ein Grund zum Lügen.
4. **Lotterie.** Beide senden zufällig eine Nachricht; sind die Nachrichten gleich, spielen beide A, sonst beide B. Das ist ein Gleichgewicht, das aus dem Reden entsteht, ohne dass ein Vermittler
   die Zufallszahl zieht; die Nachricht trägt einzeln keine Information, nur das Paar entscheidet (E5).
5. **Lernen.** Zwei Populationen (Zeile, Spalte) mit je $N$ Akteuren; ein Akteur nach dem anderen überprüft seine Strategie und wählt per Logit mit Schärfe $\beta$ nach den erwarteten Auszahlungen
   (E3, E4). Im **Lewis-Signalspiel** lernt eine Urne je Zustand und je Signal aus Erfolg (E6).
"""

CONTROLS_TABLE = """
| Regler | Was er tut | Wirkung auf die Frage |
|---|---|---|
| **Teil** | Kostenloses Reden vor dem Spiel oder Lewis-Signalspiel. | Schaltet die Hauptansicht um; die Regler des anderen Teils sind dann ausgegraut. |
| **Spiel** | Reine Koordination, Hirschjagd oder Kampf der Geschlechter. | Hirschjagd: zwei Gleichgewichte mit verschiedenem Risiko; Kampf der Geschlechter: zwei Gleichgewichte, die jede Seite anders bewertet. |
| **Hase-Auszahlung b** | Auszahlung von B gegen B in der Hirschjagd; Hirsch gegen Hirsch bringt 9, Hirsch allein 0. | Legt die Schwelle p* = b / (9 − t) fest: so groß muss der Anteil der Hirsch-Jäger beim Partner sein, damit Hirsch sich lohnt. |
| **Verlockung t** | Hase allein gegen Hirsch bringt b + t. | Gibt der Zeile einen Grund zum Lügen (E2). |
| **Reden** | Keins, nur die Zeile spricht, beide sprechen gleichzeitig. | Ändert die Strategienmenge, nicht die Auszahlungen. |
| **Nachrichten K** | Zahl der möglichen Nachrichten. K = 1 ist die Kontrolle: das Reden hat dann keinen Inhalt. | Mehr Nachrichten heißt mehr gleichwertige Strategien. |
| **Schärfe β** | Wie genau ein Akteur die beste Strategie wählt (Auszahlungen auf den Höchstwert 1 skaliert). Klein: viel Rauschen. | E3 vergleicht β = 20 und β = 40. |
| **Akteure N** | Größe jeder der zwei Populationen. | E3 und der Handschlag vergleichen N = 10, 20 und 40. |
| **Start** | Zufällige Verteilung oder alle in der Altkonvention „alle B“. | Nur aus der Altkonvention lässt sich zeigen, ob Reden sie öffnet. |
| **Schritte, Läufe** | Länge und Zahl der unabhängigen Lernläufe. | Die App kürzt die Läufe, wenn die geschätzte Rechenzeit zu lang wird. |
| **Zustände n, Signale, Runden** | Lewis-Signalspiel: n Zustände, ns Signale, Runden des Lernens. | Mit weniger Signalen als Zuständen sind nicht alle Zustände unterscheidbar. |
"""

ASSUMPTIONS_MARKDOWN = """
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
"""

RELATED = """
- **`korreliert-demo`**: ein Vermittler empfiehlt Handlungen und Gehorsam ist die Gleichgewichtsbedingung; hier reden die Beteiligten selbst (Kontrast).
- **`nash-demo`**: Gleichgewichte kleiner Spiele ohne Reden.
- **`gleichgewichtsauswahl-demo`** (Stück 2 der Reihe): Logit-Lernen derselben Art ohne Reden; welche Konvention das Rauschen wählt.
- **`fokalpunkt-demo`** (Stück 6): Koordination ohne jedes Signal; dort wird gemessen, wie oft sich zwei getrennt entstandene Konventionen treffen.
"""

MATH_MARKDOWN = r"""
**Lesehilfe.** Gegeben sind ein 2×2-Spiel mit Auszahlungen $u_1, u_2$ für Aktionen $a \in \{A, B\}$, eine Art des Redens mit $K$ Nachrichten und zwei Populationen aus je $N$ Akteuren; gesucht sind die
Gleichgewichte des erweiterten Spiels, ob eines davon informativ ist, und was das Logit-Lernen auf Dauer auswählt. Herleitungen sind eigene Rechnung, durch Handrechnung, zweite Rechenwege und Simulation
gestützt, nicht gegen Literatur geprüft.

| Symbol | Bedeutung |
|---|---|
| $u_i(a_1, a_2)$ | Auszahlung von Beteiligtem $i$ bei den Aktionen $a_1$ (Zeile), $a_2$ (Spalte) |
| $b,\ t,\ p^*$ | Hase-Auszahlung, Verlockung, Schwelle $p^* = b/(9-t)$ der Hirschjagd |
| $m,\ r$ | Nachricht und Antwortregel (Nachricht → Aktion) |
| $x,\ y$ | gemischte Strategie von Zeile und Spalte im erweiterten Spiel; $U_1, U_2$ die Auszahlungsmatrizen über alle Strategienpaare |
| $\beta,\ N$ | Schärfe der Logit-Wahl, Akteure je Population |
| $s_m$ | Anteil der Hörer, die auf Nachricht $m$ mit A antworten |
| $n,\ n_s$ | Zahl der Zustände und Signale im Lewis-Spiel |

**Gleichgewicht** des erweiterten Spiels: kein reiner Wechsel verbessert die Auszahlung (für die Zeile $x$ über ihre Strategien $i$, für die Spalte $y$ über ihre Strategien $j$):

$$U_1 y \le (x^{\top} U_1\, y)\,\mathbf 1, \qquad x^{\top} U_2 \le (x^{\top} U_2\, y)\,\mathbf 1^{\top}$$

**Schwelle der Hirschjagd.** Spielt der Partner A mit Wahrscheinlichkeit $p$, ist A genau dann mindestens so gut wie B, wenn

$$9\,p \ \ge\ b + t\,p \quad\Longleftrightarrow\quad p \ \ge\ p^* = \frac{b}{9-t}$$

**Informatives Gleichgewicht** (einseitig, zwei Nachrichten): die Reaktionsdifferenz der Spalte, größter Wert über alle Gleichgewichte, in denen die Zeile beide Nachrichten mit positiver Wahrscheinlichkeit sendet:

$$\Delta = \bigl|\, \Pr(\text{Spalte } A \mid m=1) - \Pr(\text{Spalte } A \mid m=2) \,\bigr|$$

**Lotterie** (beidseitig, $K$ gleichverteilte Nachrichten): beide spielen $A$ genau bei gleichen Nachrichten, die Auszahlung ist eine Mischung der Gleichgewichte des Spiels ohne Reden:

$$u_i^{\text{Lotterie}} = \tfrac{1}{K}\, u_i(A, A) + \bigl(1 - \tfrac{1}{K}\bigr)\, u_i(B, B)$$

**Logit-Lernen.** Der überprüfende Akteur wählt die Strategie $s$ mit der Wahrscheinlichkeit (Auszahlungen $\tilde u$ auf Höchstwert 1 skaliert, $y$ die Verteilung der anderen Population):

$$\Pr(s) = \frac{e^{\beta\, (\tilde U y)_s}}{\sum_{s'} e^{\beta\, (\tilde U y)_{s'}}}$$

**Hörer-Anteil auf eine Nachricht** (bei festen Sprechern; $n$ Sprecher senden $m$, davon $a$ mit A, $t = 0$): der Nutzen einer Hörerregel ist je Nachricht additiv, die Logit-Wahl faktorisiert also:

$$s_m = \frac{1}{1 + e^{-\beta\,(a - p^*\, n)/N}}$$

**Lewis-Spiel.** Sprecherabbildung $P_s(\text{Signal} \mid \text{Zustand})$, Hörerabbildung $P_r(\text{Handlung} \mid \text{Signal})$; Erfolgswahrscheinlichkeit und Bestwert:

$$E = \frac{1}{n} \sum_{k=1}^{n} \sum_{s=1}^{n_s} P_s(s \mid k)\, P_r(k \mid s), \qquad E_{\max} = \min\!\Bigl(1, \frac{n_s}{n}\Bigr)$$

**Roth-Erev-Lernen.** Jede Urne $w$ (je Zustand für den Sprecher, je Signal für den Hörer) startet mit Gewicht 1 je Eintrag; bei Erfolg wächst das Gewicht des gezogenen Eintrags um 1:

$$P(\text{Eintrag } j) = \frac{w_j}{\sum_{j'} w_{j'}}, \qquad w_j \leftarrow w_j + 1 \text{ bei Erfolg}$$

**Literatur.** Die Quellen stehen in der README mit Stand der Prüfung; Formeln und Zahlen dieser Demo sind eigene Rechnung.
"""
