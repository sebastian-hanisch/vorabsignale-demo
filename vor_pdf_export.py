"""PDF-Export (fpdf2, Kernschriften): Einstellungen, Kennzahlen und die Verdikte. Erst nach Klick erzeugt (DEMO-PLAYBOOK §1c)."""

from __future__ import annotations

from fpdf import FPDF

import vor_constants as C
import vor_evaluation as V

_REPLACE = {"–": "-", "—": "-", "≤": "<=", "≥": ">=", "≠": "!=", "√": "Wurzel", "²": "2", "Σ": "Summe", "°": " Grad", "·": "*", "−": "-", "„": '"', "“": '"', "∞": "unendlich",
            "→": "->", "ε": "eps", "β": "beta", "α": "alpha", "θ": "theta", "λ": "lambda", "σ": "sigma", "π": "pi", "δ": "delta", "η": "eta", "Δ": "Delta", "×": "x",
            "≈": "~", "½": "1/2", "…": "...", "‚": ",", "‘": "'", "’": "'", "”": '"', "µ": "mu", "∈": "in", "ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss",
            "⁻": "^-", "¹": "1", "₀": "0", "₁": "1", "₂": "2", "ₘ": "m", "ₛ": "s"}


def _latin(text: str) -> str:
    """Kernschriften kennen nur Latin-1: Sonderzeichen und Umlaute ersetzen, Markdown-Sterne entfernen."""
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    return text.replace("**", "").replace("`", "").encode("latin-1", "replace").decode("latin-1")


def build_pdf(view, verdicts: list[tuple[str, str]]) -> bytes:
    """`view`: V.MainView (Reden) oder V.LewisView (Lewis-Signalspiel)."""
    s = view.settings
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _latin("Signale und Konventionen: aendert Reden, was sich durchsetzt?"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    if s.part == C.PART_LEWIS:
        pdf.multi_cell(0, 5, _latin(f"Lewis-Signalspiel; {s.lewis_n} Zustaende, {s.lewis_ns} Signale, {s.lewis_rounds} Runden, {view.runs} Laeufe; Zufalls-Seed {s.seed}."), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Kennzahlen", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        cross, pairs = view.cross()
        pdf.multi_cell(0, 5, _latin(f"Mittlere Effizienz am Ende: {V.de(float(view.last.efficiency.mean()), 3)}; Bestwert min(1, ns/n): {V.de(view.best, 3)}; Laeufe mit mindestens 95 % des Bestwerts: "
                                    f"{V.pct(float(view.reached.mean()), 1)}; Fremdspiel: {'nicht definiert' if cross is None else V.de(cross, 3)} ({pairs} Paare)."), new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.multi_cell(0, 5, _latin(f"{s.game}; Hase-Auszahlung b = {V.de(s.b, 1)}; Verlockung t = {V.de(s.t, 2)}; Reden: {V.talk_label(s.talk, s.K)}; Schaerfe beta = {V.de(s.beta, 0)}; "
                                    f"N = {s.n_agents} Akteure je Population; Start: {s.start}; {s.steps} Schritte; {view.runs_used} Laeufe; Zufalls-Seed {s.seed}."), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Kennzahlen am Ende", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        if view.runs_used:
            last = view.last
            pdf.multi_cell(0, 5, _latin(f"(A, A): {V.pct(float(last.aa.mean()), 1)}; (B, B): {V.pct(float(last.bb.mean()), 1)}; Fehlkoordination: {V.pct(float(last.mis.mean()), 1)}; "
                                        f"Auszahlung Zeile {V.de(float(last.payoff1.mean()), 3)}, Spalte {V.de(float(last.payoff2.mean()), 3)}; Information der Nachricht: {V.de(float(last.info.mean()), 3)} Bit."),
                           new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.multi_cell(0, 5, "Nicht gelernt (beidseitiges Reden mit K = 3 hat 1536 Strategien je Seite); nur die Gleichgewichte wurden gezaehlt.", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Verdikte", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for _, text in verdicts:
        pdf.multi_cell(0, 5, _latin(text), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _latin("Lernregeln (Logit-Lernen zweier Populationen, Roth-Erev-Lernen) und Eigennutz sind Modellannahmen, keine Aussagen ueber Menschen, Betriebe oder Sprachmodelle. "
                                "Alle Zahlen sind eigene Rechnung, durch Handrechnung, zweite Rechenwege und Simulation gestuetzt."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
