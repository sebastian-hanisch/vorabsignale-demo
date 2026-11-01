"""Plotly-Figuren (alle Achsen fest, fixedrange). Nur Darstellung: jede Zahl kommt fertig aus Kern, Auswertung oder Messreihe."""

from __future__ import annotations

import plotly.graph_objects as go

import vor_constants as C
import vor_evaluation as V


def _base(fig: go.Figure, title: str | None = None, height: int = 340) -> go.Figure:
    fig.update_layout(title=title, height=height, margin=dict(l=55, r=15, t=45 if title else 15, b=55), legend=dict(orientation="h", y=-0.3, x=0))
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


# --- Hauptansicht Reden -----------------------------------------------------------------------------------------------------------------------
def build_trace(view: V.MainView) -> go.Figure:
    """Verlauf des Lernens: Mittel über die Läufe der Anteile (A, A), (B, B) und Fehlkoordination über die Schritte; dünn die ersten drei Läufe für (A, A)."""
    steps = view.steps
    fig = go.Figure()
    for lab, attr, col in (("(A, A)", "aa", C.COLOR_A), ("(B, B)", "bb", C.COLOR_B), ("Fehlkoordination", "mis", C.COLOR_MIS)):
        fig.add_trace(go.Scatter(x=steps, y=[getattr(s, attr).mean() for s in view.snaps], mode="lines", name=f"{lab}, Mittel", line=dict(color=col, width=3)))
    for r in range(min(3, view.runs_used)):
        fig.add_trace(go.Scatter(x=steps, y=[s.aa[r] for s in view.snaps], mode="lines", name=f"(A, A), Lauf {r + 1}", line=dict(color=C.COLOR_A, width=1, dash="dot"), opacity=0.6,
                                 showlegend=r == 0))
    fig.update_xaxes(title="Schritte")
    fig.update_yaxes(title="Anteil der Paarungen", range=[-0.02, 1.02])
    return _base(fig)


def build_info(view: V.MainView) -> go.Figure:
    """Information der Nachricht in Bit über die Schritte (Mittel über die Läufe)."""
    fig = go.Figure(go.Scatter(x=view.steps, y=[s.info.mean() for s in view.snaps], mode="lines", name="Information der Nachricht", line=dict(color=C.COLOR_LINE, width=3)))
    fig.update_xaxes(title="Schritte")
    fig.update_yaxes(title="Information (Bit)", range=[0, max(0.1, max(s.info.mean() for s in view.snaps) * 1.3)])
    return _base(fig, height=260)


def build_payoffs(view: V.MainView) -> go.Figure:
    """Mittlere Auszahlung von Zeile und Spalte über die Schritte."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view.steps, y=[s.payoff1.mean() for s in view.snaps], mode="lines", name="Zeile", line=dict(color=C.COLOR_A, width=3)))
    fig.add_trace(go.Scatter(x=view.steps, y=[s.payoff2.mean() for s in view.snaps], mode="lines", name="Spalte", line=dict(color=C.COLOR_B, width=3)))
    fig.update_xaxes(title="Schritte")
    fig.update_yaxes(title="mittlere Auszahlung")
    return _base(fig, height=260)


# --- Experimente ------------------------------------------------------------------------------------------------------------------------------
def build_bars(categories: list[str], series: dict[str, list[float]], ytitle: str, errors: dict[str, list[float]] | None = None, yrange=None, colors=None, height: int = 340) -> go.Figure:
    """Gruppierte Balken: je Serie eine Farbe, optional Fehlerbalken (Standardfehler)."""
    fig = go.Figure()
    palette = colors or [C.COLOR_A, C.COLOR_B, C.COLOR_LINE, C.COLOR_MIS, C.COLOR_GREY]
    for i, (name, ys) in enumerate(series.items()):
        err = dict(type="data", array=errors[name], visible=True) if errors and name in errors else None
        fig.add_trace(go.Bar(x=categories, y=ys, name=name, marker_color=palette[i % len(palette)], error_y=err))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title=ytitle, range=yrange)
    fig.update_xaxes(type="category")
    return _base(fig, height=height)


def build_lines(series: dict[str, tuple], xtitle: str, ytitle: str, yrange=None, hline: float | None = None, hline_text: str = "", categorical: bool = False, height: int = 340) -> go.Figure:
    """Linien mit Punkten: series = {Name: (xs, ys)} oder {Name: (xs, ys, ses)} mit Standardfehlern als Fehlerbalken."""
    fig = go.Figure()
    palette = [C.COLOR_A, C.COLOR_B, C.COLOR_LINE, C.COLOR_MIS, "#9467bd", "#8c564b", "#7f7f7f"]
    for i, (name, vals) in enumerate(series.items()):
        xs, ys = vals[0], vals[1]
        err = dict(type="data", array=list(vals[2]), visible=True) if len(vals) > 2 else None
        fig.add_trace(go.Scatter(x=list(xs), y=list(ys), mode="lines+markers", name=name, line=dict(color=palette[i % len(palette)], width=2.5), error_y=err))
    if hline is not None:
        fig.add_hline(y=hline, line=dict(color=C.COLOR_GREY, dash="dash"), annotation_text=hline_text, annotation_position="top right")
    fig.update_xaxes(title=xtitle, type="category" if categorical else None)
    fig.update_yaxes(title=ytitle, range=yrange)
    return _base(fig, height=height)


# --- Lewis-Signalspiel ------------------------------------------------------------------------------------------------------------------------
def build_lewis_trace(view: V.LewisView) -> go.Figure:
    """Effizienz über die Runden: Mittel über die Läufe, Bestwert min(1, ns/n) und 1/n als Startwert; dünn die ersten drei Läufe."""
    steps = view.steps
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=[s.efficiency.mean() for s in view.snaps], mode="lines", name="Mittel über die Läufe", line=dict(color=C.COLOR_A, width=3)))
    for r in range(min(3, view.runs)):
        fig.add_trace(go.Scatter(x=steps, y=[s.efficiency[r] for s in view.snaps], mode="lines", name=f"Lauf {r + 1}", line=dict(color=C.COLOR_LINE, width=1, dash="dot"), opacity=0.7,
                                 showlegend=r == 0))
    fig.add_hline(y=view.best, line=dict(color=C.COLOR_GREY, dash="dash"), annotation_text=f"Bestwert {V.de(view.best, 3)}", annotation_position="bottom right")
    fig.update_xaxes(title="Runden")
    fig.update_yaxes(title="Erfolgswahrscheinlichkeit", range=[0, 1.02])
    return _base(fig)


def build_lewis_hist(view: V.LewisView) -> go.Figure:
    """Verteilung der Effizienz aller Läufe am Ende (Histogramm); der Bestwert und 95 % davon sind markiert."""
    eff = view.last.efficiency
    fig = go.Figure(go.Histogram(x=eff, xbins=dict(start=0, end=1.0001, size=0.05), marker_color=C.COLOR_A))
    fig.add_vline(x=0.95 * view.best, line=dict(color=C.COLOR_MIS, dash="dash"), annotation_text="95 % des Bestwerts", annotation_position="top left")
    fig.update_xaxes(title="Effizienz am Ende", range=[0, 1.02])
    fig.update_yaxes(title="Zahl der Läufe")
    return _base(fig, height=280)
