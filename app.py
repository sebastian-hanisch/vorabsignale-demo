"""Signale und Konventionen: ändert Reden, was sich durchsetzt? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Siebtes Stück der Reihe Koordinationsspiele. Zwei Betriebe wählen gleichzeitig A oder B; vorher dürfen sie sich eine kostenlose, unverbindliche Nachricht schicken (Cheap Talk). Die Demo rechnet das
erweiterte Spiel und seine Gleichgewichte exakt, lässt zwei Populationen darin per Logit lernen und fragt, ob Reden eine Altkonvention öffnet, wer im Kampf der Geschlechter gewinnt und wann ein
Wort eine Bedeutung bekommt; dazu das Lewis-Signalspiel, in dem Bedeutung aus dem Lernen entsteht.

Lauffähig mit: streamlit run app.py
"""

import math

import streamlit as st

import vor_budget as BU
import vor_constants as C
import vor_evaluation as V
import vor_experiments as XP
import vor_games as G
import vor_mechanism as M
import vor_scenario as S
import vor_texts as T
import vor_visualization as Z
from vor_layout import dunkles_design, kopf_css
from vor_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)

st.set_page_config(page_title="Signale und Konventionen – Sebastian Hanisch", layout="wide")


# --- Zwischenspeicher (Schlüssel = alle Eingaben inklusive Zufalls-Seed) --------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _meas():
    return XP.load_measurements()


@st.cache_data(show_spinner=False)
def _mech():
    return M.load()


@st.cache_data(show_spinner=False)
def _analyse(s):
    return V.analyse(s)


@st.cache_data(show_spinner=False)
def _analyse_lewis(s):
    return V.analyse_lewis(s)


@st.cache_data(show_spinner=False)
def _eq_row(game, b, t, talk, K):
    return XP.equilibrium_row(game, b, t, talk, K)


@st.cache_data(show_spinner=False)
def _e1_rows(b, t):
    return V.e1_rows(b, t)


@st.cache_data(show_spinner=False)
def _e2_rows(b, talk_k):
    return V.e2_rows(b, talk_k)


@st.cache_data(show_spinner=False)
def _e3_live(s):
    return V.e3_live(s)


@st.cache_data(show_spinner=False)
def _e4_live(s):
    return V.e4_live(s)


@st.cache_data(show_spinner=False)
def _e5_live(s):
    return V.e5_live(s)


@st.cache_data(show_spinner=False)
def _e6_live(s):
    return V.e6_live(s)


def show(verdict):
    kind, text = verdict
    getattr(st, kind)(text)


def table(rows):
    st.dataframe(rows, hide_index=True, width="stretch")


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in vor_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🗣️ Signale und Konventionen: ändert Reden, was sich durchsetzt?")
st.markdown(T.INTRO)

with st.expander("Kostenloses Reden in einem Absatz", expanded=True):
    st.markdown(T.TALK_EXPLAINER)

with st.expander("Erklärtabelle: was die Regler tun"):
    st.markdown(T.CONTROLS_TABLE)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

is_lewis = st.session_state.get("part_select") == C.PART_LEWIS
is_hirsch = st.session_state.get("game_select") == G.GAME_HIRSCH
no_talk = st.session_state.get("talk_select") == G.TALK_NONE

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("🗣️ Spiel und Reden")
    part = st.selectbox("Teil", C.PARTS, key="part_select", help="Kostenloses Reden vor einem 2×2-Spiel, oder das Lewis-Signalspiel, in dem Bedeutung aus dem Lernen entsteht. Die Regler des anderen Teils sind ausgegraut.")
    game = st.selectbox("Spiel", G.GAMES, key="game_select", disabled=is_lewis,
                        help="Reine Koordination (gleiche Wahl zahlt 1), Hirschjagd (Hirsch gegen Hirsch 9, Hase sicher) oder Kampf der Geschlechter (jede Seite bevorzugt ein anderes Gleichgewicht).")
    b = st.slider("Hase-Auszahlung b", *bounds("b_slider"), key="b_slider", step=0.1, disabled=is_lewis or not is_hirsch,
                  help="Nur in der Hirschjagd wirksam: Auszahlung von B gegen B (Hirsch gegen Hirsch bringt 9, Hirsch allein 0). Die Schwelle p* ist b / (9 − t).")
    t = st.slider("Verlockung t", *bounds("t_slider"), key="t_slider", step=0.25, disabled=is_lewis or not is_hirsch,
                  help="Nur in der Hirschjagd wirksam: Hase allein gegen Hirsch bringt b + t. Mit t > 0 hat die Zeile einen Grund zum Lügen.")
    talk = st.selectbox("Reden", G.TALKS, key="talk_select", disabled=is_lewis, help="Keins, nur die Zeile spricht (die Spalte hört), oder beide sprechen gleichzeitig. Die Nachricht kostet nichts und bindet nicht.")
    K = st.slider("Nachrichten K", *bounds("k_slider"), key="k_slider", step=1, disabled=is_lewis or no_talk,
                  help="Zahl der möglichen Nachrichten. K = 1 ist die Kontrolle: das Reden hat keinen Inhalt. Beidseitig mit K = 3 hat jede Seite 1.536 Strategien: dort werden nur Gleichgewichte gezählt, nicht gelernt.")
    st.header("🎚️ Lernen")
    beta = st.slider("Schärfe β", *bounds("beta_slider"), key="beta_slider", step=1.0, disabled=is_lewis,
                     help="Wie genau ein Akteur die beste Strategie wählt (Auszahlungen auf den Höchstwert 1 skaliert). Klein: viel Rauschen, groß: wenig.")
    n_agents = st.slider("Akteure N je Population", *bounds("n_slider"), key="n_slider", step=1, disabled=is_lewis, help="Größe jeder der zwei Populationen (Zeile, Spalte).")
    start = st.selectbox("Start", C.STARTS, key="start_select", disabled=is_lewis, help="Zufällig verteilt, oder alle in der Altkonvention „alle B“ (B, nach Möglichkeit mit Nachricht 1 und Regeln, die nur B spielen).")
    steps = st.slider("Schritte", *bounds("steps_slider"), key="steps_slider", step=2_000, disabled=is_lewis, help="Zahl der Überprüfungen; je Schritt überprüft ein Akteur einer Population seine Strategie.")
    runs = st.slider("Läufe", *bounds("runs_slider"), key="runs_slider", step=2, help="Unabhängige Läufe, die nebeneinander gerechnet werden. Die App kürzt sie, wenn die geschätzte Rechenzeit zu lang wird.")
    st.header("🔤 Lewis-Signalspiel")
    lewis_n = st.slider("Zustände n", *bounds("lewis_n_slider"), key="lewis_n_slider", step=1, disabled=not is_lewis, help="Zahl der gleich wahrscheinlichen Zustände; der Hörer wählt eine von n Handlungen.")
    lewis_ns = st.slider("Signale", *bounds("lewis_ns_slider"), key="lewis_ns_slider", step=1, disabled=not is_lewis, help="Zahl der Signale. Mit weniger Signalen als Zuständen sind nicht alle Zustände unterscheidbar.")
    lewis_rounds = st.slider("Runden", *bounds("lewis_rounds_slider"), key="lewis_rounds_slider", step=1_000, disabled=not is_lewis, help="Runden des Roth-Erev-Lernens (Urnen).")
    st.header("🎲 Lauf")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Zufalls-Seed ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"part_select": part, "game_select": game, "b_slider": float(b), "t_slider": float(t), "talk_select": talk, "k_slider": int(K), "beta_slider": float(beta),
                   "n_slider": int(n_agents), "start_select": start, "steps_slider": int(steps), "runs_slider": int(runs), "lewis_n_slider": int(lewis_n),
                   "lewis_ns_slider": int(lewis_ns), "lewis_rounds_slider": int(lewis_rounds), "seed_input": int(seed)})

settings = S.Settings(part=part, game=game, b=float(b), t=float(t), talk=talk, K=int(K), beta=float(beta), n_agents=int(n_agents), start=start, steps=int(steps), runs=int(runs),
                      seed=int(seed), lewis_n=int(lewis_n), lewis_ns=int(lewis_ns), lewis_rounds=int(lewis_rounds))
verdicts = []   # Verdikte für den PDF-Export
meas = _meas()
mech = _mech()
if meas is None:
    st.warning("Die vorgerechneten Messreihen (data/measurements.json) fehlen oder gehören zu einem anderen Rechencode: `python tools/make_measurements.py` ausführen. Die Experimente zeigen dann keine Reihen.")

# --- Hauptansicht --------------------------------------------------------------------------------------------------------------------------
if settings.part == C.PART_LEWIS:
    st.markdown("## 🔤 Eine Sprache entsteht")
    secs = BU.lewis_estimated_seconds(settings)
    with st.spinner(f"Rechne {settings.runs} Läufe über {V.thousands(settings.lewis_rounds)} Runden (rund {max(1, math.ceil(secs))} Sekunden) …"):
        lview = _analyse_lewis(settings)
    cross, pairs = lview.cross()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mittlere Effizienz am Ende", V.de(float(lview.last.efficiency.mean()), 3), help="Mittel über die Läufe der Wahrscheinlichkeit, dass die Handlung des Hörers dem Zustand entspricht.")
    m2.metric("Bestwert min(1, ns/n)", V.de(lview.best, 3), help="Mehr als ns Zustände kann der Hörer mit ns Signalen nicht unterscheiden.")
    m3.metric("Läufe am Bestwert", V.pct(float(lview.reached.mean()), 1), help="Anteil der Läufe mit Effizienz mindestens 95 % des Bestwerts.")
    m4.metric("Fremdspiel", "–" if cross is None else V.de(cross, 3), help="Sprecher aus Lauf r gegen Hörer aus Lauf r + 1, nur Paare, in denen beide Läufe den Bestwert erreicht haben; exakt aus den Wahrscheinlichkeiten.")
    st.markdown("#### ❓ Kernfrage: Entsteht eine Sprache, und versteht ein Fremder sie?")
    v_lewis = V.verdict_lewis_live(lview)
    show(v_lewis)
    verdicts.append(v_lewis)
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(Z.build_lewis_trace(lview), width="stretch", key="lewis_trace_chart")
        st.caption("Erfolgswahrscheinlichkeit über die Runden: Mittel über die Läufe und drei einzelne Läufe. Startwert 1/n.")
    with col_b:
        st.plotly_chart(Z.build_lewis_hist(lview), width="stretch", key="lewis_hist_chart")
        st.caption("Wo die Läufe am Ende stehen: Läufe unter der Linie sind in Teil-Pooling hängen geblieben (ein Signal für mehrere Zustände).")
else:
    st.markdown("## 📊 Das erweiterte Spiel und was gelernt wird")
    ext = S.ext_of(settings)
    pstar = S.threshold(settings)
    p1, p2 = ext.P1, ext.P2
    st.markdown("**Das Spiel ohne Reden** (Auszahlung Zeile; Spalte), Aktion A = Hirsch bzw. Treffpunkt 1, B = Hase bzw. Treffpunkt 2:")
    table([{"": f"Zeile {name}", "Spalte A": f"{V.de(p1[a, 0], 1)}; {V.de(p2[a, 0], 1)}", "Spalte B": f"{V.de(p1[a, 1], 1)}; {V.de(p2[a, 1], 1)}"} for a, name in enumerate(("A", "B"))])
    if pstar is not None:
        st.caption(f"Schwelle der Hirschjagd p* = b / (9 − t) = {V.de(pstar, 3)}: A lohnt sich, wenn der Partner mit mindestens diesem Anteil A spielt.")
    row_none = _eq_row(settings.game, S.effective_b(settings), S.effective_t(settings), G.TALK_NONE, 0)
    row_talk = _eq_row(settings.game, S.effective_b(settings), S.effective_t(settings), settings.talk, S.messages(settings))
    st.markdown("**Gleichgewichte** des Spiels ohne Reden und des erweiterten Spiels mit der gewählten Redeart:")
    table(V.equilibrium_table([row_none] if settings.talk == G.TALK_NONE else [row_none, row_talk]))
    v_eq = V.verdict_equilibria(row_none, row_talk)
    show(v_eq)
    verdicts.append(v_eq)
    used = BU.used_runs(settings)
    if used == 0:
        view = V.analyse(settings)
        v_main = V.verdict_main(view)
        show(v_main)
        verdicts.append(v_main)
    else:
        secs = BU.estimated_seconds(ext, settings.steps, used)
        with st.spinner(f"Lerne {used} Läufe über {V.thousands(settings.steps)} Schritte (rund {max(1, math.ceil(secs))} Sekunden) …"):
            view = _analyse(settings)
        if used < settings.runs:
            st.info(f"Gerechnet mit {used} statt {settings.runs} Läufen: die geschätzte Rechenzeit wäre sonst länger als {int(C.MAX_SECONDS)} Sekunden.")
        last = view.last
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("(A, A)", V.pct(float(last.aa.mean()), 1), help="Mittel über die Läufe: Anteil der Paarungen, in denen beide A spielen (bei zufälliger Paarung der Akteure).")
        m2.metric("(B, B)", V.pct(float(last.bb.mean()), 1), help="Anteil der Paarungen, in denen beide B spielen.")
        m3.metric("Fehlkoordination", V.pct(float(last.mis.mean()), 1), help="Anteil der Paarungen mit verschiedenen Aktionen.")
        m4.metric("Auszahlung Zeile / Spalte", f"{V.de(float(last.payoff1.mean()), 2)} / {V.de(float(last.payoff2.mean()), 2)}", help="Mittlere Auszahlung bei zufälliger Paarung.")
        m5.metric("Information der Nachricht", f"{V.de(float(last.info.mean()), 3)} Bit", help="Wie viel die Nachricht über die Aktion des Partners verrät (exakt aus den Besetzungen); ohne Reden 0.")
        st.markdown("#### ❓ Kernfrage: Was setzt sich durch, und trägt die Nachricht etwas?")
        v_main = V.verdict_main(view)
        show(v_main)
        verdicts.append(v_main)
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(Z.build_trace(view), width="stretch", key="trace_chart")
            st.caption("Verlauf des Lernens: Mittel über die Läufe, dünn gepunktet drei einzelne Läufe für (A, A).")
        with col_b:
            st.plotly_chart(Z.build_payoffs(view), width="stretch", key="payoff_chart")
            st.caption("Mittlere Auszahlung der Zeile und der Spalte über die Schritte.")
            st.plotly_chart(Z.build_info(view), width="stretch", key="info_chart")
            st.caption("Information der Nachricht in Bit über die Schritte (Mittel über die Läufe).")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["1 Gleichgewichtsmenge", "2 Verlockung", "3 Altkonvention verlassen", "4 Wer spricht, gewinnt", "5 Die Lotterie", "6 Lewis: eine Sprache entsteht"])


def live_button(label: str, key: str, help_text: str) -> bool:
    """▶️-Knopf: merkt sich die Einstellungen, für die gerechnet wurde; true, solange diese Einstellungen noch gelten."""
    if st.button(label, key=key, help=help_text):
        st.session_state[f"{key}_signature"] = settings
    return st.session_state.get(f"{key}_signature") == settings


# --- Tab 1 ---------------------------------------------------------------------------------------------------------------------------------
with tab1:
    st.subheader("🔬 Experiment E1: Ändert Reden die Gleichgewichtsmenge?")
    st.caption("Vier Spiele, fünf Arten des Redens, alles exakt (Aufzählung der reinen Gleichgewichte, Trägerpaar-Rechnung in Brüchen für alle Gleichgewichte bei kein Reden und einseitigem Reden mit K ≤ 2). "
               "Hirschjagd mit b = 7: Hirsch gegen Hirsch 9, Hirsch allein 0, Hase 7 (mit t = 1: Hase allein gegen Hirsch 8).")
    rows1 = _e1_rows(C.DEFAULT_B, 0.0)
    table(V.e1_table(rows1))
    v1 = V.verdict_e1(rows1)
    show(v1)
    verdicts.append(v1)
    st.markdown("**Mit den Reglern der Seitenleiste nachrechnen** (Spiel, b und t aus der Seitenleiste; dazu beidseitiges Reden mit K = 3, wo nur gezählt wird: 1.536 Strategien je Seite)")
    if live_button("▶️ Gleichgewichte zählen", "e1_button", "Zählt die reinen Gleichgewichte aller fünf Arten des Redens für Spiel, b und t der Seitenleiste, dazu beidseitig mit K = 3; rechnet einige Sekunden."):
        with st.spinner("Zähle Gleichgewichte (einige Sekunden) …"):
            live1 = _e1_rows(S.effective_b(settings), S.effective_t(settings))
        table(V.e1_table(live1, only_game=settings.game))
        three = V.two_sided_three(settings.game, S.effective_b(settings), S.effective_t(settings))
        st.caption(f"Beidseitig mit K = 3: {V.thousands(three['strategies'])} Strategien je Seite, {V.thousands(three['pure'])} reine Gleichgewichte; Auszahlungspaare der reinen Gleichgewichte: {V.pairs_text(three['pure_payoffs'])}.")

# --- Tab 2 ---------------------------------------------------------------------------------------------------------------------------------
with tab2:
    st.subheader("🔬 Experiment E2: Verlockung und Glaubwürdigkeit")
    st.caption("In der Hirschjagd mit Hase-Auszahlung b: Gibt es ein Gleichgewicht, in dem die Nachricht etwas bedeutet, wenn Hase allein gegen Hirsch b + t bringt? "
               "Einseitiges Reden mit K = 2, exakt. Das Beispiel in der Art von Aumann hat b = 7 und t = 1 (die Zahlen 9, 0, 8, 7; kein Zitat).")
    e2_b = st.selectbox("Hase-Auszahlung b", (3.6, 4.5, 5.4, 7.0), index=3, key="e2_b", format_func=lambda x: V.de(x, 1))
    rows2 = _e2_rows(e2_b, (G.TALK_ONE, 2))
    table(V.e2_table(rows2))
    v2 = V.verdict_e2(rows2)
    show(v2)
    verdicts.append(v2)
    st.markdown("**Mit den Reglern der Seitenleiste nachrechnen** (b aus der Seitenleiste, t läuft durch 0 / 0,25 / 0,5 / 1; einseitiges Reden mit K = 2)")
    if live_button("▶️ Verlockungen durchrechnen", "e2_button", "Rechnet alle Gleichgewichte für t = 0, 0,25, 0,5 und 1 bei der Hase-Auszahlung der Seitenleiste; rechnet einige Sekunden."):
        with st.spinner("Rechne Gleichgewichte (einige Sekunden) …"):
            live2 = _e2_rows(S.effective_b(settings), (G.TALK_ONE, 2))
        table(V.e2_table(live2))

# --- Tab 3 ---------------------------------------------------------------------------------------------------------------------------------
with tab3:
    st.subheader("🔬 Experiment E3: Altkonvention verlassen")
    st.caption("Start: alle spielen B (Hase). Die Hirschjagd hat die Schwelle p* = b / 9 (t = 0). Gemessen wird der Anteil der Paarungen (A, A) nach 40.000 Schritten, je Schärfe β. "
               + (f"Vorgerechnet: {meas['blocks']} Blöcke zu je {meas['runs_per_block']} Läufen, N = 20 Akteure je Population; ± ist der Standardfehler der Blöcke." if meas else ""))
    e3_beta = st.selectbox("Schärfe β", C.E3_BETAS, key="e3_beta", format_func=lambda x: V.de(x, 0))
    if meas:
        table(V.e3_table(meas, e3_beta))
        st.plotly_chart(Z.build_lines({V.talk_label(tk, k): ([V.de(p, 1) for p in C.THRESHOLD_PSTARS], [XP.find_cell(meas, family="e3", b=round(9 * p, 6), talk=tk, K=k, beta=e3_beta)["aa"]["mean"] for p in C.THRESHOLD_PSTARS],
                                                             [XP.find_cell(meas, family="e3", b=round(9 * p, 6), talk=tk, K=k, beta=e3_beta)["aa"]["se"] for p in C.THRESHOLD_PSTARS]) for tk, k in C.E3_TALKS},
                                  "Schwelle p*", "Anteil (A, A) nach 40.000 Schritten", yrange=[-0.02, 1.02], categorical=True), width="stretch", key="e3_chart")
        v3 = V.verdict_e3(meas, e3_beta)
        show(v3)
        verdicts.append(v3)
        st.markdown("#### Warum öffnet Reden die Altkonvention? Der Hörer-Anteil")
        st.caption("Gemessen an den Läufen, die noch in der Altkonvention stehen (weniger als ein Zehntel der Paarungen (A, A)), einseitiges Reden: s_max ist der größte Anteil der Hörer, die auf irgendeine Nachricht mit A antworten, "
                   "n_min die Zahl der Sprecher auf der am seltensten gesendeten Nachricht. Die Formel gibt den Hörer-Anteil an, wenn nur ein einziger Sprecher mit A auf einer Nachricht liegt und n Sprecher sie senden.")
        table(V.mech_table(meas))
        if mech is not None:
            table(V.mech_event_table(mech))
            st.caption(f"Ereignisstudie: jeder Schritt aufgezeichnet, {mech['cells'][0]['runs']} Läufe je Zelle (4 Blöcke), N = 20, 40.000 Schritte. Austritt: erstmals spielt mindestens die Hälfte der Sprecher A; n und s gelten für die Nachricht mit den meisten A-Sprechern zu diesem Zeitpunkt.")
        v3m = V.verdict_mechanism(meas, mech)
        show(v3m)
        verdicts.append(v3m)
        e3_p = st.selectbox("Schwelle p* für die Formel", C.THRESHOLD_PSTARS, key="e3_p", format_func=lambda x: V.de(x, 1))
        table(V.hearer_table(e3_beta, 20, e3_p))
        st.markdown("#### Geheimer Handschlag und Abhängigkeit von N")
        st.caption("Hirschjagd mit b = 7 (p* = 7/9). Beidseitiges Reden mit K = 2: ein Mutant „sende 2, spiele A, wenn der andere auch 2 sendet, sonst B“ verdient gegen Bewohner (immer B) 7 wie diese und gegen seinesgleichen 9. "
                   "Schritte: 10.000 je Akteur.")
        table(V.hs_table(meas))
        v3h = V.verdict_handshake(meas)
        show(v3h)
        verdicts.append(v3h)
    st.markdown("**Mit den Reglern der Seitenleiste nachrechnen** (Hirschjagd mit b, t, β, N, Schritten und Zufalls-Seed der Seitenleiste, Start in der Altkonvention, ein Block; die fünf Arten des Redens)")
    est3 = sum(BU.estimated_seconds(G.build(G.GAME_HIRSCH, S.effective_b(settings), S.effective_t(settings), tk, k), settings.steps, BU.used_runs(S.with_(settings, talk=G.TALK_TWO, K=2))) for tk, k in C.E3_TALKS)
    if live_button("▶️ Redearten durchrechnen", "e3_button", f"Fünf Lernläufe in der Hirschjagd der Seitenleiste, Start in der Altkonvention; rechnet rund {max(1, math.ceil(est3))} Sekunden."):
        with st.spinner(f"Rechne fünf Lernläufe (rund {max(1, math.ceil(est3))} Sekunden) …"):
            live3 = _e3_live(S.with_(settings, game=G.GAME_HIRSCH))
        table(V.e3_live_table(live3))
        st.caption(f"Hirschjagd mit b = {V.de(settings.b, 1)}, t = {V.de(settings.t, 2)}, β = {V.de(settings.beta, 0)}, N = {settings.n_agents}, {V.thousands(settings.steps)} Schritte, {live3['runs']} Läufe, ein Block.")

# --- Tab 4 ---------------------------------------------------------------------------------------------------------------------------------
with tab4:
    st.subheader("🔬 Experiment E4: Wer spricht, gewinnt")
    st.caption("Kampf der Geschlechter (Zeile 2/1, Spalte 1/2), β = 20, N = 20, 40.000 Schritte. Auszahlung von Zeile und Spalte bei zufälligem Start und beim Start in der Altkonvention „alle B“ (Spalte bevorzugt). "
               + (f"Vorgerechnet: {meas['blocks']} Blöcke zu je {meas['runs_per_block']} Läufen." if meas else ""))
    if meas:
        e4_start = st.selectbox("Start", C.STARTS, key="e4_start")
        table(V.e4_table(meas, e4_start))
        cats = [V.talk_label(tk, k) for tk, k in C.E3_TALKS]
        cells4 = [XP.find_cell(meas, family="bos", talk=tk, K=k, start=e4_start) for tk, k in C.E3_TALKS]
        st.plotly_chart(Z.build_bars(cats, {"Zeile": [c["p1"]["mean"] for c in cells4], "Spalte": [c["p2"]["mean"] for c in cells4]}, "mittlere Auszahlung", errors={"Zeile": [c["p1"]["se"] for c in cells4], "Spalte": [c["p2"]["se"] for c in cells4]},
                                  yrange=[0, 2.2]), width="stretch", key="e4_chart")
        v4 = V.verdict_e4(meas)
        show(v4)
        verdicts.append(v4)
        st.markdown("#### Hirschjagd und reine Koordination bei zufälligem Start")
        e4_game = st.selectbox("Spiel", [("Reine Koordination", 0.0), ("Hirschjagd", 0.0), ("Hirschjagd", 1.0)], key="e4_game", format_func=lambda x: x[0] + (" mit Verlockung t = 1" if x[1] else ""))
        table(V.rand_table(meas, e4_game[0], e4_game[1]))
        v4r = V.verdict_random(meas)
        show(v4r)
        verdicts.append(v4r)
    st.markdown("**Mit den Reglern der Seitenleiste nachrechnen** (Kampf der Geschlechter mit β, N, Schritten, Start und Zufalls-Seed der Seitenleiste, ein Block; die fünf Arten des Redens)")
    est4 = sum(BU.estimated_seconds(G.build(G.GAME_BOS, 7.0, 0.0, tk, k), settings.steps, BU.used_runs(S.with_(settings, talk=G.TALK_TWO, K=2))) for tk, k in C.E3_TALKS)
    if live_button("▶️ Redearten im Kampf der Geschlechter", "e4_button", f"Fünf Lernläufe; rechnet rund {max(1, math.ceil(est4))} Sekunden."):
        with st.spinner(f"Rechne fünf Lernläufe (rund {max(1, math.ceil(est4))} Sekunden) …"):
            live4 = _e4_live(S.with_(settings, game=G.GAME_BOS))
        table(V.e4_live_table(live4))
        st.caption(f"β = {V.de(settings.beta, 0)}, N = {settings.n_agents}, {V.thousands(settings.steps)} Schritte, {live4['runs']} Läufe, Start: {settings.start}.")

# --- Tab 5 ---------------------------------------------------------------------------------------------------------------------------------
with tab5:
    st.subheader("🔬 Experiment E5: Die Lotterie")
    st.caption("Beidseitiges Reden, K = 2: beide senden gleichverteilt eine Nachricht; gleiche Nachrichten → beide A, verschiedene → beide B. Exakt: ist das ein Gleichgewicht, und was zahlt es? Danach: wird es gelernt?")
    rows5 = V.e5_rows(C.DEFAULT_B, 0.0)
    table(V.e5_table(rows5))
    v5 = V.verdict_e5(rows5, meas, mech)
    show(v5)
    verdicts.append(v5)
    if meas:
        table(V.e5_learned_table(meas))
    st.markdown("**Mit den Reglern der Seitenleiste nachrechnen** (Spiel, b, t aus der Seitenleiste; beidseitig K = 2: Gleichgewichtsprüfung gegen alle 64 reinen Abweichungen und ein Lernlauf mit β, N, Schritten und Zufalls-Seed)")
    est5 = BU.estimated_seconds(G.build(settings.game, S.effective_b(settings), S.effective_t(settings), G.TALK_TWO, 2), settings.steps, BU.used_runs(S.with_(settings, talk=G.TALK_TWO, K=2)))
    if live_button("▶️ Lotterie prüfen und lernen", "e5_button", f"Exakte Prüfung und ein Lernlauf; rechnet rund {max(1, math.ceil(est5))} Sekunden."):
        with st.spinner(f"Prüfe und lerne (rund {max(1, math.ceil(est5))} Sekunden) …"):
            live5 = _e5_live(settings)
        table(V.e5_live_table(live5))

# --- Tab 6 ---------------------------------------------------------------------------------------------------------------------------------
with tab6:
    st.subheader("🔬 Experiment E6: Lewis – eine Sprache entsteht")
    st.caption("n gleich wahrscheinliche Zustände, ns Signale, Erfolg zahlt 1; Roth-Erev-Lernen (Urnen). Reine Gleichgewichte durch Vollaufzählung (n = ns); Anteil der Läufe, die den Bestwert erreichen, und Fremdspiel. "
               + (f"Vorgerechnet: {meas['blocks']} Blöcke zu je {C.LEWIS_RUNS} Läufen." if meas else ""))
    st.markdown("**Gleichgewichte** (Vollaufzählung aller Sprecher- und Hörerabbildungen, n = ns):")
    table(V.lewis_ne_table())
    if meas:
        e6_cp = st.selectbox("Runden", C.LEWIS_CHECKS, index=2, key="e6_cp", format_func=V.thousands)
        table(V.e6_table(meas, e6_cp))
        pairs_eq = [(n, ns) for n, ns in C.LEWIS_PAIRS if n == ns]
        st.plotly_chart(Z.build_lines({f"nach {V.thousands(c)} Runden": ([str(n) for n, _ in pairs_eq], [XP.find_lewis(meas, n, ns)[str(c)]["share"]["mean"] for n, ns in pairs_eq],
                                                                         [XP.find_lewis(meas, n, ns)[str(c)]["share"]["se"] for n, ns in pairs_eq]) for c in C.LEWIS_CHECKS},
                                  "Zahl der Zustände n (Signale ns = n)", "Anteil der Läufe am Bestwert", yrange=[0, 1.02], categorical=True), width="stretch", key="e6_chart")
        v6 = V.verdict_e6(meas)
        show(v6)
        verdicts.append(v6)
    st.markdown("**Mit den Reglern der Seitenleiste nachrechnen** (Läufe, Runden und Zufalls-Seed der Seitenleiste; n läuft durch 2 bis 6 mit ns = n)")
    est6 = sum(settings.lewis_rounds * (33e-6 + 0.22e-6 * settings.runs) for _ in range(5))
    if live_button("▶️ Zustände durchrechnen", "e6_button", f"Fünf Lewis-Läufe (n = 2 bis 6, ns = n); rechnet rund {max(1, math.ceil(est6))} Sekunden."):
        with st.spinner(f"Rechne fünf Lewis-Läufe (rund {max(1, math.ceil(est6))} Sekunden) …"):
            live6 = _e6_live(settings)
        table(V.e6_live_table(live6))
        st.caption(f"{settings.runs} Läufe, {V.thousands(settings.lewis_rounds)} Runden, ns = n, ein Block.")

# --- Verwandte Demos -----------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔗 Verwandte Demos")
st.markdown(T.RELATED)

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(T.ASSUMPTIONS_MARKDOWN)

st.markdown("---")
with st.expander("📐 Mathematische Formulierung"):
    st.markdown(T.MATH_MARKDOWN)

st.markdown("---")
st.subheader("📄 Ergebnisse als PDF")
if st.button("PDF erstellen", key="pdf_button"):
    from vor_pdf_export import build_pdf
    st.session_state["pdf_bytes"] = build_pdf(lview if settings.part == C.PART_LEWIS else view, verdicts)
if st.session_state.get("pdf_bytes"):
    st.download_button("⬇️ PDF herunterladen", st.session_state["pdf_bytes"], file_name="vorabsignale.pdf", mime="application/pdf", key="pdf_download")

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Koordinationsspiele: von Gleichwahl bis Kaskade](https://sebastianhanisch.net/konzepte-koordinationsspiele.html)."
)
