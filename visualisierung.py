"""Grafische Auswertung einer PLSP-SD-E-Loesung (matplotlib).

Erzeugt eine zusammenhaengende Abbildung mit fuenf aufeinander abgestimmten Panels:

  1. Produktionsplan        : Wer wird wann in welcher Menge produziert (x_{kt}).
  2. Leistungszustand       : Maschinenzustand je Periode (An / Standby / Aus).
  3. Ruestzustand & -wechsel: Auf welches Produkt ist gerüstet (omega_{kt}); Pfeile
                              markieren Ruestwechsel i->k (chi_{ikt}).
  4. Emissionszusammensetzung: Aufschluesselung von E_t in seine Bestandteile (12).
  5. Kostenaufschluesselung : Wasserfall der Zielfunktion (1) inkl. Verkaufserloes.

Das Modul gehoert zur Auswertungsschicht (getrennt von Modell und Daten) und setzt ein
bereits geloestes Vollmodell (PLSP-SD-E) voraus.
"""

from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("Agg")  # kein Display noetig (Headless/Server)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from gurobipy import GRB

from instanz import Instanz

# ----------------------------------------------------------------------------
# Farbkonzept (einheitlich ueber alle Panels)
# ----------------------------------------------------------------------------
_ZUSTAND_FARBE = {"An": "#2e7d32", "Standby": "#f9a825", "Aus": "#9e9e9e"}
_ZUSTAND_KURZ = {"An": "An", "Standby": "Standby", "Aus": "Aus"}
_EMISSION_FARBEN = {
    "Produktion": "#1565c0",
    "Rüsten": "#6a1b9a",
    "Lager": "#00838f",
    "An (e^on)": "#2e7d32",
    "Standby (e^sb)": "#f9a825",
    "Aus (e^aus)": "#9e9e9e",
    "Anschalten (e^an)": "#c62828",
}
_KOSTEN_POS = "#1565c0"
_KOSTEN_NEG = "#c62828"
_KOSTEN_SUM = "#37474f"


def _produktfarben(K: int):
    """Liefert eine stabile Farbzuordnung je Produkt 1..K."""
    palette = plt.get_cmap("tab10").colors
    return {k: palette[(k - 1) % len(palette)] for k in range(1, K + 1)}


def visualisiere(
    modell,
    daten: Instanz,
    titel: str = "PLSP-SD-E – Lösungsübersicht",
    dateipfad: Optional[str] = None,
):
    """Erzeugt die fuenfteilige Loesungsuebersicht fuer ein geloestes Vollmodell.

    Parameter
    ---------
    modell : gp.Model
        Bereits optimiertes PLSP-SD-E-Modell (Status OPTIMAL).
    daten : Instanz
        Zugehoerige Dateninstanz (fuer Parameter/Beschriftung).
    titel : str
        Ueberschrift der Abbildung.
    dateipfad : str | None
        Wird ein Pfad angegeben, wird die Abbildung als Datei (z. B. PNG) gespeichert.

    Rueckgabe
    ---------
    matplotlib.figure.Figure
    """
    if modell.Status != GRB.OPTIMAL:
        raise ValueError(f"Modell ist nicht optimal geloest (Status={modell.Status}).")

    def val(name: str) -> float:
        v = modell.getVarByName(name)
        return v.X if v is not None else 0.0

    produkte = list(daten.produkte())
    perioden = list(daten.perioden())
    wechsel = daten.wechsel_paare()
    K, T = daten.K, daten.T
    p_farbe = _produktfarben(K)

    # ----- Lösungswerte auslesen ------------------------------------------------
    x = {(k, t): val(f"x[{k},{t}]") for k in produkte for t in perioden}
    y = {(k, t): val(f"y[{k},{t}]") for k in produkte for t in perioden}
    omega = {(k, t): val(f"omega[{k},{t}]") for k in produkte for t in perioden}
    u = {t: val(f"u[{t}]") for t in perioden}

    def zustand(t: int) -> str:
        if val(f"z_an[{t}]") > 0.5:
            return "An"
        if val(f"z_sb[{t}]") > 0.5:
            return "Standby"
        return "Aus"

    # Emissionsbestandteile je Periode gemaess (12)
    em = {
        "Produktion": {t: sum(daten.e_p[k] * x[k, t] for k in produkte) for t in perioden},
        "Rüsten": {t: sum((daten.e_fix[i, k] + daten.e_var * daten.tr[i, k])
                          * val(f"chi[{i},{k},{t}]") for (i, k) in wechsel) for t in perioden},
        "Lager": {t: sum(daten.e_l[k] * y[k, t] for k in produkte) for t in perioden},
        "An (e^on)": {t: daten.e_on * val(f"z_an[{t}]") for t in perioden},
        "Standby (e^sb)": {t: daten.e_sb * val(f"z_sb[{t}]") for t in perioden},
        "Aus (e^aus)": {t: daten.e_aus * val(f"z_aus[{t}]") for t in perioden},
        "Anschalten (e^an)": {t: daten.e_an * u[t] for t in perioden},
    }
    E = {t: val(f"E[{t}]") for t in perioden}

    # Kostenbestandteile der Zielfunktion (1)
    k_lager = sum(daten.h[k] * y[k, t] for k in produkte for t in perioden)
    k_ruest = sum(daten.s[k] * val(f"chi[{i},{k},{t}]")
                  for (i, k) in wechsel for t in perioden)
    k_zukauf = sum(daten.pi_B * val(f"B[{t}]") for t in perioden)
    k_verkauf = sum(daten.pi_S * val(f"S[{t}]") for t in perioden)
    Z = k_lager + k_ruest + k_zukauf - k_verkauf

    # ----- Figure-Layout --------------------------------------------------------
    breite = max(9.0, 1.7 * T + 3.0)
    fig = plt.figure(figsize=(breite, 15.0), constrained_layout=True)
    gs = GridSpec(
        5, 1, figure=fig,
        height_ratios=[3.0, 0.7, 1.1, 3.0, 2.6],
    )
    ax_prod = fig.add_subplot(gs[0])
    ax_zust = fig.add_subplot(gs[1], sharex=ax_prod)
    ax_ruest = fig.add_subplot(gs[2], sharex=ax_prod)
    ax_emis = fig.add_subplot(gs[3], sharex=ax_prod)
    ax_kost = fig.add_subplot(gs[4])

    fig.suptitle(titel, fontsize=16, fontweight="bold")

    # === Panel 1: Produktionsplan ==============================================
    bw = 0.8 / K
    for idx, k in enumerate(produkte):
        offs = -0.4 + bw * (idx + 0.5)
        xs = [t + offs for t in perioden]
        ys = [x[k, t] for t in perioden]
        ax_prod.bar(xs, ys, width=bw * 0.92, color=p_farbe[k],
                    label=f"Produkt {k}", edgecolor="white", linewidth=0.5)
        for t, xi, yi in zip(perioden, xs, ys):
            if yi > 1e-6:
                ax_prod.text(xi, yi, f"{yi:g}", ha="center", va="bottom", fontsize=9)
    ax_prod.set_ylabel("Produktionsmenge\n$x_{kt}$")
    ax_prod.set_title("1) Produktionsplan – wer wird wann in welcher Menge produziert",
                      loc="left", fontweight="bold", fontsize=11)
    ax_prod.legend(loc="upper right", ncol=K, framealpha=0.9)
    ax_prod.grid(axis="y", alpha=0.3)
    _maxx = max([x[k, t] for k in produkte for t in perioden] + [1e-9])
    ax_prod.set_ylim(0, _maxx * 1.25)

    # === Panel 2: Leistungszustand =============================================
    for t in perioden:
        zk = zustand(t)
        ax_zust.bar(t, 1.0, width=1.0, color=_ZUSTAND_FARBE[zk],
                    edgecolor="white", linewidth=1.5)
        ax_zust.text(t, 0.5, _ZUSTAND_KURZ[zk], ha="center", va="center",
                     color="white", fontweight="bold", fontsize=10)
    ax_zust.set_ylim(0, 1)
    ax_zust.set_yticks([])
    ax_zust.set_ylabel("Leistungs-\nzustand")
    ax_zust.set_title("2) Maschinenzustand je Periode", loc="left",
                      fontweight="bold", fontsize=11)
    leg2 = [Patch(facecolor=_ZUSTAND_FARBE[z], label=z) for z in ("An", "Standby", "Aus")]
    ax_zust.legend(handles=leg2, loc="center left", bbox_to_anchor=(1.005, 0.5),
                   framealpha=0.9, fontsize=9)

    # === Panel 3: Rüstzustand & Rüstwechsel ====================================
    for t in perioden:
        # gerüstetes Produkt am Ende von t (omega_{kt} = 1)
        geruestet = [k for k in produkte if omega[k, t] > 0.5]
        k_akt = geruestet[0] if geruestet else None
        farbe = p_farbe[k_akt] if k_akt is not None else "#e0e0e0"
        ax_ruest.bar(t, 1.0, width=1.0, color=farbe, edgecolor="white", linewidth=1.5)
        if k_akt is not None:
            ax_ruest.text(t, 0.30, f"P{k_akt}", ha="center", va="center",
                          color="white", fontweight="bold", fontsize=10)
    # Rüstwechsel als Pfeil + Beschriftung (im Headroom oberhalb des Bandes)
    for (i, k) in wechsel:
        for t in perioden:
            if val(f"chi[{i},{k},{t}]") > 0.5:
                ax_ruest.annotate(
                    f"{i}→{k}", xy=(t, 1.02), xytext=(t, 1.42),
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="#b71c1c",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="#b71c1c", lw=1.2),
                    arrowprops=dict(arrowstyle="-|>", color="#b71c1c", lw=2.0),
                )
    ax_ruest.set_ylim(0, 1.8)
    ax_ruest.set_yticks([])
    ax_ruest.set_ylabel("Rüstzustand\n$\\omega_{kt}$")
    ax_ruest.set_title("3) Rüstzustand (Produkt) und Rüstwechsel $i\\rightarrow k$",
                       loc="left", fontweight="bold", fontsize=11, pad=8)

    # === Panel 4: Emissionszusammensetzung =====================================
    boden = {t: 0.0 for t in perioden}
    for name, farbe in _EMISSION_FARBEN.items():
        werte = [em[name][t] for t in perioden]
        if all(abs(w) < 1e-9 for w in werte):
            continue  # Bestandteil ohne Beitrag nicht in die Legende
        ax_emis.bar(perioden, werte, width=0.6,
                    bottom=[boden[t] for t in perioden],
                    color=farbe, label=name, edgecolor="white", linewidth=0.4)
        for t, w in zip(perioden, werte):
            boden[t] += w
    for t in perioden:
        if E[t] > 1e-6:
            ax_emis.text(t, E[t], f"$E_t$={E[t]:.2f}", ha="center", va="bottom",
                         fontsize=9, fontweight="bold")
    ax_emis.set_ylabel("Emission $E_t$")
    ax_emis.set_title("4) Zusammensetzung der Gesamtemission je Periode (12)",
                      loc="left", fontweight="bold", fontsize=11)
    ax_emis.legend(loc="center left", bbox_to_anchor=(1.005, 0.5),
                   framealpha=0.9, fontsize=9, title="Emissionsquelle")
    ax_emis.grid(axis="y", alpha=0.3)
    _maxe = max(list(E.values()) + [1e-9])
    ax_emis.set_ylim(0, _maxe * 1.25)
    ax_emis.set_xlabel("Mikroperiode $t$")
    ax_emis.set_xticks(perioden)

    # === Panel 5: Kostenaufschlüsselung (Wasserfall) ===========================
    namen = ["Lagerkosten", "Rüstkosten", "Zukauf\n$\\pi^B B_t$",
             "Verkauf\n$-\\pi^S S_t$", "Zielwert $Z$"]
    werte = [k_lager, k_ruest, k_zukauf, -k_verkauf]
    kum = 0.0
    for i, w in enumerate(werte):
        farbe = _KOSTEN_POS if w >= 0 else _KOSTEN_NEG
        ax_kost.bar(i, w, bottom=kum, color=farbe, edgecolor="black", linewidth=0.8)
        spitze = kum + w
        ax_kost.text(i, spitze + (0.01 if w >= 0 else -0.01) * abs(Z if Z else 1),
                     f"{w:+.2f}", ha="center",
                     va="bottom" if w >= 0 else "top", fontsize=9, fontweight="bold")
        if i < len(werte) - 1:  # Verbindungslinie zum naechsten Balken
            ax_kost.plot([i + 0.4, i + 1 - 0.4], [spitze, spitze],
                         color="gray", linestyle="--", linewidth=0.8)
        kum = spitze
    # Summenbalken Z von 0
    ax_kost.bar(len(werte), kum, bottom=0.0, color=_KOSTEN_SUM,
                edgecolor="black", linewidth=0.8)
    ax_kost.text(len(werte), kum, f"{kum:+.2f}", ha="center",
                 va="bottom" if kum >= 0 else "top", fontsize=10, fontweight="bold")
    # Headroom, damit Wertelabels nicht mit dem Titel kollidieren
    c, tops = 0.0, []
    for w in werte:
        c += w
        tops.append(c)
    alle = tops + [0.0, kum]
    ymin, ymax = min(alle), max(alle)
    spanne = (ymax - ymin) or 1.0
    ax_kost.set_ylim(ymin - 0.10 * spanne, ymax + 0.16 * spanne)
    ax_kost.axhline(0, color="black", linewidth=1.0)
    ax_kost.set_xticks(range(len(namen)))
    ax_kost.set_xticklabels(namen)
    ax_kost.set_ylabel("Beitrag zu $Z$")
    ax_kost.set_title("5) Kostenaufschlüsselung der Zielfunktion (1) – Kosten (+) minus "
                      "Verkaufserlös (−)", loc="left", fontweight="bold", fontsize=11)
    ax_kost.grid(axis="y", alpha=0.3)

    # gemeinsame x-Achse für Panels 1–4
    ax_emis.set_xlim(0.5, T + 0.5)
    for ax in (ax_prod, ax_zust, ax_ruest):
        plt.setp(ax.get_xticklabels(), visible=False)

    if dateipfad:
        fig.savefig(dateipfad, dpi=130, bbox_inches="tight")
    return fig
