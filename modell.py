"""MILP-Modell PLSP-SD-E (gurobipy).

Proportional Lot-Sizing and Scheduling Problem mit Sequenzabhängigen Rüstzeiten und
Emissionshandel (Erweiterung um Leistungszustände An/Standby/Aus, Anschaltvorgang und
EU-ETS-artigen Zertifikatshandel).

Dieses Modul enthält ausschließlich die Modelllogik (Aufbau des gurobipy-Modells aus
einer :class:`instanz.Instanz`). Es sind keine Zahlenwerte hartkodiert; sämtliche Daten
stammen aus der übergebenen Instanz.

================================================================================
ANNAHMEN UND DEFAULT-ENTSCHEIDUNGEN GEGENÜBER DER SPEZIFIKATION (1)-(15)
================================================================================
Die folgenden Festlegungen ergänzen die verbindliche Spezifikation, ohne eine
Restriktion hinzuzufügen, wegzulassen oder die Notation zu verändern. Sie sind hier
gesammelt sichtbar dokumentiert.

A1  Indizierung 1-basiert: Produkte k, i in {1..K}, Mikroperioden t in {1..T}, exakt
    wie in der mathematischen Notation.

A2  Anfangsperiode t=0: Die Zustands- und Bestandsgrößen mit (t-1)-Bezug werden für
    t=0 als FIXIERTE Variablen geführt: omega_{k,0}, y_{k,0}, z^{aus}_0 und J_0.
    Sie werden über die Anfangsbedingungen (8) [omega, y_0], (15) [z^aus_0] bzw. den
    Parameter J_0 festgesetzt. Dadurch greifen alle (t-1)-Bezüge bei t=1 einheitlich
    und lesbar auf diese Anfangswerte zu (statt Sonderfälle im Code zu führen).

A3  Rüstwechsel ohne Self-Loop: chi_{ikt} ist nur für i != k definiert
    (Logik-Hinweis "kein i=k-Self-Loop"; ein Verbleib auf k erzeugt RHS=0 in (6) und
    damit keinen Wechsel). Alle Summen "Sum_i" in (3), (6), (7), (12) sowie in der
    Zielfunktion (1) laufen entsprechend über i in {1..K} mit i != k. Würde man i=k
    zulassen, wären die Self-Loops in jedem Optimum 0 (sie kosten s_k bzw. tr_{ik});
    die Optimallösung ist identisch.

A4  tb_k > 0 wird vorausgesetzt (Belegungszeit je Einheit echt positiv), da tb_k in den
    Big-M-Termen im Nenner steht: M_{kt} = min(b_t / tb_k, Sum_{tau>=t} d_{k,tau}) und
    M_t = b_t / min_k tb_k. Beide Big-M werden in build_model aus den Daten berechnet.

A5  Induziertes binäres Verhalten: chi_{ikt} und u_t werden als CONTINUOUS (>= 0)
    deklariert; ihre 0/1-Annahme wird durch (4),(6),(7) bzw. (9),(11) zusammen mit den
    binären omega/z und den Vorzeichen in Ziel/Emission erzwungen (gemäß Spezifikation).

A6  J_t ist frei (lb = -unendlich). Die Nichtnegativität wird ausschließlich durch die
    separate Restriktion (14) J_t >= 0 erzwungen, exakt wie spezifiziert.

A7  reduktion_basismodell=True setzt alle Emissionsparameter (e^p, e^{fix}, e^{var},
    e^l, e^{on}, e^{sb}, e^{aus}, e^{an}) sowie die Zertifikatspreise pi^B, pi^S auf 0.
    Die Restriktionen (9)-(15) bleiben im Modell, sind dann aber zielfunktionsneutral,
    sodass exakt das Standard-PLSP-SD reproduziert wird.

A8  .lp-Export erfolgt durch den Aufrufer via ``modell.write(pfad)``; die in der
    Spezifikation vorgegebene Signatur von build_model bleibt unverändert.
================================================================================
"""

from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB

from instanz import Instanz


def build_model(daten: Instanz, reduktion_basismodell: bool = False) -> gp.Model:
    """Baut das vollständige PLSP-SD-E als gurobipy-Modell.

    Parameter
    ---------
    daten : Instanz
        Vollständige Probleminstanz (Mengen, Basis- und Erweiterungsparameter).
    reduktion_basismodell : bool, optional
        Falls True, werden sämtliche Emissionsparameter und die Zertifikatspreise
        pi^B, pi^S auf 0 gesetzt (siehe A7). Die Erweiterungsrestriktionen (9)-(15)
        bleiben strukturell erhalten, wirken aber zielfunktionsneutral, sodass exakt
        das reine PLSP-SD (Restriktionen (1)-(8)) reproduziert wird.

    Rückgabe
    --------
    gp.Model
        Das aufgebaute, noch nicht optimierte Modell. Ein optionaler .lp-Export ist
        über ``modell.write("datei.lp")`` durch den Aufrufer möglich (A8).
    """

    # -------------------------------------------------------------------------
    # 0) Indexmengen und (ggf. reduzierte) Parameter bereitstellen
    # -------------------------------------------------------------------------
    produkte = daten.produkte()             # k bzw. i in {1..K}
    perioden = daten.perioden()             # t in {1..T}
    perioden0 = daten.perioden_mit_null()   # t in {0..T} (inkl. Anfangsperiode 0, A2)
    wechsel = daten.wechsel_paare()         # (i, k) mit i != k (A3)
    K, T, i0 = daten.K, daten.T, daten.i_0

    # Emissions-/Preisparameter; im Reduktionsmodus auf 0 gesetzt (A7).
    if reduktion_basismodell:
        e_p = {k: 0.0 for k in produkte}
        e_fix = {(i, k): 0.0 for (i, k) in wechsel}
        e_var = 0.0
        e_l = {k: 0.0 for k in produkte}
        e_on = e_sb = e_aus = e_an = 0.0
        pi_B = pi_S = 0.0
    else:
        e_p = daten.e_p
        e_fix = daten.e_fix
        e_var = daten.e_var
        e_l = daten.e_l
        e_on, e_sb, e_aus, e_an = daten.e_on, daten.e_sb, daten.e_aus, daten.e_an
        pi_B, pi_S = daten.pi_B, daten.pi_S

    # -------------------------------------------------------------------------
    # 1) Big-M aus den Daten berechnen (keine Zahlenwerte hartkodiert, A4)
    #    M_{kt} = min( b_t / tb_k , Sum_{tau >= t} d_{k,tau} )
    #    M_t    = b_t / min_k tb_k
    # -------------------------------------------------------------------------
    tb_min = min(daten.tb[k] for k in produkte)
    M_kt = {
        (k, t): min(
            daten.b[t] / daten.tb[k],
            sum(daten.d[k, tau] for tau in range(t, T + 1)),
        )
        for k in produkte
        for t in perioden
    }
    M_t = {t: daten.b[t] / tb_min for t in perioden}

    # -------------------------------------------------------------------------
    # 2) Modell und Variablen anlegen
    #    Variablennamen spiegeln die Notation. omega/z binär; chi/u kontinuierlich.
    # -------------------------------------------------------------------------
    modell = gp.Model("PLSP-SD-E")

    # Produktionsmenge x_{kt} >= 0 (t in 1..T)
    x_kt = modell.addVars(produkte, perioden, lb=0.0, name="x")
    # Lagerbestand y_{kt} >= 0 am Ende von t; t=0 als Anfangslager (A2)
    y_kt = modell.addVars(produkte, perioden0, lb=0.0, name="y")
    # Rüstzustand omega_{kt} in {0,1}; t=0 als Anfangs-Rüstzustand (A2, (8))
    omega_kt = modell.addVars(produkte, perioden0, vtype=GRB.BINARY, name="omega")
    # Rüstwechsel chi_{ikt} >= 0 (kontinuierlich, binär induziert; i != k, A3/A5)
    chi_ikt = modell.addVars(
        [(i, k, t) for (i, k) in wechsel for t in perioden], lb=0.0, name="chi"
    )
    # Leistungszustände z^{an}, z^{sb}, z^{aus} in {0,1}
    z_an_t = modell.addVars(perioden, vtype=GRB.BINARY, name="z_an")
    z_sb_t = modell.addVars(perioden, vtype=GRB.BINARY, name="z_sb")
    # z^{aus} mit Anfangsperiode 0 (A2, (15))
    z_aus_t = modell.addVars(perioden0, vtype=GRB.BINARY, name="z_aus")
    # Anschaltindikator u_t >= 0 (kontinuierlich, binär induziert, A5)
    u_t = modell.addVars(perioden, lb=0.0, name="u")
    # Gesamtemission E_t >= 0
    E_t = modell.addVars(perioden, lb=0.0, name="E")
    # Zertifikatssaldo J_t (frei, A6); t=0 als Anfangssaldo J_0
    J_t = modell.addVars(perioden0, lb=-GRB.INFINITY, name="J")
    # Zukauf B_t >= 0, Verkauf S_t >= 0
    B_t = modell.addVars(perioden, lb=0.0, name="B")
    S_t = modell.addVars(perioden, lb=0.0, name="S")

    # -------------------------------------------------------------------------
    # 3) Zielfunktion (1)
    #    min Z = Sum_t Sum_k ( h_k y_{kt} + Sum_i s_k chi_{ikt} )
    #            + Sum_t ( pi^B B_t - pi^S S_t )
    #    Hinweis: s_k ist zielproduktabhängig (Rüstkosten je Wechsel IN das Produkt k).
    # -------------------------------------------------------------------------
    lagerkosten = gp.quicksum(daten.h[k] * y_kt[k, t] for k in produkte for t in perioden)
    ruestkosten = gp.quicksum(
        daten.s[k] * chi_ikt[i, k, t] for (i, k) in wechsel for t in perioden
    )
    zertifikatskosten = gp.quicksum(pi_B * B_t[t] - pi_S * S_t[t] for t in perioden)
    modell.setObjective(lagerkosten + ruestkosten + zertifikatskosten, GRB.MINIMIZE)

    # -------------------------------------------------------------------------
    # 4) Nebenbedingungen (2)-(15)
    # -------------------------------------------------------------------------

    # (2) Lagerbilanz (Bestandsfortschreibung): y_{k,t-1} + x_{kt} - y_{kt} = d_{kt}
    modell.addConstrs(
        (y_kt[k, t - 1] + x_kt[k, t] - y_kt[k, t] == daten.d[k, t]
         for k in produkte for t in perioden),
        name="c2_lagerbilanz",
    )

    # (3) Kapazität der Mikroperiode (Produktion + sequenzabhängiges Rüsten):
    #     Sum_k tb_k x_{kt} + Sum_i Sum_k tr_{ik} chi_{ikt} <= b_t
    #     (kein Anschaltterm: Anschalten verbraucht keine Kapazität)
    modell.addConstrs(
        (gp.quicksum(daten.tb[k] * x_kt[k, t] for k in produkte)
         + gp.quicksum(daten.tr[i, k] * chi_ikt[i, k, t] for (i, k) in wechsel)
         <= daten.b[t]
         for t in perioden),
        name="c3_kapazitaet",
    )

    # (4) Eindeutiger Rüstzustand je Mikroperiode: Sum_k omega_{kt} = 1
    modell.addConstrs(
        (gp.quicksum(omega_kt[k, t] for k in produkte) == 1 for t in perioden),
        name="c4_ruestzustand_eindeutig",
    )

    # (5) Produktion nur bei (vor-)gerüstetem Zustand: x_{kt} <= M_{kt} (omega_{k,t-1} + omega_{kt})
    modell.addConstrs(
        (x_kt[k, t] <= M_kt[k, t] * (omega_kt[k, t - 1] + omega_kt[k, t])
         for k in produkte for t in perioden),
        name="c5_produktion_nur_geruestet",
    )

    # (6) Rüstzustandsfortschreibung (Wechsel IN k): Sum_i chi_{ikt} >= omega_{kt} - omega_{k,t-1}
    #     i = Vorgänger, k = Zielindex; Summe über Vorgänger i (i != k).
    modell.addConstrs(
        (gp.quicksum(chi_ikt[i, k, t] for i in produkte if i != k)
         >= omega_kt[k, t] - omega_kt[k, t - 1]
         for k in produkte for t in perioden),
        name="c6_wechsel_in_k",
    )

    # (7) Spiegelbild (Wechsel AUS k): Sum_i chi_{kit} <= omega_{k,t-1}
    #     k = Vorgängerindex, Summe über Nachfolger i (Variable chi_{kit}!), i != k.
    modell.addConstrs(
        (gp.quicksum(chi_ikt[k, i, t] for i in produkte if i != k)
         <= omega_kt[k, t - 1]
         for k in produkte for t in perioden),
        name="c7_wechsel_aus_k",
    )

    # (8) Anfangsbedingungen Rüstzustand und Anfangslager:
    #     omega_{i0,0} = 1 ; omega_{k,0} = 0 fuer k != i0 ; y_{k,0} gegeben.
    modell.addConstr(omega_kt[i0, 0] == 1, name="c8_omega0_start")
    modell.addConstrs(
        (omega_kt[k, 0] == 0 for k in produkte if k != i0),
        name="c8_omega0_rest",
    )
    modell.addConstrs(
        (y_kt[k, 0] == daten.y_0[k] for k in produkte),
        name="c8_anfangslager",
    )

    # (9) Eindeutiger Leistungszustand: z^{an}_t + z^{sb}_t + z^{aus}_t = 1
    modell.addConstrs(
        (z_an_t[t] + z_sb_t[t] + z_aus_t[t] == 1 for t in perioden),
        name="c9_leistungszustand_eindeutig",
    )

    # (10) An-Zustand nur bei Produktion: Sum_k x_{kt} <= M_t z^{an}_t
    #      (Rüsten ist vom Leistungszustand entkoppelt; keine Kopplung von chi an z)
    modell.addConstrs(
        (gp.quicksum(x_kt[k, t] for k in produkte) <= M_t[t] * z_an_t[t]
         for t in perioden),
        name="c10_an_bei_produktion",
    )

    # (11) Anschaltvorgang erkennen (Aus -> Ein): u_t >= z^{aus}_{t-1} - z^{aus}_t
    modell.addConstrs(
        (u_t[t] >= z_aus_t[t - 1] - z_aus_t[t] for t in perioden),
        name="c11_anschaltvorgang",
    )

    # (12) Gesamtemission der Periode:
    #      E_t = Sum_k e^p_k x_{kt}
    #            + Sum_i Sum_k (e^{fix}_{ik} + e^{var} tr_{ik}) chi_{ikt}
    #            + Sum_k e^l_k y_{kt}
    #            + e^{on} z^{an}_t + e^{sb} z^{sb}_t + e^{aus} z^{aus}_t + e^{an} u_t
    modell.addConstrs(
        (E_t[t] ==
         gp.quicksum(e_p[k] * x_kt[k, t] for k in produkte)
         + gp.quicksum((e_fix[i, k] + e_var * daten.tr[i, k]) * chi_ikt[i, k, t]
                       for (i, k) in wechsel)
         + gp.quicksum(e_l[k] * y_kt[k, t] for k in produkte)
         + e_on * z_an_t[t] + e_sb * z_sb_t[t] + e_aus * z_aus_t[t] + e_an * u_t[t]
         for t in perioden),
        name="c12_emission",
    )

    # (13) Zertifikatssaldo: J_t = J_{t-1} + A_t + B_t - S_t - E_t
    #      Anfangssaldo J_0 wird über die fixierte Variable J_{0} bereitgestellt (A2).
    modell.addConstr(J_t[0] == daten.J_0, name="c13_anfangssaldo_J0")
    modell.addConstrs(
        (J_t[t] == J_t[t - 1] + daten.A[t] + B_t[t] - S_t[t] - E_t[t]
         for t in perioden),
        name="c13_zertifikatssaldo",
    )

    # (14) Nichtnegativer Zertifikatssaldo: J_t >= 0
    modell.addConstrs(
        (J_t[t] >= 0 for t in perioden),
        name="c14_saldo_nichtnegativ",
    )

    # (15) Anfangsbedingung Leistungszustand: z^{aus}_0 = 0
    modell.addConstr(z_aus_t[0] == 0, name="c15_z_aus0")

    modell.update()
    return modell
