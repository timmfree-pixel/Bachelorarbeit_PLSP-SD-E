"""Reines PLSP-SD-Basismodell (nur Restriktionen (1)-(8)).

Dieses Modul dient AUSSCHLIESSLICH der Validierung: Es baut das klassische
Proportional Lot-Sizing and Scheduling Problem mit Sequenzabhängigen Rüstzeiten
(ohne jede Emissions-/Zertifikatserweiterung) als eigenständiges Referenzmodell.

Gegen dieses Referenzmodell wird der Reduktionsschalter
``modell.build_model(daten, reduktion_basismodell=True)`` geprüft: Bei abgeschalteten
Emissionen müssen beide Modelle denselben optimalen Zielwert liefern
(Validierung gegen Kaczmarczyk-artige Instanzen).

Es werden bewusst nur die Variablen x, y, omega, chi und die Restriktionen (2)-(8)
sowie die Zielfunktion (1) ohne Zertifikatsterm modelliert.
"""

from __future__ import annotations

import gurobipy as gp
from gurobipy import GRB

from instanz import Instanz


def build_plsp_sd_basis(daten: Instanz) -> gp.Model:
    """Baut das reine PLSP-SD (Restriktionen (1)-(8)) ohne Emissionserweiterung.

    Identische Variablen-/Restriktionsdefinitionen wie im Kern des PLSP-SD-E, jedoch
    ohne Leistungszustände, Anschaltvorgang, Emissionen und Zertifikatshandel. Dient
    als Referenz für den Reduktionstest.
    """
    produkte = daten.produkte()
    perioden = daten.perioden()
    perioden0 = daten.perioden_mit_null()
    wechsel = daten.wechsel_paare()  # (i, k) mit i != k, vgl. A3 in modell.py
    K, T, i0 = daten.K, daten.T, daten.i_0

    # Big-M wie im Vollmodell: M_{kt} = min(b_t/tb_k, Sum_{tau>=t} d_{k,tau})
    M_kt = {
        (k, t): min(
            daten.b[t] / daten.tb[k],
            sum(daten.d[k, tau] for tau in range(t, T + 1)),
        )
        for k in produkte
        for t in perioden
    }

    modell = gp.Model("PLSP-SD")

    x_kt = modell.addVars(produkte, perioden, lb=0.0, name="x")
    y_kt = modell.addVars(produkte, perioden0, lb=0.0, name="y")
    omega_kt = modell.addVars(produkte, perioden0, vtype=GRB.BINARY, name="omega")
    chi_ikt = modell.addVars(
        [(i, k, t) for (i, k) in wechsel for t in perioden], lb=0.0, name="chi"
    )

    # (1) Zielfunktion ohne Zertifikatsterm: min Sum_t Sum_k (h_k y_{kt} + Sum_i s_k chi_{ikt})
    modell.setObjective(
        gp.quicksum(daten.h[k] * y_kt[k, t] for k in produkte for t in perioden)
        + gp.quicksum(daten.s[k] * chi_ikt[i, k, t] for (i, k) in wechsel for t in perioden),
        GRB.MINIMIZE,
    )

    # (2) Lagerbilanz
    modell.addConstrs(
        (y_kt[k, t - 1] + x_kt[k, t] - y_kt[k, t] == daten.d[k, t]
         for k in produkte for t in perioden),
        name="c2_lagerbilanz",
    )
    # (3) Kapazität
    modell.addConstrs(
        (gp.quicksum(daten.tb[k] * x_kt[k, t] for k in produkte)
         + gp.quicksum(daten.tr[i, k] * chi_ikt[i, k, t] for (i, k) in wechsel)
         <= daten.b[t]
         for t in perioden),
        name="c3_kapazitaet",
    )
    # (4) Eindeutiger Rüstzustand
    modell.addConstrs(
        (gp.quicksum(omega_kt[k, t] for k in produkte) == 1 for t in perioden),
        name="c4_ruestzustand_eindeutig",
    )
    # (5) Produktion nur bei (vor-)gerüstetem Zustand
    modell.addConstrs(
        (x_kt[k, t] <= M_kt[k, t] * (omega_kt[k, t - 1] + omega_kt[k, t])
         for k in produkte for t in perioden),
        name="c5_produktion_nur_geruestet",
    )
    # (6) Rüstzustandsfortschreibung (Wechsel IN k)
    modell.addConstrs(
        (gp.quicksum(chi_ikt[i, k, t] for i in produkte if i != k)
         >= omega_kt[k, t] - omega_kt[k, t - 1]
         for k in produkte for t in perioden),
        name="c6_wechsel_in_k",
    )
    # (7) Spiegelbild (Wechsel AUS k)
    modell.addConstrs(
        (gp.quicksum(chi_ikt[k, i, t] for i in produkte if i != k)
         <= omega_kt[k, t - 1]
         for k in produkte for t in perioden),
        name="c7_wechsel_aus_k",
    )
    # (8) Anfangsbedingungen
    modell.addConstr(omega_kt[i0, 0] == 1, name="c8_omega0_start")
    modell.addConstrs(
        (omega_kt[k, 0] == 0 for k in produkte if k != i0), name="c8_omega0_rest"
    )
    modell.addConstrs(
        (y_kt[k, 0] == daten.y_0[k] for k in produkte), name="c8_anfangslager"
    )

    modell.update()
    return modell
