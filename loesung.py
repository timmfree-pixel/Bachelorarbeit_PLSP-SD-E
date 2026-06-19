"""Loesung und Auswertung (dritte, von Modell und Daten getrennte Schicht).

Funktionen zum Optimieren eines gebauten Modells und zum Auslesen/Anzeigen der
Loesung (Produktionsplan, Ruestzustaende, Leistungszustaende, Emissionen,
Zertifikatskonto). Enthaelt keine Modellierungslogik.
"""

from __future__ import annotations

from typing import Dict

import gurobipy as gp
from gurobipy import GRB

from instanz import Instanz


def loese(modell: gp.Model, ausgabe: bool = False) -> int:
    """Optimiert das Modell und gibt den Gurobi-Status zurueck.

    ausgabe=False schaltet das Solver-Log ab (fuer Tests).
    """
    modell.Params.OutputFlag = 1 if ausgabe else 0
    modell.optimize()
    return modell.Status


def variablenwerte(modell: gp.Model, praefix: str) -> Dict[str, float]:
    """Liest alle Variablen mit gegebenem Namenspraefix als {Name: Wert} aus."""
    return {
        v.VarName: v.X
        for v in modell.getVars()
        if v.VarName.startswith(praefix)
    }


def zeige_loesung(modell: gp.Model, daten: Instanz) -> None:
    """Druckt eine kompakte, periodenweise Uebersicht der Loesung (OR-Auswertung)."""
    if modell.Status != GRB.OPTIMAL:
        print(f"Kein optimaler Status (Status={modell.Status}).")
        return

    def val(name: str) -> float:
        var = modell.getVarByName(name)
        return var.X if var is not None else 0.0

    print(f"Zielwert Z = {modell.ObjVal:.4f}")
    print("-" * 72)
    kopf = f"{'t':>2} | {'Produkt':>22} | {'Leistung':>10} | {'u':>3} | {'E_t':>8} | {'J_t':>8}"
    print(kopf)
    print("-" * 72)
    for t in daten.perioden():
        # Produktionsmengen und Ruestzustand
        prod = ", ".join(
            f"x_{k}={val(f'x[{k},{t}]'):.2f}"
            for k in daten.produkte()
            if val(f"x[{k},{t}]") > 1e-6
        ) or "-"
        # Leistungszustand bestimmen
        if val(f"z_an[{t}]") > 0.5:
            zustand = "An"
        elif val(f"z_sb[{t}]") > 0.5:
            zustand = "Standby"
        else:
            zustand = "Aus"
        u = val(f"u[{t}]")
        E = val(f"E[{t}]")
        J = val(f"J[{t}]")
        print(f"{t:>2} | {prod:>22} | {zustand:>10} | {u:>3.0f} | {E:>8.2f} | {J:>8.2f}")
    print("-" * 72)

    # Ruestwechsel auflisten
    wechsel = [
        (i, k, t)
        for (i, k) in daten.wechsel_paare()
        for t in daten.perioden()
        if val(f"chi[{i},{k},{t}]") > 0.5
    ]
    if wechsel:
        print("Ruestwechsel (i->k @ t):",
              ", ".join(f"{i}->{k}@{t}" for (i, k, t) in wechsel))
    else:
        print("Ruestwechsel: keine")
