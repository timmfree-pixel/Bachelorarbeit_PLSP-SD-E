"""Diagnostischer Sensitivitaets-Durchlauf fuer den PLSP-SD-E-Datensatz.

ZWECK (diagnostisch, nicht final): Pruefen, ob der literaturgestuetzte Datensatz aus
``datensatz_plsp_sd_e.py`` analysefaehig ist - bewegt sich die Loesung ueber die
Parameterachsen sichtbar/interpretierbar, oder bleibt sie flach/kippt ins Triviale?
Pro Achse 2-3 Punkte. Das Modell wird NICHT veraendert; hier liegt nur Analyse-Code.

Echte Signaturen (aus modell.py / colab_plsp_sd_e.py):
  build_model(daten, reduktion_basismodell=False) -> gp.Model   (gibt NUR das Modell)
Variablennamen im Modell: x[k,t], y[k,t], omega[k,t], chi[i,k,t], z_an[t], z_sb[t],
  z_aus[t], u[t], E[t], J[t], B[t], S[t].

Hinweis zu AUFGABE 1: ``kennzahlen`` repliziert die Variablen-Extraktion aus
``report_loesung`` (Zugriff per getVarByName), unterdrueckt aber jede Ausgabe - es wird
KEINE Periodentabelle je Lauf gedruckt.

Hinweis zu AUFGABE 2: Mein Reduktionsschalter fixiert z_an NICHT (struktureller Befund
(c) aus dem Reduktionstest). ``referenz_naiv`` setzt daher z_an_t.lb = 1 direkt auf dem
gebauten Modell (Dauer-An-Politik) - funktional die gemeinte naive Referenz.
"""

from __future__ import annotations

from dataclasses import replace

import gurobipy as gp
from gurobipy import GRB
import pandas as pd

from modell import build_model
import datensatz_plsp_sd_e as ds

# --- Colab-Konventionen: alle Spalten/Zeilen sichtbar, display() statt print -------
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
try:
    from IPython.display import display
except Exception:  # pragma: no cover - Fallback fuer reines Python
    display = print


def _val(model, name):
    """Variablenwert per Name (0.0, falls Variable nicht existiert)."""
    v = model.getVarByName(name)
    return v.X if v is not None else 0.0


def _status_txt(status):
    return {GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD", GRB.UNBOUNDED: "UNBOUNDED"}.get(
        status, f"STATUS_{status}")


# =============================================================================
# AUFGABE 1 - Leiser Kennzahlen-Extraktor
# =============================================================================
def kennzahlen(inst, label):
    """Loest inst (OutputFlag=0), prueft Status und liefert ein Kennzahlen-dict
    OHNE die volle Periodentabelle zu drucken.

    Bei nicht-optimalem Status: status zurueckgeben, alle Kennzahlen = None.
    """
    m = build_model(inst)                 # reduktion_basismodell=False (Vollmodell)
    m.Params.OutputFlag = 0
    m.optimize()

    leer = {"label": label, "status": _status_txt(m.Status),
            "E_total": None, "zukauf": None, "verkauf": None, "Z": None,
            "n_an": None, "n_sb": None, "n_aus": None, "n_anlauf": None,
            "n_ruest": None, "lager_em": None}
    if m.Status != GRB.OPTIMAL:
        return leer

    perioden = list(inst.perioden())
    produkte = list(inst.produkte())
    wechsel = inst.wechsel_paare()

    E_total = sum(_val(m, f"E[{t}]") for t in perioden)
    zukauf = sum(_val(m, f"B[{t}]") for t in perioden)
    verkauf = sum(_val(m, f"S[{t}]") for t in perioden)
    n_an = sum(1 for t in perioden if _val(m, f"z_an[{t}]") > 0.5)
    n_sb = sum(1 for t in perioden if _val(m, f"z_sb[{t}]") > 0.5)
    n_aus = sum(1 for t in perioden if _val(m, f"z_aus[{t}]") > 0.5)
    n_anlauf = sum(_val(m, f"u[{t}]") for t in perioden)
    n_ruest = sum(1 for (i, k) in wechsel for t in perioden
                  if _val(m, f"chi[{i},{k},{t}]") > 0.5)
    lager_em = sum(inst.e_l[k] * _val(m, f"y[{k},{t}]")
                   for k in produkte for t in perioden)

    return {"label": label, "status": "OPTIMAL",
            "E_total": E_total, "zukauf": zukauf, "verkauf": verkauf, "Z": m.ObjVal,
            "n_an": n_an, "n_sb": n_sb, "n_aus": n_aus,
            "n_anlauf": round(n_anlauf, 3), "n_ruest": n_ruest, "lager_em": lager_em}


# =============================================================================
# AUFGABE 2 - Naive PLSP-SD-Referenz (bewusste Referenzdefinition)
# =============================================================================
def referenz_naiv(inst):
    """Emission der kostenminimalen Losgroessenloesung OHNE Emissionsmanagement.

    Referenzdefinition (losgroessen-optimal, aber ohne Standby/Aus-Management):
      - Zertifikate neutralisieren: pi_B = pi_S = 0; Cap nicht-bindend (A_t = 1e6)
        -> Zielfunktion reduziert auf Ruest- + Lagerkosten -> kostenminimaler
        Zeitplan (entspricht der reinen PLSP-SD-Loesung).
      - Maschine NICHT strategisch abschalten: z_an_t.lb = 1 fuer alle Perioden bis
        zur letzten Produktionsperiode (naive Dauer-An-Politik).
      - Emission mit den ECHTEN (nicht neutralisierten) e_*-Koeffizienten auslesen.

    Rueckgabe: (E_referenz_oder_None, status_txt).
    """
    neutral = replace(inst, pi_B=0.0, pi_S=0.0,
                      A={t: 1e6 for t in inst.perioden()})
    m = build_model(neutral)              # echte e_*-Koeffizienten bleiben erhalten
    m.Params.OutputFlag = 0

    # letzte Produktionsperiode = spaeteste Periode mit positivem Bedarf
    letzte_prod = max((t for (k, t), v in inst.d.items() if v > 0), default=inst.T)
    for t in range(1, letzte_prod + 1):   # Dauer-An: z_an_t.lb = 1
        v = m.getVarByName(f"z_an[{t}]")
        if v is not None:
            v.lb = 1.0
    m.update()
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        return None, _status_txt(m.Status)
    E_referenz = sum(_val(m, f"E[{t}]") for t in inst.perioden())
    return E_referenz, "OPTIMAL"


# =============================================================================
# AUFGABE 3 - Achsen-Durchlauf
# =============================================================================
def _zeile(label, inst):
    """Eine Tabellenzeile: kennzahlen(kontrolliert) + referenz_naiv + Delta_E."""
    k = kennzahlen(inst, label)
    E_ref, ref_status = referenz_naiv(inst)
    E_ctrl = k["E_total"]
    delta = (E_ref - E_ctrl) if (E_ref is not None and E_ctrl is not None) else None

    def r(x, n=3):
        return round(x, n) if isinstance(x, (int, float)) else x

    return {
        "Variante": label,
        "E_kontrolliert": r(E_ctrl),
        "E_referenz": r(E_ref),
        "Delta_E": r(delta),
        "Zukauf": r(k["zukauf"]),
        "Verkauf": r(k["verkauf"]),
        "Z": r(k["Z"], 2),
        "n_sb": k["n_sb"],
        "n_aus": k["n_aus"],
        "n_anlauf": k["n_anlauf"],
        "n_ruest": k["n_ruest"],
        "Status": k["status"] if ref_status == "OPTIMAL" else f"{k['status']}/ref:{ref_status}",
    }


def _achsen():
    """Definition der fuenf Achsen mit je 2-3 Punkten (label, Instanz)."""
    il = ds.variante_lange_luecke()
    return {
        "alpha (Cap-Strenge)": [
            ("alpha=0.7", ds.variante_alpha(0.7)),
            ("alpha=0.8 (Basis)", ds.basis()),
            ("alpha=0.9", ds.variante_alpha(0.9)),
        ],
        "pi (Zertifikatspreis)": [
            ("pi_B=63", ds.variante_preise(63)),
            ("pi_B=80 (Basis)", ds.basis()),
            ("pi_B=94", ds.variante_preise(94)),
        ],
        "g (Anlauffaktor)": [
            ("g=2", ds.variante_g(2)),
            ("g=3", ds.variante_g(3)),
            ("g=4 (Basis)", ds.basis()),
        ],
        "Leerlauf (Lueckenlaenge)": [
            ("L=2 (Basis)", ds.basis()),
            ("L=4 (lange Luecke)", il),
            ("L=4, e_an=3*e_sb", replace(il, e_an=3 * il.e_sb)),
        ],
        "rho (Kapazitaet)": [
            ("b=50 (rho~0.67)", ds.variante_rho(50)),
            ("b=60 (Basis, rho~0.56)", ds.basis()),
            ("b=70 (rho~0.48)", ds.variante_rho(70)),
        ],
    }


def run_diagnose():
    """Fuehrt AUFGABE 3 (Achsen-Tabellen) und AUFGABE 4 (Kurzinterpretation) aus."""
    achsen = _achsen()
    ergebnisse = {}

    for achse, punkte in achsen.items():
        print("=" * 100)
        print(f"ACHSE: {achse}")
        print("=" * 100)
        zeilen = [_zeile(label, inst) for label, inst in punkte]
        df = pd.DataFrame(zeilen)[
            ["Variante", "E_kontrolliert", "E_referenz", "Delta_E", "Zukauf",
             "Verkauf", "Z", "n_sb", "n_aus", "n_anlauf", "n_ruest", "Status"]
        ]
        display(df)
        ergebnisse[achse] = zeilen
        print()

    _interpretation(ergebnisse)
    return ergebnisse


# =============================================================================
# AUFGABE 4 - Kurzinterpretation je Achse (datengestuetzt)
# =============================================================================
def _interpretation(ergebnisse):
    print("=" * 100)
    print("AUFGABE 4 - KURZINTERPRETATION JE ACHSE")
    print("=" * 100)

    def num(zeilen, feld):
        return [z[feld] for z in zeilen if isinstance(z[feld], (int, float))]

    for achse, zeilen in ergebnisse.items():
        Es = num(zeilen, "E_kontrolliert")
        spann = (max(Es) - min(Es)) if Es else 0.0
        mittel = (sum(Es) / len(Es)) if Es else 0.0
        rel = (spann / mittel) if mittel else 0.0
        flach = rel < 0.02   # < 2 % relative Bewegung -> flach
        bewegung = (f"E_kontrolliert {min(Es):.3f}..{max(Es):.3f} "
                    f"(Spanne {spann:.3f}, {rel*100:.1f} %)") if Es else "keine optimale Loesung"

        deltas = num(zeilen, "Delta_E")
        delta_pos = any(d > 1e-9 for d in deltas)

        n_sb = num(zeilen, "n_sb")
        n_aus = num(zeilen, "n_aus")
        umschlag = (any(s > 0 for s in n_sb) and any(a > 0 for a in n_aus))

        print(f"\n# {achse}")
        print(f"  Bewegung: {bewegung} -> {'FLACH (Nachkalibrierung pruefen)' if flach else 'bewegt (analysefaehig)'}")
        if achse.startswith("g") or achse.startswith("Leerlauf"):
            print(f"  Standby->Aus-Umschlag: {'JA' if umschlag else 'NEIN'} "
                  f"(n_sb je Punkt={n_sb}, n_aus je Punkt={n_aus})")
        if deltas:
            print(f"  Delta_E>0 (PLSP-SD-E senkt ggü. naiver Referenz) in >=1 Variante: "
                  f"{'JA' if delta_pos else 'NEIN - explizit gemeldet'} "
                  f"(Delta_E={[round(d,3) for d in deltas]})")
        else:
            print("  Delta_E nicht bestimmbar (kein optimaler Lauf).")


if __name__ == "__main__":
    run_diagnose()
