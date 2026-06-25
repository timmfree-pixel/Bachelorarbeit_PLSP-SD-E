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


# =============================================================================
# Konsistenz-Checks (repliziert aus report_loesung; report_loesung liegt in der
# Colab-Konsolidierung, die wegen der !pip-Zeile nicht als Modul importierbar ist)
# =============================================================================
def _konsistenz_checks(m, inst):
    """Die 7 Checks aus report_loesung; Rueckgabe (anzahl_ok, [(name, ok)])."""
    P = list(inst.produkte())
    Ts = list(inst.perioden())
    W = inst.wechsel_paare()
    eps = 1e-6

    def v(n):
        return _val(m, n)

    res = []
    # 1) z_an + z_sb + z_aus == 1
    res.append(("(1) z_an+z_sb+z_aus==1", all(
        abs(v(f"z_an[{t}]") + v(f"z_sb[{t}]") + v(f"z_aus[{t}]") - 1) < eps for t in Ts)))
    # 2) x_{k,t} > 0 => z_an_t == 1
    res.append(("(2) x>0 => z_an=1", all(
        not (sum(v(f"x[{k},{t}]") for k in P) > eps) or v(f"z_an[{t}]") > 1 - eps for t in Ts)))
    # 3) u_t >= z_aus_{t-1} - z_aus_t
    res.append(("(3) u_t >= z_aus_{t-1}-z_aus_t", all(
        v(f"u[{t}]") >= v(f"z_aus[{t-1}]") - v(f"z_aus[{t}]") - eps for t in Ts)))
    # 4) omega aendert sich nur durch einen Ruestwechsel chi
    ok4 = True
    for t in Ts:
        geaendert = any(abs(v(f"omega[{k},{t}]") - v(f"omega[{k},{t-1}]")) > eps for k in P)
        chi_aktiv = sum(v(f"chi[{i},{k},{t}]") for (i, k) in W) > 0.5
        if geaendert and not chi_aktiv:
            ok4 = False
    res.append(("(4) omega nur via chi", ok4))
    # 5) kein Self-Loop chi_{k,k,t}
    res.append(("(5) kein Self-Loop chi_kkt", all(
        (m.getVarByName(f"chi[{k},{k},{t}]") is None
         or abs(m.getVarByName(f"chi[{k},{k},{t}]").X) < eps) for k in P for t in Ts)))
    # 6) Zertifikatebilanz
    res.append(("(6) J_t == J_{t-1}+A+B-S-E", all(
        abs(v(f"J[{t}]") - (v(f"J[{t-1}]") + inst.A[t] + v(f"B[{t}]") - v(f"S[{t}]") - v(f"E[{t}]"))) < 1e-4
        for t in Ts)))
    # 7) J_t >= 0
    res.append(("(7) J_t >= 0", all(v(f"J[{t}]") >= -eps for t in Ts)))
    return sum(1 for _, ok in res if ok), res


# =============================================================================
# AUFGABE 2 - E_baseline messen (Vollmodell-Optimum der ungestoerten Basis)
# =============================================================================
def e_baseline_messen(inst=None):
    """Misst E_baseline am Vollmodell-Optimum der (ungestoerten) Basis und gibt es aus.

    E ist ueber alpha/pi flach (verifiziert), daher ist das Vollmodell-Optimum eine
    saubere, reproduzierbare Baseline. (A_t=1e6 ist UNGEEIGNET: der Zielwert wird dann
    numerisch vom Zertifikatshandel dominiert und Emission/Kosten verschwinden im
    Rundungsrauschen.)
    """
    inst = inst or ds.basis()
    k = kennzahlen(inst, "basis")
    E = k["E_total"]
    print("=" * 100)
    print("AUFGABE 2 - E_baseline (gemessen) und A_t-Neukalibrierung")
    print("=" * 100)
    if E is None:
        print(f"Basis nicht optimal (Status={k['status']}).")
        return None
    a_t = ds.ALPHA_BASIS * E / inst.T
    print(f"E_baseline_neu (Vollmodell-Optimum) = {E:.4f} t CO2e")
    print(f"Modul-Konstante E_BASELINE_EST       = {ds.E_BASELINE_EST}  "
          f"({'stimmt ueberein' if abs(E - ds.E_BASELINE_EST) < 0.01 else 'WEICHT AB -> Konstante anpassen!'})")
    print(f"Neues A_t = alpha*E_baseline/T = {ds.ALPHA_BASIS}*{E:.2f}/{inst.T} = {a_t:.4f} je Periode "
          f"(Summe {ds.ALPHA_BASIS * E:.3f}); J_0 = {inst.J_0}")
    print(f"Kontrolle: basis().A[1] = {ds.basis().A[1]:.4f}")
    return E


# =============================================================================
# AUFGABE 3 - Verifikation der neuen Basis
# =============================================================================
def verifiziere_basis():
    """Loest die neue Basis (Vollmodell) und bestaetigt (i)-(iv)."""
    inst = ds.basis()
    m = build_model(inst)
    m.Params.OutputFlag = 0
    m.optimize()
    print("\n" + "=" * 100)
    print("AUFGABE 3 - Verifikation der neuen Basis (L=3)")
    print("=" * 100)
    if m.Status != GRB.OPTIMAL:
        print(f"FEHLER: Basis nicht optimal (Status={_status_txt(m.Status)}).")
        return

    Ts = list(inst.perioden())
    P = list(inst.produkte())
    leerlauf = [t for t in Ts if sum(_val(m, f"x[{k},{t}]") for k in P) < 1e-6]
    n_sb = sum(1 for t in Ts if _val(m, f"z_sb[{t}]") > 0.5)
    n_aus = sum(1 for t in Ts if _val(m, f"z_aus[{t}]") > 0.5)
    y_max = max(_val(m, f"y[{k},{t}]") for k in P for t in Ts)
    lager_em = sum(inst.e_l[k] * _val(m, f"y[{k},{t}]") for k in P for t in Ts)
    n_ok, checks = _konsistenz_checks(m, inst)

    ok_i = (leerlauf == [4, 5, 6])
    ok_ii = (n_aus == 0 and n_sb == 3)
    ok_iii = (y_max > 1e-6)
    ok_iv = (n_ok == 7)
    print(f"  (i)   drei Leerlaufperioden t4-t6:           {'OK' if ok_i else 'FEHLT'} "
          f"(leerlauf-Perioden = {leerlauf})")
    print(f"  (ii)  ungestoert STANDBY in t4-t6, n_aus=0:  {'OK' if ok_ii else 'FEHLT'} "
          f"(n_sb={n_sb}, n_aus={n_aus})")
    print(f"  (iii) Lager aktiv (Vorproduktion P3-Spitze): {'OK' if ok_iii else 'FEHLT'} "
          f"(y_max={y_max:.1f}, Lageremission={lager_em:.3f})")
    print(f"  (iv)  alle 7 Konsistenz-Checks:              {'OK' if ok_iv else 'FEHLT'} "
          f"({n_ok}/7)")
    if not ok_iv:
        for name, ok in checks:
            if not ok:
                print(f"          FEHLER bei {name}")


# =============================================================================
# AUFGABE 4 - alpha-/pi-Durchlauf wiederholen (Kernfrage: Emissionseffekt?)
# =============================================================================
def run_alpha_pi():
    """AUFGABE 4: nur Achsen alpha {0.7,0.8,0.9} und pi {63,80,94}; Tabelle + Deutung."""
    achsen = {
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
    }
    print("\n" + "=" * 100)
    print("AUFGABE 4 - alpha-/pi-Durchlauf auf der NEUEN L=3-Basis")
    print("=" * 100)
    for achse, punkte in achsen.items():
        print(f"\nACHSE: {achse}")
        zeilen = [_zeile(label, inst) for label, inst in punkte]
        df = pd.DataFrame(zeilen)[
            ["Variante", "E_kontrolliert", "E_referenz", "Delta_E", "Zukauf",
             "Verkauf", "Z", "n_sb", "n_aus", "n_anlauf", "n_ruest", "Status"]
        ]
        display(df)
        _interpret_alpha_pi(achse, zeilen)


def _interpret_alpha_pi(achse, zeilen):
    Es = [z["E_kontrolliert"] for z in zeilen if isinstance(z["E_kontrolliert"], (int, float))]
    naus = [z["n_aus"] for z in zeilen if isinstance(z["n_aus"], (int, float))]
    spann = (max(Es) - min(Es)) if Es else 0.0
    rel = (spann / (sum(Es) / len(Es))) if Es and sum(Es) else 0.0
    bewegt = rel >= 0.02
    abschalt_aktiv = any(a > 0 for a in naus)

    print(f"  - Bewegt sich E_kontrolliert? {'JA (analysefaehig)' if bewegt else 'NEIN - FLACH'} "
          f"(E {min(Es):.3f}..{max(Es):.3f}, Spanne {spann:.3f} = {rel*100:.1f} %)")
    print(f"  - Abschalten unter Druck (n_aus>0)? {'JA' if abschalt_aktiv else 'NEIN'} "
          f"(n_aus je Punkt = {naus})")
    if not bewegt:
        print("  - BEFUND: Achse weiterhin FLACH in der Emission (nur Kosten/Zukauf/Verkauf "
              "reagieren). Strukturelle Ursache: der Leerlauf-Zustand ist emissions-, nicht "
              "preisdeterminiert; das Modell sitzt bereits am Emissionsboden (bei L=3 Standby, "
              "da Abschalten 1.8 t > Standby 1.35 t emittiert).")
        print("  - Vorgeschlagene naechste Stellschrauben (NICHT umgesetzt): "
              "(a) Luecke L=4; (b) hoehere e_sb/e_an-Relation (g<3, damit Abschalten emissions-"
              "guenstiger als Standby wird). CAVEAT (empirisch belegt): auch L=4/L=5 bewegen E "
              "NICHT - der Zustand kippt zwar Standby->Aus, bleibt aber emissionsneutral. Ein "
              "echter preiselastischer Minderungshebel (Kosten-gegen-Emission-Abwaegung) fehlt "
              "im realistischen Preisband; alpha/pi sind dann prinzipiell nur Kostenachsen.")


if __name__ == "__main__":
    # Aktuelle Aufgabe: AUFGABE 2-4 auf der neuen L=3-Basis.
    e_baseline_messen()
    verifiziere_basis()
    run_alpha_pi()
    # Der vollstaendige 5-Achsen-Lauf bleibt verfuegbar:
    #   run_diagnose()
