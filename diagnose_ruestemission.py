"""AUFGABE 3+4: 2D-Diagnose  Ruestemission (m)  x  {alpha, pi}  auf basis_freielager().

Kernfrage: Werden die Achsen alpha (Cap-Strenge) und pi (Zertifikatspreis) EMISSIONS-
wirksam, wenn (a) das Lager frei waehlbar ist (basis_freielager) und (b) die Rust-
emissionen substanziell und gestuft sind (m skaliert e_fix, e_var)? Wenn ja: ab welchem m?

Es wird NICHTS am Modell geaendert; hier liegt nur Datensatz-Variation + Analyse.
Echte Signaturen: build_model(daten, reduktion_basismodell=False) -> gp.Model.
Variablennamen: x[k,t], y[k,t], chi[i,k,t], z_sb[t], z_aus[t], E[t], B[t], S[t].
Kalibrierung: pro m wird E_baseline NEU am Vollmodell-Optimum bestimmt und A_t fuer die
alpha-Achse relativ zu DIESEM E_baseline gesetzt (damit alpha ueber m vergleichbar bleibt).
"""

from __future__ import annotations

from dataclasses import replace

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
except Exception:  # Fallback fuer reines Python
    display = print

M_STUFEN = [1, 3, 5, 10, 20]      # AUFGABE 2: Ruestemissions-Skalierung
ALPHAS = [0.7, 0.8, 0.9]          # Cap-Strenge
PIS = [63, 80, 94]                # Zertifikatspreis pi_B (EUR/t)
SCHWELLE_PCT = 1.0                # "emissionswirksam" := E-Spanne > 1 % der Baseline
COLS = ["Variante", "E_kontrolliert", "E_%Basis", "Zukauf", "Verkauf", "Z",
        "n_ruest", "lager_menge", "n_sb", "n_aus", "Status"]


def _val(m, n):
    v = m.getVarByName(n)
    return v.X if v is not None else 0.0


def _status_txt(s):
    return {GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD", GRB.UNBOUNDED: "UNBOUNDED"}.get(s, f"STATUS_{s}")


def _solve(inst):
    m = build_model(inst)
    m.Params.OutputFlag = 0
    m.optimize()
    return m


def _e_total(inst):
    """Gesamtemission am Vollmodell-Optimum (None falls nicht optimal)."""
    m = _solve(inst)
    if m.Status != GRB.OPTIMAL:
        return None
    return sum(_val(m, f"E[{t}]") for t in inst.perioden())


def _metrik(inst, label, e_base):
    """Eine Kennzahlenzeile (loest still). lager_menge = Sum_t Sum_k y_{kt}."""
    m = _solve(inst)
    if m.Status != GRB.OPTIMAL:
        return {"Variante": label, "E_kontrolliert": None, "E_%Basis": None,
                "Zukauf": None, "Verkauf": None, "Z": None, "n_ruest": None,
                "lager_menge": None, "n_sb": None, "n_aus": None,
                "Status": _status_txt(m.Status)}
    T = list(inst.perioden())
    P = list(inst.produkte())
    W = inst.wechsel_paare()
    E = sum(_val(m, f"E[{t}]") for t in T)
    return {
        "Variante": label,
        "E_kontrolliert": round(E, 4),
        "E_%Basis": round(100.0 * E / e_base, 2) if e_base else None,
        "Zukauf": round(sum(_val(m, f"B[{t}]") for t in T), 3),
        "Verkauf": round(sum(_val(m, f"S[{t}]") for t in T), 3),
        "Z": round(m.ObjVal, 2),
        "n_ruest": sum(1 for (i, k) in W for t in T if _val(m, f"chi[{i},{k},{t}]") > 0.5),
        "lager_menge": round(sum(_val(m, f"y[{k},{t}]") for k in P for t in T), 3),
        "n_sb": sum(1 for t in T if _val(m, f"z_sb[{t}]") > 0.5),
        "n_aus": sum(1 for t in T if _val(m, f"z_aus[{t}]") > 0.5),
        "Status": "OPTIMAL",
    }


def _spanne(rows, feld="E_kontrolliert"):
    vs = [r[feld] for r in rows if isinstance(r[feld], (int, float))]
    return (max(vs) - min(vs)) if vs else 0.0


def _reagiert(rows, feld):
    vs = [r[feld] for r in rows if isinstance(r[feld], (int, float))]
    return (max(vs) - min(vs) > 1e-6) if vs else False


def aufgabe1(K=4, T=8):
    """AUFGABE 1: zeigt, dass freies Lager die erzwungene Vorproduktion aufloest."""
    print("=" * 100)
    print("AUFGABE 1 - Lager freigeben (Kapazitaetsspitze d_(3,3)=70 aufloesen)")
    print("=" * 100)
    zeilen = []
    for name, inst in [("basis()  (b=60)", ds.basis(K, T)),
                       ("basis_freielager()  (b=80)", ds.basis_freielager(K, T))]:
        m = _solve(inst)
        P, Ts = list(inst.produkte()), list(inst.perioden())
        lager = sum(_val(m, f"y[{k},{t}]") for k in P for t in Ts)
        E = sum(_val(m, f"E[{t}]") for t in Ts)
        ys = {f"y[{k},{t}]": round(_val(m, f"y[{k},{t}]"), 1)
              for k in P for t in Ts if _val(m, f"y[{k},{t}]") > 1e-6}
        zeilen.append({"Instanz": name, "b_t": inst.b[1], "Z": round(m.ObjVal, 2),
                       "lager_menge": round(lager, 2), "E_total": round(E, 4),
                       "Lager>0": ys or "-"})
    display(pd.DataFrame(zeilen))
    print("Befund: b_t=80 (> groesste Spitze 70) -> Vorproduktion entfaellt, "
          "lager_menge faellt auf 0 (just-in-time), solange Ruestemission/-kosten klein.\n")


def run(K=4, T=8):
    aufgabe1(K, T)

    uebersicht = []
    for mstu in M_STUFEN:
        base_m = ds.variante_ruestemission(mstu, ds.basis_freielager(K, T))
        e_base = _e_total(base_m)               # AUFGABE 3: Baseline pro m neu bestimmen
        if e_base is None:
            print(f"m={mstu}: Baseline nicht optimal - uebersprungen.")
            continue

        # Achsen, jeweils auf DIESES e_base kalibriert
        alpha_rows = [
            _metrik(replace(base_m, A=ds._A_konstant(a, T, e_base=e_base)),
                    f"alpha={a}", e_base) for a in ALPHAS
        ]
        base_cal = replace(base_m, A=ds._A_konstant(ds.ALPHA_BASIS, T, e_base=e_base))
        pi_rows = [
            _metrik(replace(base_cal, pi_B=p, pi_S=p * (1.0 - ds.SPREAD_BASIS)),
                    f"pi_B={p}", e_base) for p in PIS
        ]

        print("=" * 100)
        print(f"m = {mstu:2d}   (e_fix = {mstu * 0.1:.2f} uniform,  e_var = {mstu * 0.02:.3f})"
              f"   E_baseline(m) = {e_base:.4f} t")
        print("=" * 100)
        print(f"[m={mstu}] ALPHA-Achse (Cap-Strenge),  A_t = alpha * E_baseline / T:")
        display(pd.DataFrame(alpha_rows)[COLS])
        print(f"[m={mstu}] PI-Achse (Zertifikatspreis),  alpha = {ds.ALPHA_BASIS} fix:")
        display(pd.DataFrame(pi_rows)[COLS])
        print()

        sa, sp = _spanne(alpha_rows), _spanne(pi_rows)
        uebersicht.append({
            "m": mstu,
            "E_baseline": round(e_base, 4),
            "E-Spanne_alpha_%": round(100.0 * sa / e_base, 3),
            "E-Spanne_pi_%": round(100.0 * sp / e_base, 3),
            "n_ruest_reagiert": "ja" if (_reagiert(alpha_rows, "n_ruest")
                                         or _reagiert(pi_rows, "n_ruest")) else "nein",
            "lager_reagiert": "ja" if (_reagiert(alpha_rows, "lager_menge")
                                       or _reagiert(pi_rows, "lager_menge")) else "nein",
            "alpha_wirksam": "JA" if 100.0 * sa / e_base > SCHWELLE_PCT else "nein",
            "pi_wirksam": "JA" if 100.0 * sp / e_base > SCHWELLE_PCT else "nein",
        })

    print("=" * 100)
    print("AUFGABE 4 - UEBERSICHT: ab welchem m werden alpha / pi emissionswirksam?")
    print(f"(emissionswirksam := E-Spanne ueber die Achse > {SCHWELLE_PCT:.0f} % der Baseline)")
    print("=" * 100)
    dfu = pd.DataFrame(uebersicht)[
        ["m", "E_baseline", "E-Spanne_alpha_%", "E-Spanne_pi_%",
         "n_ruest_reagiert", "lager_reagiert", "alpha_wirksam", "pi_wirksam"]
    ]
    display(dfu)

    def _schwelle(feld):
        for r in uebersicht:
            if r[feld] == "JA":
                return r["m"]
        return None

    sa_m, sp_m = _schwelle("alpha_wirksam"), _schwelle("pi_wirksam")
    print()
    print(f"SCHWELLE  alpha emissionswirksam ab m = "
          f"{sa_m if sa_m is not None else 'NICHT erreicht (auch bei m=20 <= 1 %)'}")
    print(f"SCHWELLE  pi    emissionswirksam ab m = "
          f"{sp_m if sp_m is not None else 'NICHT erreicht (auch bei m=20 <= 1 %)'}")

    if sa_m is None and sp_m is None:
        any_ruest = any(r["n_ruest_reagiert"] == "ja" for r in uebersicht)
        any_lager = any(r["lager_reagiert"] == "ja" for r in uebersicht)
        print("\nBEFUND: Selbst bei m=20 bewegt sich E ueber alpha/pi nicht > 1 %.")
        print(f"  n_ruest reagiert auf einer Achse: {'JA' if any_ruest else 'NEIN'};  "
              f"lager_menge reagiert: {'JA' if any_lager else 'NEIN'}.")
        if not any_ruest and not any_lager:
            print("  URSACHE: Die Ruestfolge ist durch die Bedarfsstruktur (eine Bedarfsspitze")
            print("  je Periode; nur P1 doppelt, aber nicht buendelbar wegen Kapazitaet/Lagerkosten)")
            print("  weiterhin FIXIERT (n_ruest konstant). Freies Lager beseitigt die ERZWUNGENE")
            print("  Vorproduktion, schafft aber keinen DISKRETIONAEREN Buendelungsspielraum.")
            print("  Ohne Rust-Lager-Trade-off bleibt die Setup-Emission ein FIXER E-Anteil ->")
            print("  alpha/pi wirken nur auf Zukauf/Verkauf/Kosten, nicht auf die Emission.")
            print("  REMEDY (nicht umgesetzt): Bedarf mit MEHRFACH-Nachfrage je Produkt (echter")
            print("  Buendelungsspielraum), damit hohe Ruestemission Losgroessen-Buendelung lohnt.")
    return uebersicht


if __name__ == "__main__":
    run()
