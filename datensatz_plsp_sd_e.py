"""Literaturgestützter Datensatz für das PLSP-SD-E (Basisinstanz + Sensitivitätsvarianten).

Dieses Modul ERZEUGT konkrete ``Instanz``-Objekte (Schema aus ``instanz.py``); es
enthält keine Modelllogik. In Colab: dieses Modul (und ``instanz.py``) einfügen,
dann z.B. ``inst = basis()`` und ``inst`` an ``build_model`` übergeben.

================================================================================
QUELLEN-ANKER (bewusst minimal gehalten)
================================================================================
[H]  Haase (1994, Diss.) / Haase & Kimms (2000): PLSP-Struktur, Kostenverhältnisse.
     -> h_k = 1 normiert; Rüstkosten s_k über TBO (K = h * d_quer * TBO^2 / 2);
        Kapazitätsauslastung rho; sequenzabhängige Rüstzeiten tr_{ik} (asymmetrisch,
        Dreiecksungleichung).
[RH] Retel Helmrich (2013, Diss. "Green Lot-Sizing", Kap. 3): Emissionsstruktur
     "fix + linear" für Produktion, Rüsten (warm-up) und Lager.
[BBG] Beck/Biel/Glock (2019): Maschinenzustands-Emissionen als Vielfache der
     Standby-Grundlast: Aus = 0, An ~ 1.3x Standby, Anlauf = g x Standby (g = 4).
[EUA] EU-ETS-Marktwert: pi^B ~ 80 EUR/t (EUA-Schlusskurs ~80.48 EUR/t, 23.06.2026,
     EEX / Trading Economics); pi^S = pi^B * (1 - Spread).

Jede NICHT belegte Zahl ist unten als "eigene Setzung" gekennzeichnet.

================================================================================
KALIBRIERUNG (a-priori; mit Baseline-Lauf zu verifizieren)
================================================================================
Größe: K = 4 Produkte, T = 8 Mikroperioden, Startrüstung auf Produkt 1 (i_0 = 1).

Bedarf (eine Bedarfsspitze je Periode; 3-Perioden-Lücke t4-t6; eine kapazitäts-
sprengende Spitze in t3 erzwingt Vorproduktion -> Lager aktiv):
    t1: P1=50   t2: P2=45   t3: P3=70   t4: -   t5: -   t6: -   t7: P4=50   t8: P1=40
  -> Leerlaufperioden t4, t5, t6 (L=3, für Idle/Standby/Aus).
  -> t3-Bedarf (70) > Kapazität (60) erzwingt 10 Einheiten P3-Vorproduktion (in t2
     via Rüst-Carryover; Lageremission/-kosten aktiv, nicht optional).

E_baseline (GEMESSEN am Vollmodell-Optimum der L=3-Basis, nicht mehr geschätzt):
  Produktion : 255 Einheiten * e_p 0.01            = 2.55 t   (fix; Bedarf, kein Backlog)
  Betrieb An : 5 An-Perioden * e_on 0.60           = 3.00 t   (t1,t2,t3,t7,t8)
  Standby    : 3 Perioden (t4,t5,t6) * e_sb 0.45    = 1.35 t
  Aus        : 0                                    = 0.00 t   (L=3 < Schwelle 4 -> Standby)
  Anlauf     : 0 (kein Aus->Ein im Basisfall)       = 0.00 t
  Rüsten     : 4 Wechsel * (e_fix 0.1 + e_var 0.02 * tr 3) = 4 * 0.16 = 0.64 t
  Lager      : 10 Einh. * 1 Periode * e_l(P3) 0.02  = 0.20 t
  --------------------------------------------------------------------
  E_baseline = 7.74 t   (-> Konstante E_BASELINE_EST = 7.74)

Cap / Zuteilung: Summe(A_t) = alpha * E_baseline, alpha = 0.8 (Basis).
  -> A_t konstant ("glatt") = 0.8 * 7.74 / 8 = 0.774 je Periode; J_0 = 0.
  -> Defizit = (1 - 0.8) * 7.74 = 1.548 t muss zugekauft werden (Cap bindet).

Skalen-Kommensurabilität:
  Losgrößenkosten (Basislösung): Rüsten s_2+s_3+s_4+s_1 = 25+39+28+51 = 143
                                 + Lager 10 * 1            = 10   -> ~153
  Zertifikat-Defizitkosten: pi^B 80 * 1.5 t               ~ 120
  -> Beide Seiten ~ gleiche Größenordnung (153 vs. 120); keine dominiert (>70 %).
  Dies ist der Grund für die kleinen Emissionskoeffizienten (Emissionen in t CO2e):
  nur so ist der reale Preis 80 EUR/t mit den normierten Kosten (h=1) kommensurabel.

Banking-Aktivierung: A_t glatt (0.774), Emissionsprofil klumpig (Produktionsperioden
  ~1.1-1.8 t, Leerlaufperioden ~0.45 t). In t4-t6 wird Überschuss gebankt und in den
  Produktionsperioden t7,t8 verbraucht -> kumulative Bankinglogik wird aktiv.

Standby-vs-Aus-Schwelle: Aus lohnt (emissionsseitig) erst ab Lückenlänge
  L > e_an / (e_sb - e_aus) = 1.8 / 0.45 = 4. Im Basisfall (Lücke L=3) gewinnt daher
  STANDBY (n_aus = 0); AUS zeigt sich erst ab L > 4 bzw. bei kleinerem g.

WICHTIG (Diagnose-Befund, L=3): Der Leerlauf-Zustand ist EMISSIONSdeterminiert - das
  Modell wählt stets den Emissionsboden (bei L=3 Standby, da Abschalten mit 1.8 t MEHR
  emittiert als 3*0.45=1.35 t Standby). Cap-Strenge (alpha) und Preis (pi) bewegen die
  Emission daher NICHT; sie ändern nur Zukauf/Verkauf/Kosten. Empirisch bestätigt für
  L=3/L=4/L=5 (bei L=5 kippt der Zustand Standby->Aus unter Druck, doch E bleibt gleich).
  Siehe diagnose_sensitivitaet.run_alpha_pi().
"""

from __future__ import annotations

from dataclasses import replace

from instanz import Instanz


# --- Quellen-/Kalibrierungskonstanten --------------------------------------------
TBO = 3                 # [H]  Time-Between-Orders
E_BASELINE_EST = 7.74   # GEMESSENE Baseline-Emission (t CO2e) der L=3-Basis am Vollmodell-
                        # Optimum (nicht mehr geschätzt); variante_alpha skaliert relativ dazu.
ALPHA_BASIS = 0.8       # Cap-Strenge im Basisfall
SPREAD_BASIS = 0.07     # [EUA] Geld-Brief-Spread pi^B -> pi^S
PI_B_BASIS = 80.0       # [EUA] EUR/t

# Lageremission je Produkt (produktspezifisch: Volumen/Masse) [RH] / eigene Skalierung
_E_L = {1: 0.03, 2: 0.04, 3: 0.02, 4: 0.03, 5: 0.03, 6: 0.03}


# --- Hilfsfunktionen -------------------------------------------------------------
def _demand_eintraege(K: int) -> dict:
    """Bedarfsspitzen (nur Nicht-Null-Einträge); 3-Perioden-Lücke t4-t6. K>4 füllt die
    Lücke von vorne (dichtere Sequenzstruktur).

    P3-Spitze (70 > Kapazität 60) liegt in t3 -> erzwingt weiterhin 10 Einh.
    Vorproduktion (Lager aktiv); hintere Bedarfe auf t7/t8 (nach der Lücke).
    """
    d = {(1, 1): 50.0, (2, 2): 45.0, (3, 3): 70.0, (4, 7): 50.0, (1, 8): 40.0}
    if K >= 5:
        d[(5, 4)] = 40.0      # füllt t4 (vorderste Lückenperiode)
    if K >= 6:
        d[(6, 5)] = 40.0      # füllt t5
    return d


def _voll_d(K: int, T: int, eintraege: dict) -> dict:
    """Vollständiges Bedarfs-Dict (alle (k,t) = 0, dann Spitzen setzen)."""
    d = {(k, t): 0.0 for k in range(1, K + 1) for t in range(1, T + 1)}
    d.update(eintraege)
    return d


def _tr_matrix(K: int) -> dict:
    """[H] Sequenzabhängige Rüstzeit, zyklisch-asymmetrisch, Dreiecksungleichung.

    tr_{ik} = 2 + ((k - i) mod K): natürliche Reihenfolge 1->2->...->K hat minimale
    Rüstzeit 3; Rücksprünge sind teurer (bis K+1). Werte 3..(K+1); jeder direkte
    Wechsel <= K+1, jeder 2er-Pfad >= 6 -> Dreiecksungleichung für K<=5 erfüllt.
    """
    return {
        (i, k): float(2 + ((k - i) % K))
        for i in range(1, K + 1)
        for k in range(1, K + 1)
        if i != k
    }


def _ruestkosten(d_voll: dict, K: int, T: int, tbo: int = TBO) -> dict:
    """[H] Rüstkosten je Zielprodukt über TBO: s_k = h * d_quer_k * tbo^2 / 2, h=1."""
    s = {}
    for k in range(1, K + 1):
        d_quer = sum(d_voll[(k, t)] for t in range(1, T + 1)) / T
        s[k] = float(round(1.0 * d_quer * tbo ** 2 / 2))
    return s


def _e_fix(K: int, wert: float = 0.1) -> dict:
    """[RH] Fixe Rüstemission je Wechsel (warm-up-Stoß); hier uniform."""
    return {
        (i, k): wert
        for i in range(1, K + 1)
        for k in range(1, K + 1)
        if i != k
    }


def _A_konstant(alpha: float, T: int, e_base: float = E_BASELINE_EST) -> dict:
    """Glatte (konstante) Gratiszuteilung mit Summe = alpha * E_baseline."""
    a_t = alpha * e_base / T
    return {t: a_t for t in range(1, T + 1)}


# --- Basisinstanz ----------------------------------------------------------------
def basis(K: int = 4, T: int = 8) -> Instanz:
    """Literaturgestützte Basisinstanz des PLSP-SD-E (K=4, T=8 als Standard).

    Für K in {5, 6} werden Produkte in die Leerlaufperioden gelegt (dichtere
    Sequenzstruktur). ACHTUNG: K>4 verändert E_baseline -> A_t sollte nach einem
    Baseline-Lauf neu kalibriert werden (siehe Caveats im Begleitdokument).
    """
    eintr = _demand_eintraege(K)
    d = _voll_d(K, T, eintr)
    e_sb = 0.45                                   # [BBG] Standby-Grundlast (Basis)
    return Instanz(
        K=K,
        T=T,
        i_0=1,                                    # eigene Setzung: Startrüstung P1
        d=d,
        h={k: 1.0 for k in range(1, K + 1)},      # [H] normiert
        s=_ruestkosten(d, K, T),                  # [H] via TBO
        tb={k: 1.0 for k in range(1, K + 1)},     # eigene Setzung: Einheits-Produktionszeit
        tr=_tr_matrix(K),                         # [H] sequenzabhängig
        b={t: 60.0 for t in range(1, T + 1)},     # kalibriert auf rho ~ 0.56
        tau_an=2.0,                               # eigene Setzung: Anlaufzeit (Kapazität)
        e_p={k: 0.01 for k in range(1, K + 1)},   # [RH] linear; skaliert (Kommensurabilität)
        e_fix=_e_fix(K),                          # [RH] fixe Rüstemission
        e_var=0.02,                               # [RH] zeitproportionale Rüstemission
        e_l={k: _E_L[k] for k in range(1, K + 1)},# [RH] produktspezifisch (Volumen/Masse)
        e_on=0.60,                                # [BBG] An ~ 1.3 x Standby
        e_sb=e_sb,                                # [BBG]
        e_aus=0.0,                                # [BBG] Aus = 0
        e_an=4.0 * e_sb,                          # [BBG] Anlauf = 4 x Standby (g=4) -> 1.8
        A=_A_konstant(ALPHA_BASIS, T),            # kalibriert: alpha=0.8
        pi_B=PI_B_BASIS,                          # [EUA]
        pi_S=PI_B_BASIS * (1.0 - SPREAD_BASIS),   # [EUA] 80 * 0.93 = 74.4
        J_0=0.0,                                  # [RH]/Absi: Anfangssaldo 0
        y_0={k: 0.0 for k in range(1, K + 1)},    # eigene Setzung
    )


# --- Sensitivitätsvarianten (jeweils EINE Achse gegenüber der Basis) -------------
def variante_alpha(alpha: float, K: int = 4, T: int = 8) -> Instanz:
    """(a) Cap-Strenge. alpha in {0.7, 0.8, 0.9}. Kleiner = knapper = mehr Zukauf."""
    return replace(basis(K, T), A=_A_konstant(alpha, T))


def variante_preise(pi_B: float, spread: float = SPREAD_BASIS, K: int = 4, T: int = 8) -> Instanz:
    """(c) Zertifikatspreise. pi_B im Band 63..94 EUR; pi_S = pi_B*(1-spread)."""
    return replace(basis(K, T), pi_B=pi_B, pi_S=pi_B * (1.0 - spread))


def variante_g(g: float, K: int = 4, T: int = 8) -> Instanz:
    """(d) Anlauffaktor. e_an = g * e_sb. g in {2,3,4}. Schwelle Standby->Aus: L>g.

    Bei g=2 gewinnt im Basisfall (Lücke L=2) bereits AUS -> demonstriert den Umschlag.
    """
    inst = basis(K, T)
    return replace(inst, e_an=g * inst.e_sb)


def variante_rho(b_wert: float, K: int = 4, T: int = 8) -> Instanz:
    """(b) Kapazität / Auslastung. b in {50,60,70} -> rho ~ {0.67,0.56,0.48}."""
    return replace(basis(K, T), b={t: b_wert for t in range(1, T + 1)})


def variante_lange_luecke(T: int = 8) -> Instanz:
    """(b) Lange Leerlauflücke: 4 Produkte, Lücke t4..t7 (L=4), Wiederanlauf in t8.

    Bedarf t1:P1=60, t2:P2=55, t3:P3=50, t8:P4=45. Bei L=4 liegt das Modell mit g=4
    an der Standby/Aus-Schwelle; in Kombination mit variante_g(3) wird AUS strikt
    optimal (Anlauf + Wiederanlauf demonstrierbar, inkl. tau_an in der Kapazität).
    ACHTUNG: geringere Gesamtemission -> A_t nach Baseline-Lauf neu kalibrieren.
    """
    eintr = {(1, 1): 60.0, (2, 2): 55.0, (3, 3): 50.0, (4, 8): 45.0}
    d = _voll_d(4, T, eintr)
    return replace(basis(4, T), d=d, s=_ruestkosten(d, 4, T))


def variante_produktzahl(K: int, T: int = 8) -> Instanz:
    """(e) Produktzahl. K in {4,5,6}. K>4 füllt Leerlaufperioden (dichtere Sequenz).

    ACHTUNG: ändert E_baseline; A_t nach Baseline-Lauf neu kalibrieren.
    """
    return basis(K, T)


# --- Diagnose-Erweiterung: freies Lager + gestufte Ruestemission ----------------
def basis_freielager(K: int = 4, T: int = 8, b_wert: float = 80.0) -> Instanz:
    """Basis OHNE kapazitaetserzwungene Vorproduktion (AUFGABE 1).

    Ursache der erzwungenen Vorproduktion in ``basis()`` ist die Bedarfsspitze
    d_{3,3}=70 > Kapazitaet b_t=60 (10 Einh. P3 muessen in t2 vorgezogen werden).
    Hier wird b_t so angehoben (Default 80), dass KEINE Periodennachfrage die
    Kapazitaet uebersteigt (groesste Spitze 70 < 80) -> Vorproduktion wird OPTIONAL
    statt erzwungen. Bedarfsstruktur, Luecke t4-t6 und alle uebrigen Parameter
    bleiben unveraendert. (Diagnostische Datensatz-Variante; Modell unveraendert.)
    """
    return replace(basis(K, T), b={t: b_wert for t in range(1, T + 1)})


def variante_ruestemission(m: float, basis_inst: Instanz) -> Instanz:
    """Skaliert NUR die Ruestemissionen e_fix und e_var mit dem Faktor m (AUFGABE 2).

    Ruestkosten s_k und Ruestzeiten tr bleiben UNVERAENDERT - es geht rein um die
    physikalische Emission des Sortenwechsels (warm-up). m=1 reproduziert die
    uebergebene Basis.

        e_fix_neu = m * e_fix,   e_var_neu = m * e_var.
    """
    return replace(
        basis_inst,
        e_fix={ik: m * v for ik, v in basis_inst.e_fix.items()},
        e_var=m * basis_inst.e_var,
    )


# --- Standalone-Übersicht (optional, zur Kontrolle vor dem Lösen) ----------------
if __name__ == "__main__":
    inst = basis()
    print("BASISINSTANZ PLSP-SD-E")
    print(f"  K={inst.K}, T={inst.T}, i_0={inst.i_0}, tau_an={inst.tau_an}")
    print(f"  Rüstkosten s_k (via TBO={TBO}): {inst.s}")
    print(f"  Lagerkosten h_k: {inst.h}")
    print(f"  Kapazität b_t: {inst.b[1]} (konstant)")
    print(f"  e_p={inst.e_p[1]}, e_on={inst.e_on}, e_sb={inst.e_sb}, "
          f"e_aus={inst.e_aus}, e_an={inst.e_an}")
    print(f"  e_fix={inst.e_fix[(1, 2)]} (uniform), e_var={inst.e_var}, e_l={inst.e_l}")
    print(f"  A_t (alpha={ALPHA_BASIS}): {inst.A[1]} (konstant), J_0={inst.J_0}")
    print(f"  pi_B={inst.pi_B}, pi_S={round(inst.pi_S, 2)}")
    print("\n  Beziehungen: 0 <= e_aus <= e_sb <= e_on :",
          0 <= inst.e_aus <= inst.e_sb <= inst.e_on,
          "| pi_B >= pi_S :", inst.pi_B >= inst.pi_S)
    d_spitzen = {kt: v for kt, v in inst.d.items() if v > 0}
    print(f"  Bedarfsspitzen: {d_spitzen}")
    print(f"  Summe Bedarf: {sum(inst.d.values())}")
