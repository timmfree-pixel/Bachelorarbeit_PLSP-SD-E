"""Konkrete Probleminstanzen (Daten) fuer Tests und Validierung.

Strikte Trennung: Hier liegen ausschliesslich Zahlenwerte (Instanzdaten), keine
Modelllogik. Jede Factory-Funktion liefert eine fertige :class:`instanz.Instanz`.

Enthalten sind:
- ``basis_instanz_k2_t4``  : kleine, gut konditionierte Instanz (K=2, T=4) fuer den
                             Bau-/Loesungstest und den Reduktionstest.
- ``grosse_instanz_k6_t8`` : grosse, realitaetsnahe Instanz (K=6, T=8) mit
                             Produktfamilien zum umfassenden Test des Vollmodells.
- ``standby_instanz``      : Verhaltenstest 3a (Standby ueberbrueckt kurzen Leerlauf).
- ``aus_instanz``          : Verhaltenstest 3b (Abschalten ueberbrueckt langen Leerlauf).
- ``ruesten_im_aus_instanz``: Verhaltenstest 3c (Ruestwechsel im Aus-Modus).

Hinweis zu den Verhaltenstests (3a-3c): Damit der Leistungszustand kostenwirksam ist,
werden die Emissionen ueber einen Netto-Verkaeufer-Regime relevant gemacht
(pi^S > 0, Gratiszuteilung A_t so gross, dass die Gesamtzuteilung die Gesamtemission
stets uebersteigt). Jede vermiedene Emissionseinheit ist dann ein Verkaufserloes.
Zusaetzlich werden die Lagerkosten h_k bewusst hoch gesetzt, damit auf-Vorrat-Produktion
keine billigere Alternative zum Leerlauf ist und der Leerlauf tatsaechlich entsteht
(geforderte "bewusst incentivierte Instanz"). Durchgehend gilt e^{aus}=0 und
e^{aus} <= e^{sb} <= e^{on}.
"""

from __future__ import annotations

from instanz import Instanz


def _voll_tr(K: int, wert_offdiag: float = 0.0) -> dict:
    """Hilfsfunktion: tr_{ik} fuer alle (i,k) inkl. Diagonale (Diagonale = 0)."""
    return {
        (i, k): (0.0 if i == k else wert_offdiag)
        for i in range(1, K + 1)
        for k in range(1, K + 1)
    }


def _voll_efix(K: int, wert_offdiag: float = 0.0) -> dict:
    """Hilfsfunktion: e^{fix}_{ik} fuer alle (i,k) inkl. Diagonale (Diagonale = 0)."""
    return {
        (i, k): (0.0 if i == k else wert_offdiag)
        for i in range(1, K + 1)
        for k in range(1, K + 1)
    }


def basis_instanz_k2_t4() -> Instanz:
    """Kleine, gut konditionierte Instanz K=2, T=4 (Bau-/Loesungs- und Reduktionstest).

    Zwei Produkte mit alternierendem Bedarf, reichlich Kapazitaet, moderate
    Emissionen und Zertifikatsparameter. Mit reduktion_basismodell=True muss der
    Zielwert mit dem reinen PLSP-SD uebereinstimmen.
    """
    K, T, i_0 = 2, 4, 1
    produkte = range(1, K + 1)
    perioden = range(1, T + 1)

    d = {(1, 1): 2, (1, 2): 0, (1, 3): 3, (1, 4): 0,
         (2, 1): 0, (2, 2): 2, (2, 3): 0, (2, 4): 2}
    h = {1: 1.0, 2: 1.0}
    s = {1: 5.0, 2: 5.0}
    tb = {1: 1.0, 2: 1.0}
    tr = _voll_tr(K, wert_offdiag=1.0)
    b = {t: 10.0 for t in perioden}

    e_p = {1: 1.0, 2: 1.0}
    e_fix = _voll_efix(K, wert_offdiag=0.5)
    e_var = 0.2
    e_l = {1: 0.1, 2: 0.1}
    e_on, e_sb, e_aus, e_an = 4.0, 2.0, 0.0, 3.0
    A = {t: 50.0 for t in perioden}
    pi_B, pi_S = 2.0, 1.0
    J_0 = 10.0
    y_0 = {k: 0.0 for k in produkte}

    return Instanz(K=K, T=T, i_0=i_0, d=d, h=h, s=s, tb=tb, tr=tr, b=b,
                   e_p=e_p, e_fix=e_fix, e_var=e_var, e_l=e_l,
                   e_on=e_on, e_sb=e_sb, e_aus=e_aus, e_an=e_an,
                   A=A, pi_B=pi_B, pi_S=pi_S, J_0=J_0, y_0=y_0)


def grosse_instanz_k6_t8() -> Instanz:
    """Grosse, realitaetsnahe Test-Instanz: 6 Produkte, 8 Mikroperioden.

    Diese Instanz dient dazu, das vollstaendige PLSP-SD-E "einmal richtig" zu testen.
    Alle sechs Produkte haben Nachfrage. Die Parameter sind oekonomisch konsistent
    gewaehlt; insbesondere besitzen die Produkte eine **Familienstruktur**, die die
    sequenzabhaengigen Ruestzeiten erst bedeutsam macht.

    Struktur und gewaehlte Verhaeltnisse
    ------------------------------------
    - Familien: A = {1,2,3}, B = {4,5,6}.
      * Ruestzeit tr_{ik}: innerhalb einer Familie guenstig (1), familienuebergreifend
        teuer und **asymmetrisch** (A->B = 3, B->A = 4) -> echte Sequenzabhaengigkeit,
        starker Anreiz zur Gruppierung nach Familie.
      * Fixe Ruestemission e^{fix}_{ik}: innerhalb 0.5, familienuebergreifend 2.0.
    - Ruestkosten s_k (8..15) >> Lagerkosten h_k (0.8..2.0) -> Losgroessen-Abwaegung
      zwischen Batching (gross produzieren + lagern) und Wiederbesuch (erneut ruesten).
    - Produktionszeit tb_k in [0.8, 1.5]; Kapazitaet b_t = 30 (moderat bindend).
    - Emissionen: Produktion dominiert (e^p_k in [0.9, 2.2]); Leistungszustaende
      e^{aus}=0 <= e^{sb}=2 <= e^{on}=5; Anschaltstoss e^{an}=3 (so ist ein 2-Perioden-
      Stillstand per Abschalten billiger als Standby: 3 < 2*2).
    - Zertifikate: pi^B=3 >= pi^S=2 (Geld-Brief-Spanne); **fallende** Gratiszuteilung
      A_t = 22,20,...,8 (ETS-Verknappung) bei J_0=10. Die Gesamtzuteilung liegt unter
      der Gesamtemission -> Netto-Kaeufer-Regime: Emissionsvermeidung lohnt sich.

    Im Optimum ausgeloeste Modellmechanismen (Validierungszweck)
    -----------------------------------------------------------
    Losgroessenbildung mit Lagerhaltung und Produkt-Wiederbesuch; familienweise
    gruppierte, sequenzabhaengige Ruestwechsel (nur 2 familienuebergreifende Wechsel);
    Leistungszustaende An und **Aus** inkl. **Anschaltvorgang** (Wiederanlauf nach
    Stillstand); Lager-Emissionen; Zertifikat-**Zukauf und -Verkauf** mit Banking auf
    dem Zertifikatskonto.

    Hinweis: Bei 6 nachgefragten Produkten und nur einem Ruestwechsel je Periode werden
    ~6 Perioden zum Einruesten benoetigt; es verbleiben hoechstens 2 Leerlaufperioden,
    die entweder einen 2-er-Aus-Block (-> Aus, wie hier) ODER zwei Einzel-Leerlaeufe
    (-> Standby) bilden koennen, nie beides zugleich. Der Standby-Zustand wird daher
    separat in ``standby_instanz`` (Verhaltenstest 3a) eindeutig demonstriert.
    """
    K, T, i_0 = 6, 8, 1
    produkte = range(1, K + 1)
    perioden = range(1, T + 1)

    def familie(k: int) -> str:
        return "A" if k <= 3 else "B"

    # Sequenzabhaengige Ruestzeiten und fixe Ruestemissionen (Familienstruktur)
    tr = {
        (i, k): (0.0 if i == k
                 else 1.0 if familie(i) == familie(k)
                 else 3.0 if familie(i) == "A"   # A -> B
                 else 4.0)                         # B -> A
        for i in produkte for k in produkte
    }
    e_fix = {
        (i, k): (0.0 if i == k
                 else 0.5 if familie(i) == familie(k)
                 else 2.0)
        for i in produkte for k in produkte
    }

    # Bedarf d_{kt} (Zeilen = Produkte, Spalten = Perioden 1..8); jedes Produkt hat Bedarf
    bedarf = {
        1: [5, 0, 4, 0, 0, 4, 0, 0],
        2: [4, 4, 0, 0, 3, 0, 0, 4],
        3: [0, 5, 4, 0, 0, 0, 4, 0],
        4: [0, 0, 6, 5, 0, 0, 4, 0],
        5: [0, 0, 0, 5, 6, 0, 0, 4],
        6: [0, 0, 0, 0, 5, 6, 0, 4],
    }
    d = {(k, t): float(bedarf[k][t - 1]) for k in produkte for t in perioden}

    h = {1: 1.0, 2: 1.5, 3: 0.8, 4: 2.0, 5: 1.2, 6: 1.0}
    s = {1: 10.0, 2: 12.0, 3: 8.0, 4: 15.0, 5: 11.0, 6: 13.0}
    tb = {1: 1.0, 2: 1.2, 3: 0.8, 4: 1.5, 5: 1.0, 6: 1.3}
    b = {t: 30.0 for t in perioden}

    e_p = {1: 1.0, 2: 1.2, 3: 0.9, 4: 2.0, 5: 1.8, 6: 2.2}
    e_l = {1: 0.05, 2: 0.08, 3: 0.05, 4: 0.10, 5: 0.08, 6: 0.10}
    e_var = 0.3
    e_on, e_sb, e_aus, e_an = 5.0, 2.0, 0.0, 3.0

    A_werte = [22, 20, 18, 16, 14, 12, 10, 8]   # fallende Gratiszuteilung (Verknappung)
    A = {t: float(A_werte[t - 1]) for t in perioden}
    pi_B, pi_S = 3.0, 2.0
    J_0 = 10.0
    y_0 = {k: 0.0 for k in produkte}

    return Instanz(K=K, T=T, i_0=i_0, d=d, h=h, s=s, tb=tb, tr=tr, b=b,
                   e_p=e_p, e_fix=e_fix, e_var=e_var, e_l=e_l,
                   e_on=e_on, e_sb=e_sb, e_aus=e_aus, e_an=e_an,
                   A=A, pi_B=pi_B, pi_S=pi_S, J_0=J_0, y_0=y_0)


def standby_instanz(D: float = 1.0) -> Instanz:
    """Verhaltenstest 3a: Standby ueberbrueckt kurzen Leerlauf (Neustart teuer).

    1 Produkt, T=3, Bedarf d=[D,0,D], reichlich Kapazitaet. e^{sb} klein (>0),
    e^{an} sehr gross. Erwartung: Leerlauf in t=2 wird als Standby ueberbrueckt
    (z^{sb}_2 = 1), da ein Abschalten den teuren Anschaltvorgang in t=3 ausloesen
    wuerde. Hohe Lagerkosten h verhindern, dass der Bedarf in t=3 stattdessen per
    Vorrat aus t=1 gedeckt wird (der Leerlauf entsteht so tatsaechlich).
    """
    K, T, i_0 = 1, 3, 1
    perioden = range(1, T + 1)

    d = {(1, 1): D, (1, 2): 0.0, (1, 3): D}
    h = {1: 1000.0}                  # Lagerung teuer -> Produktion auf Bedarf, echter Leerlauf in t=2
    s = {1: 0.0}                     # nur ein Produkt -> keine Ruestwechsel
    tb = {1: 1.0}
    tr = _voll_tr(K)                 # leer relevant (keine i!=k Paare)
    b = {t: 100.0 for t in perioden}  # reichlich Kapazitaet

    e_p = {1: 1.0}
    e_fix = _voll_efix(K)
    e_var = 0.0
    e_l = {1: 0.0}
    e_on, e_sb, e_aus, e_an = 10.0, 1.0, 0.0, 1000.0   # e_aus <= e_sb <= e_on; Neustart sehr teuer
    A = {t: 10000.0 for t in perioden}                  # Netto-Verkaeufer: Zuteilung >> Emission
    pi_B, pi_S = 2.0, 1.0                               # pi^B >= pi^S; pi^S > 0 macht Emission kostenwirksam
    J_0 = 0.0
    y_0 = {1: 0.0}

    return Instanz(K=K, T=T, i_0=i_0, d=d, h=h, s=s, tb=tb, tr=tr, b=b,
                   e_p=e_p, e_fix=e_fix, e_var=e_var, e_l=e_l,
                   e_on=e_on, e_sb=e_sb, e_aus=e_aus, e_an=e_an,
                   A=A, pi_B=pi_B, pi_S=pi_S, J_0=J_0, y_0=y_0)


def aus_instanz(D: float = 1.0) -> Instanz:
    """Verhaltenstest 3b: Abschalten ueberbrueckt langen Leerlauf (Neustart billig).

    1 Produkt, T=5, Bedarf d=[D,0,0,0,D], reichlich Kapazitaet. e^{sb} gross,
    e^{an} klein. Erwartung: Leerlauf t=2..4 wird durch Abschalten ueberbrueckt
    (z^{aus}_t = 1) mit einem Wiederanlauf zu t=5 (Sum_t u_t >= 1). Hohe Lagerkosten
    verhindern Vorratsproduktion.
    """
    K, T, i_0 = 1, 5, 1
    perioden = range(1, T + 1)

    d = {(1, 1): D, (1, 2): 0.0, (1, 3): 0.0, (1, 4): 0.0, (1, 5): D}
    h = {1: 1000.0}
    s = {1: 0.0}
    tb = {1: 1.0}
    tr = _voll_tr(K)
    b = {t: 100.0 for t in perioden}

    e_p = {1: 1.0}
    e_fix = _voll_efix(K)
    e_var = 0.0
    e_l = {1: 0.0}
    e_on, e_sb, e_aus, e_an = 100.0, 100.0, 0.0, 1.0   # e_aus <= e_sb <= e_on; Standby teuer, Neustart billig
    A = {t: 10000.0 for t in perioden}
    pi_B, pi_S = 2.0, 1.0
    J_0 = 0.0
    y_0 = {1: 0.0}

    return Instanz(K=K, T=T, i_0=i_0, d=d, h=h, s=s, tb=tb, tr=tr, b=b,
                   e_p=e_p, e_fix=e_fix, e_var=e_var, e_l=e_l,
                   e_on=e_on, e_sb=e_sb, e_aus=e_aus, e_an=e_an,
                   A=A, pi_B=pi_B, pi_S=pi_S, J_0=J_0, y_0=y_0)


def ruesten_im_aus_instanz(D: float = 1.0) -> Instanz:
    """Verhaltenstest 3c: Ruestwechsel im Aus-Modus.

    2 Produkte, T=5, Startprodukt i_0=1, Bedarf d_1=[D,0,0,0,0], d_2=[0,0,0,0,D].
    e^{aus}=0, e^{sb} gross, e^{an} klein. Die Kapazitaeten erzwingen, dass der
    Wechsel 1->2 in eine Leerlaufperiode (t in {2,3,4}) verschoben wird:
      - b_1 so knapp, dass nur die Produktion von Produkt 1 hineinpasst:
        tb_1*D <= b_1 < tb_1*D + tr_{1,2}
      - b_5 so knapp, dass nur die Produktion von Produkt 2 hineinpasst:
        tb_2*D <= b_5 < tb_2*D + tr_{1,2}
      - b_t (t=2,3,4) >= tr_{1,2}  (Wechsel im Leerlauf moeglich)
    Da e^{aus} < e^{sb}, erfolgt der Wechsel bevorzugt in einer Aus-Periode.
    Hohe Lagerkosten verhindern, dass Produkt 2 vorzeitig produziert und gelagert wird.
    Erwartung: ex. t in {2,3,4} mit chi_{1,2,t} = 1 UND z^{aus}_t = 1.
    """
    K, T, i_0 = 2, 5, 1
    produkte = range(1, K + 1)
    perioden = range(1, T + 1)

    tb = {1: 1.0, 2: 1.0}
    tr_wert = 1.0                     # tr_{1,2} = tr_{2,1} = 1
    tr = _voll_tr(K, wert_offdiag=tr_wert)

    d = {(1, 1): D, (1, 2): 0.0, (1, 3): 0.0, (1, 4): 0.0, (1, 5): 0.0,
         (2, 1): 0.0, (2, 2): 0.0, (2, 3): 0.0, (2, 4): 0.0, (2, 5): D}

    # Kapazitaeten gemaess Spezifikation 3c:
    #   tb_1*D = D <= b_1 < D + tr = D + 1   -> b_1 = D (nur Produktion 1, kein Wechsel)
    #   tb_2*D = D <= b_5 < D + tr = D + 1   -> b_5 = D (nur Produktion 2, kein Wechsel)
    #   b_2,b_3,b_4 >= tr = 1                 -> Wechsel im Leerlauf moeglich
    b = {1: tb[1] * D, 2: tr_wert, 3: tr_wert, 4: tr_wert, 5: tb[2] * D}

    h = {1: 1000.0, 2: 1000.0}        # Lagerung teuer -> Produkt 2 erst in t=5
    s = {1: 1.0, 2: 1.0}

    e_p = {1: 1.0, 2: 1.0}
    e_fix = _voll_efix(K, wert_offdiag=0.0)
    e_var = 0.0
    e_l = {1: 0.0, 2: 0.0}
    e_on, e_sb, e_aus, e_an = 100.0, 100.0, 0.0, 1.0   # e_aus=0 < e_sb; Aus billiger als Standby
    A = {t: 10000.0 for t in perioden}
    pi_B, pi_S = 2.0, 1.0
    J_0 = 0.0
    y_0 = {k: 0.0 for k in produkte}

    return Instanz(K=K, T=T, i_0=i_0, d=d, h=h, s=s, tb=tb, tr=tr, b=b,
                   e_p=e_p, e_fix=e_fix, e_var=e_var, e_l=e_l,
                   e_on=e_on, e_sb=e_sb, e_aus=e_aus, e_an=e_an,
                   A=A, pi_B=pi_B, pi_S=pi_S, J_0=J_0, y_0=y_0)


# ---------------------------------------------------------------------------
# Standby/Aus-Trade-off-Nachweis (gemeinsame Parameter, nur T und Bedarf variieren)
# ---------------------------------------------------------------------------
# Beide Instanzen unten teilen exakt dieselben Parameter und unterscheiden sich nur
# in der Anzahl aufeinanderfolgender Leerlaufperioden zwischen zwei Produktionen:
#   1 Leerlauf  -> Standby billiger (e^{sb}=2 < e^{an}=3 fuer Abschalten+Wiederanlauf)
#   2 Leerlauf  -> Abschalten billiger (2*e^{sb}=4 > e^{an}=3)
#
# Symbol -> tatsaechliches Feld in instanz.py (nichts geraten):
#   e^p->e_p, e^on->e_on, e^sb->e_sb, e^aus->e_aus, e^an->e_an,
#   e^h (Lageremission)->e_l, c^h (Lagerkosten)->h, tau_an->tau_an,
#   C_t->b, A_t->A, pi^S->pi_S, pi^B->pi_B, Startruestzustand omega_{1,0}=1 -> i_0=1
#   (erzwungen durch Restriktion (8)); z^{aus}_0=0 ist durch (15) fest (kein Feld).
#   Setup ohne Wirkung (K=1, wechsel_paare leer): s=0, tr=0, e_fix=0, e_var=0.
#
# GEMELDET (nicht still ergaenzt): Fuer die PRODUKTIONSKOSTEN c^p existiert KEIN Feld
# - die Zielfunktion (1) in modell.py kennt nur Lager-, Ruest- und Zertifikatskosten.
# c^p ist hier trade-off-neutral (Gesamtproduktion = Gesamtbedarf, in jeder
# zulaessigen Loesung gleich) und daher weggelassen. tb_k (Produktionszeit) ist vom
# Task nicht vorgegeben; konventionsgemaess wie in den uebrigen Instanzen tb_1=1.0.

def _trade_off_basis(T: int, bedarf: dict) -> Instanz:
    """Gemeinsamer Bauplan beider Trade-off-Instanzen (K=1, identische Parameter)."""
    K, i_0 = 1, 1
    produkte = range(1, K + 1)
    perioden = range(1, T + 1)

    d = {(1, t): float(bedarf[t]) for t in perioden}
    h = {1: 100.0}                     # c^h hoch -> Vorproduktion gesperrt
    s = {1: 0.0}                       # kein Ruestwechsel (K=1)
    tb = {1: 1.0}                      # Produktionszeit/Einheit (Konvention; Task: n/a)
    tr = _voll_tr(K)                   # {(1,1): 0.0}, keine i!=k Paare
    b = {t: 10.0 for t in perioden}    # C_t = 10 in jeder Periode

    e_p = {1: 1.0}
    e_fix = _voll_efix(K)              # {(1,1): 0.0}
    e_var = 0.0
    e_l = {1: 0.1}                     # e^h Lageremission je Einheit/Periode
    e_on, e_sb, e_aus, e_an = 4.0, 2.0, 0.0, 3.0
    A = {t: 50.0 for t in perioden}    # freie Allokation A_t
    pi_B, pi_S = 12.0, 10.0            # pi^B >= pi^S
    J_0 = 0.0
    y_0 = {1: 0.0}
    tau_an = 2.0                       # Anlaufzeit (Kapazitaetsverbrauch je u_t)

    return Instanz(K=K, T=T, i_0=i_0, d=d, h=h, s=s, tb=tb, tr=tr, b=b,
                   e_p=e_p, e_fix=e_fix, e_var=e_var, e_l=e_l,
                   e_on=e_on, e_sb=e_sb, e_aus=e_aus, e_an=e_an,
                   A=A, pi_B=pi_B, pi_S=pi_S, J_0=J_0, y_0=y_0, tau_an=tau_an)


def test_standby() -> Instanz:
    """Trade-off Teil 1 (K=1, T=3): EINE Leerlaufperiode -> STANDBY gewinnt.

    Bedarf d_1 = [5, 0, 5]; der einzelne Leerlauf in t=2 wird per Standby ueberbrueckt,
    da e^{sb}=2 guenstiger ist als ein Abschalten mit Wiederanlauf in t=3
    (e^{an}=3, zusaetzlich Anlaufzeit tau_an=2 auf der Kapazitaet von t=3).
    Erwartung: z^{sb}_2 = 1 (Standby), kein z^{aus}, Sum u_t = 0.
    """
    return _trade_off_basis(T=3, bedarf={1: 5, 2: 0, 3: 5})


def test_aus() -> Instanz:
    """Trade-off Teil 2 (K=1, T=4): ZWEI Leerlaufperioden -> ABSCHALTEN gewinnt.

    Bedarf d_1 = [5, 0, 0, 5]; die zwei aufeinanderfolgenden Leerlaeufe (t=2,3) werden
    per Abschalten ueberbrueckt, da 2*e^{sb}=4 teurer waere als ein einmaliger
    Wiederanlauf e^{an}=3 zu t=4. Die Kapazitaet C_t=10 traegt den Anlaufzeit-Term
    (tb_1*5 + tau_an*1 = 7 <= 10).
    Erwartung: z^{aus}_2 = z^{aus}_3 = 1 (Aus), Wiederanlauf u_4 = 1.
    """
    return _trade_off_basis(T=4, bedarf={1: 5, 2: 0, 3: 0, 4: 5})


def ruesten_im_aus_forced() -> Instanz:
    """Erzwingt einen Ruestwechsel WAEHREND einer Aus-Periode ueber die Kapazitaet.

    K=2, T=4, Startruestung omega_{1,0}=1 (i_0=1), z^{aus}_0=0 (durch (15) fest).
    Bedarf: d_1=[5,0,0,0], d_2=[0,0,0,5]. Die Maschine startet auf Produkt 1,
    produziert es in t=1 und muss bis t=4 auf Produkt 2 umgeruestet sein -> genau ein
    Ruestwechsel 1->2.

    Kapazitaeten C_t (= Feld b) aus den ECHTEN Parametern kalibriert, sodass der
    Wechsel weder in t=1 noch in t=4 passt und damit in eine Leerlaufperiode t in {2,3}
    gezwungen wird (die im Optimum Aus sind: e^{aus}=0 < e^{sb}, und zwei Leerlaeufe
    machen den einmaligen Wiederanlauf e^{an}=3 billiger als 2*e^{sb}=4):
      a_k = tb_k = 1.0 (Produktionszeit/Einheit),  tr_{12}=tr_{21}=1.0,  tau_an = 2.0
      C_1 = tb_1*5          = 5.0  (exakt Produktion P1; kein Platz fuer tr=1)
      C_4 = tb_2*5 + tau_an = 7.0  (exakt Produktion P2 + Anlaufzeit; kein Platz fuer tr=1)
      C_2 = C_3 = 100.0            (grosszuegig; Platz fuer den Wechsel)
    Der Block in t=4 traegt nur, weil der Wiederanlauf u_4=1 ist (Maschine war Aus):
    tb_2*5 + tau_an*1 = 7 = C_4, ein zusaetzliches tr=1 wuerde 8 > 7 verletzen.

    Erwartung: ex. t in {2,3} mit chi_{1,2,t} = 1 UND z^{aus}_t = 1.

    Hinweise: Setup-Emissionen (e_fix, e_var) = 0 -> der Wechsel ist emissions- und
    timing-neutral (nur die Kapazitaet erzwingt seine Lage). Fuer Produktionskosten
    c^p existiert kein Feld (timing-neutral, entfaellt).
    """
    K, T, i_0 = 2, 4, 1
    produkte = range(1, K + 1)
    perioden = range(1, T + 1)

    tb = {1: 1.0, 2: 1.0}                # a_k: Produktionszeit je Einheit
    tr = _voll_tr(K, wert_offdiag=1.0)   # tr_{12}=tr_{21}=1.0, Diagonale 0
    tau_an = 2.0                         # Anlaufzeit (Kapazitaetsverbrauch je u_t)

    d = {(1, 1): 5.0, (1, 2): 0.0, (1, 3): 0.0, (1, 4): 0.0,
         (2, 1): 0.0, (2, 2): 0.0, (2, 3): 0.0, (2, 4): 5.0}

    # Kapazitaeten aus den echten Parametern berechnet (kein Platz fuer tr in t=1,4):
    C_1 = tb[1] * 5.0                    # = 5.0
    C_4 = tb[2] * 5.0 + tau_an           # = 7.0
    C_offen = 100.0                      # t=2,3 grosszuegig
    b = {1: C_1, 2: C_offen, 3: C_offen, 4: C_4}

    h = {1: 1000.0, 2: 1000.0}           # Lagerkosten hoch -> keine Vorproduktion
    s = {1: 1.0, 2: 1.0}                 # Ruestkosten (timing-neutral)

    e_p = {1: 1.0, 2: 1.0}
    e_fix = _voll_efix(K, wert_offdiag=0.0)  # keine fixe Ruestemission
    e_var = 0.0
    e_l = {1: 0.1, 2: 0.1}               # e^h Lageremission
    e_on, e_sb, e_aus, e_an = 4.0, 2.0, 0.0, 3.0
    A = {t: 50.0 for t in perioden}      # reichlich -> Netto-Verkaeufer
    pi_B, pi_S = 12.0, 10.0              # pi^B >= pi^S > 0
    J_0 = 0.0
    y_0 = {k: 0.0 for k in produkte}

    return Instanz(K=K, T=T, i_0=i_0, d=d, h=h, s=s, tb=tb, tr=tr, b=b,
                   e_p=e_p, e_fix=e_fix, e_var=e_var, e_l=e_l,
                   e_on=e_on, e_sb=e_sb, e_aus=e_aus, e_an=e_an,
                   A=A, pi_B=pi_B, pi_S=pi_S, J_0=J_0, y_0=y_0, tau_an=tau_an)
