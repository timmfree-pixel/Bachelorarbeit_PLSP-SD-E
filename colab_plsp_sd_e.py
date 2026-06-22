# =============================================================================
#  PLSP-SD-E  --  Proportional Lot-Sizing & Scheduling mit sequenzabhaengigen
#  Ruestzeiten und Emissionshandel.  KONSOLIDIERTE EINZELDATEI FUER GOOGLE COLAB.
# =============================================================================
#  Komplett in EINE Colab-Zelle einfuegen und ausfuehren. Reihenfolge:
#    1) Instanz-Datenklasse  2) konkrete Instanzen  3) Basismodell  4) Vollmodell
#    5) Auswertung (report_loesung / solve_und_report)  6) Beispielblock.
#
#  -------------------------------------------------------------------------
#  INSTALLATION & SETUP  (in Colab: die !pip-Zeile wird automatisch ausgefuehrt)
#  -------------------------------------------------------------------------
#  LIZENZHINWEIS: Die per pip installierte Gurobi-Lizenz ist GROESSENBESCHRAENKT
#  (ausreichend fuer kleine Instanzen wie die Toy-Instanz unten). Fuer grosse
#  Instanzen ist eine WLS-Lizenz (Web License Service) noetig:
#  https://www.gurobi.com/features/web-license-service/
# =============================================================================
!pip install gurobipy pandas

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

# pandas so konfigurieren, dass ALLE Spalten und Zeilen vollstaendig angezeigt werden
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# In Colab/Jupyter rendert display() DataFrames als HTML-Tabelle; ausserhalb davon
# (reines Python) faellt es auf print() zurueck, damit die Datei ueberall laeuft.
try:
    from IPython.display import display
except Exception:  # pragma: no cover
    display = print



# =============================================================================
# 1) DATENKLASSE  (aus instanz.py)
# =============================================================================
"""Datenschema für das PLSP-SD-E (Proportional Lot-Sizing and Scheduling Problem
mit Sequenzabhängigen Rüstzeiten und Emissionshandel).

Dieses Modul enthält AUSSCHLIESSLICH die Datenstruktur (Schema) einer Probleminstanz.
Es enthält bewusst KEINE Zahlenwerte und KEINE Modelllogik (strikte Trennung von
Modell, Daten und Auswertung). Konkrete Instanzen werden in ``instanzen.py`` erzeugt,
die Modelllogik liegt in ``modell.py``.

OR-Terminologie der Indizes/Parameter (deutsch):
- Mikroperiode      : feinste Zeiteinheit t, in der höchstens ein Rüstwechsel und die
                      Produktion eines (gerüsteten) Produkts stattfindet.
- Rüstzustand       : omega_{kt} = 1, falls die Anlage in t auf Produkt k gerüstet ist.
- Rüstwechsel       : chi_{ikt} = 1, falls in t von Vorgängerprodukt i auf k gewechselt
                      wird (sequenzabhängige Rüstzeit tr_{ik}).
- Leistungszustand  : An / Standby / Aus (z^an, z^sb, z^aus) der Anlage je Mikroperiode.
- Anschaltvorgang   : u_t erkennt den Übergang Aus -> Ein (Standby/An) und verursacht
                      den Anschalt-Emissionsstoß e^an.
- Emissionszertifikat: J_t Saldo, A_t Gratiszuteilung, B_t Zukauf, S_t Verkauf.
- Anfangsbedingung  : i_0 (Startprodukt), y_{k,0} (Anfangslager), J_0 (Anfangssaldo).
"""




@dataclass
class Instanz:
    """Vollständige Dateninstanz des PLSP-SD-E.

    Die Indexkonvention ist 1-basiert, um die mathematische Notation exakt zu
    spiegeln: Produkte k, i in {1, ..., K}, Mikroperioden t in {1, ..., T}.

    Mengen / Indizes
    ----------------
    K   : Anzahl Produkte (Index k bzw. i laeuft ueber 1..K).
    T   : Anzahl Mikroperioden (Index t laeuft ueber 1..T).
    i_0 : Startprodukt (Produktindex in 1..K), auf das die Anlage zu Beginn
          (Periode 0) gerüstet ist; Anfangsbedingung (8).

    Basisparameter (PLSP-SD)
    ------------------------
    d   : d_{kt}  Bedarf von Produkt k in Periode t            (Dict[(k, t)] -> float)
    h   : h_k     Lagerkostensatz je Einheit Endbestand        (Dict[k] -> float)
    s   : s_k     Rüstkosten je Wechsel IN das Zielprodukt k    (Dict[k] -> float)
    tb  : tb_k    Produktionszeit (Belegung) je Einheit von k   (Dict[k] -> float, > 0)
    tr  : tr_{ik} Sequenzabhängige Rüstzeit Wechsel i -> k       (Dict[(i, k)] -> float)
    b   : b_t     Verfügbare Kapazität (Zeit) der Mikroperiode t (Dict[t] -> float)
    tau_an: tau^{an} Anlaufzeit (produktunabhängig) je Anschaltvorgang u_t. Belegt
                  Kapazität in (3): Verbrauchsseite + tau^{an} u_t. Default 0.0
                  (rückwärtskompatibel: ohne gesetzte Anlaufzeit bleibt (3)
                  unverändert; im Reduktionsmodus ist u_t = 0, der Term entfällt).

    Erweiterungsparameter (Emissionen / Zertifikatshandel)
    ------------------------------------------------------
    e_p   : e^p_k        Emission je produzierter Einheit von k        (Dict[k])
    e_fix : e^{fix}_{ik} Fixe Rüstemission beim Wechsel i -> k          (Dict[(i, k)])
    e_var : e^{var}      Variable Rüstemission je Rüstzeiteinheit       (float, Skalar)
    e_l   : e^l_k        Emission je Einheit Endbestand (Lager) von k   (Dict[k])
    e_on  : e^{on}       Emission je Periode im Leistungszustand An     (float)
    e_sb  : e^{sb}       Emission je Periode im Leistungszustand Standby(float)
    e_aus : e^{aus}      Emission je Periode im Leistungszustand Aus    (float)
    e_an  : e^{an}       Emissionsstoß je Anschaltvorgang (Aus -> Ein)  (float)
    A     : A_t          Gratiszuteilung Zertifikate in Periode t       (Dict[t])
    pi_B  : pi^B         Kaufpreis je Zertifikat                        (float)
    pi_S  : pi^S         Verkaufspreis je Zertifikat                    (float)
    J_0   : J_0          Anfangssaldo Zertifikatskonto (Anfangsbedingung)(float)
    y_0   : y_{k,0}      Anfangslagerbestand von k (Anfangsbedingung)    (Dict[k])

    Modellannahmen (vom Aufrufer einzuhalten, im Modell nicht erzwungen):
        e^{aus} <= e^{sb} <= e^{on};   pi^B >= pi^S;   alle Parameter >= 0;   tb_k > 0.
    """

    # --- Mengen / Indizes -------------------------------------------------------
    K: int
    T: int
    i_0: int

    # --- Basisparameter ---------------------------------------------------------
    d: Dict[Tuple[int, int], float]
    h: Dict[int, float]
    s: Dict[int, float]
    tb: Dict[int, float]
    tr: Dict[Tuple[int, int], float]
    b: Dict[int, float]

    # --- Erweiterungsparameter (Emissionen / Zertifikate) -----------------------
    e_p: Dict[int, float]
    e_fix: Dict[Tuple[int, int], float]
    e_var: float
    e_l: Dict[int, float]
    e_on: float
    e_sb: float
    e_aus: float
    e_an: float
    A: Dict[int, float]
    pi_B: float
    pi_S: float
    J_0: float
    y_0: Dict[int, float]

    # --- Anlaufzeit-Erweiterung (Kapazitätsverbrauch beim Anschalten) -----------
    # tau^{an}: produktunabhängige Anlaufzeit je Anschaltvorgang u_t. Wird in der
    # Kapazitätsrestriktion (3) auf der Verbrauchsseite als + tau_an * u_t addiert.
    # Default 0.0 -> rückwärtskompatibel (Term inert, (3) unverändert), per Instanz
    # aktivierbar. Muss als einziges Feld einen Default tragen (Dataclass-Regel:
    # Felder mit Default am Ende); bestehende Instanzen konstruieren per Keyword.
    tau_an: float = 0.0

    # --- abgeleitete Hilfsmengen (1-basiert) ------------------------------------
    def produkte(self) -> range:
        """Indexmenge der Produkte {1, ..., K} (fuer k bzw. i)."""
        return range(1, self.K + 1)

    def perioden(self) -> range:
        """Indexmenge der Mikroperioden {1, ..., T} (fuer t)."""
        return range(1, self.T + 1)

    def perioden_mit_null(self) -> range:
        """Indexmenge {0, 1, ..., T} inkl. Anfangsperiode 0 fuer Zustands-/Lagervariablen."""
        return range(0, self.T + 1)

    def wechsel_paare(self):
        """Geordnete Produktpaare (i, k) mit i != k (Rüstwechsel ohne Self-Loop).

        Siehe Modellannahme im Kopf von ``modell.py``: chi_{ikt} ist nur fuer i != k
        definiert ("kein i=k-Self-Loop"); ein Verbleib auf k erzeugt keinen Wechsel.
        """
        return [(i, k) for i in self.produkte() for k in self.produkte() if i != k]

    def __post_init__(self) -> None:
        """Leichte Konsistenzpruefung der Instanz (keine inhaltliche Modellannahme)."""
        if self.K < 1 or self.T < 1:
            raise ValueError("K und T muessen >= 1 sein.")
        if not (1 <= self.i_0 <= self.K):
            raise ValueError(f"Startprodukt i_0={self.i_0} liegt nicht in 1..K={self.K}.")
        for k in self.produkte():
            if self.tb.get(k, 0.0) <= 0.0:
                raise ValueError(
                    f"tb_{k} muss > 0 sein (Division in Big-M M_kt und M_t)."
                )


# =============================================================================
# 2) KONKRETE INSTANZEN  (aus instanzen.py)
# =============================================================================
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
# Symbol -> tatsaechliches Feld in der Datenklasse (nichts geraten):
#   e^p->e_p, e^on->e_on, e^sb->e_sb, e^aus->e_aus, e^an->e_an,
#   e^h (Lageremission)->e_l, c^h (Lagerkosten)->h, tau_an->tau_an,
#   C_t->b, A_t->A, pi^S->pi_S, pi^B->pi_B, Startruestzustand omega_{1,0}=1 -> i_0=1
#   (erzwungen durch Restriktion (8)); z^{aus}_0=0 ist durch (15) fest (kein Feld).
#   Setup ohne Wirkung (K=1, wechsel_paare leer): s=0, tr=0, e_fix=0, e_var=0.
#
# GEMELDET (nicht still ergaenzt): Fuer die PRODUKTIONSKOSTEN c^p existiert KEIN Feld
# - die Zielfunktion (1) kennt nur Lager-, Ruest- und Zertifikatskosten. c^p ist hier
# trade-off-neutral (Gesamtproduktion = Gesamtbedarf, in jeder zulaessigen Loesung
# gleich) und daher weggelassen. tb_k (Produktionszeit) ist vom Task nicht vorgegeben;
# konventionsgemaess wie in den uebrigen Instanzen tb_1=1.0.

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


# =============================================================================
# 3) REINES PLSP-SD-BASISMODELL  (aus plsp_sd_basis.py)
# =============================================================================
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


# =============================================================================
# 4) VOLLMODELL PLSP-SD-E  (aus modell.py)
# =============================================================================
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

    # Anlaufzeit tau^{an} (produktunabhängig): Kapazitätsverbrauch je Anschalt-
    # vorgang u_t in (3). Bewusst NICHT im Reduktionsmodus genullt (Zeit-/Kapazitäts-
    # größe, keine Emissions-/Preisgröße); im Reduktionsmodus ist u_t = 0 (z^{an}_t = 1
    # ist dort zielfunktionsneutral wählbar), sodass der Term + tau^{an} u_t automatisch
    # entfällt und der Reduktionsvergleich gegen das reine PLSP-SD exakt aufgeht.
    tau_an = daten.tau_an

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

    # (3) Kapazität der Mikroperiode (Produktion + sequenzabhängiges Rüsten + Anlaufzeit):
    #     Sum_k tb_k x_{kt} + Sum_i Sum_k tr_{ik} chi_{ikt} + tau^{an} u_t <= b_t
    #     Erweiterung um die Anlaufzeit: jeder Anschaltvorgang u_t belegt zusätzlich
    #     tau^{an} Kapazitätszeit (produktunabhängig) auf der Verbrauchsseite. Im
    #     Reduktionsmodus ist u_t = 0 (z^{an}_t = 1 wählbar), sodass der Term entfällt
    #     und (3) exakt das reine PLSP-SD reproduziert.
    modell.addConstrs(
        (gp.quicksum(daten.tb[k] * x_kt[k, t] for k in produkte)
         + gp.quicksum(daten.tr[i, k] * chi_ikt[i, k, t] for (i, k) in wechsel)
         + tau_an * u_t[t]
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


# =============================================================================
# 5) AUSWERTUNG: report_loesung  +  solve_und_report  (Loesungs-/Auswertungsschicht)
# =============================================================================
# Diese Schicht ist strikt von Modell und Daten getrennt: sie liest ein bereits
# optimal geloestes Modell aus und erzeugt Tabellen/Kennzahlen/Checks.


def report_loesung(model, daten, vars=None):
    """Vollstaendige Auswertung einer optimalen PLSP-SD-E-Loesung.

    Erzeugt nacheinander:
      (A) eine Periodentabelle (pandas.DataFrame, eine Zeile je Mikroperiode t),
      (B) eine Kennzahlen-Uebersicht (Zielwert + Emissionsaufschluesselung je Quelle),
      (C) automatische Konsistenz-Checks (7 Stueck) mit OK/FEHLER je Periode.

    Parameter
    ---------
    model : gp.Model
        Bereits optimiertes Vollmodell (Status OPTIMAL).
    daten : Instanz
        Zugehoerige Dateninstanz (Parameter/Beschriftung).
    vars : optional
        Wird aus Signaturgruenden akzeptiert, aber NICHT benoetigt: ``build_model``
        gibt ausschliesslich das ``gp.Model`` zurueck (keinen separaten vars-Container).
        Die Variablen werden daher per Name aus dem Modell gelesen
        (Projektkonvention, vgl. ``loesung.py`` / ``visualisierung.py``).

    Rueckgabe
    ---------
    pandas.DataFrame | None
        Die Periodentabelle (A); ``None``, falls das Modell nicht optimal ist.
    """
    import pandas as pd
    from gurobipy import GRB

    if model.Status != GRB.OPTIMAL:
        print(f"Modell ist nicht optimal geloest (Status={model.Status}); kein Report.")
        return None

    # --- Variablenzugriff per Name (build_model liefert nur das gp.Model) --------
    def val(name):
        v = model.getVarByName(name)
        return v.X if v is not None else 0.0

    def b01(name):
        """Binaerwert (0/1) robust gerundet (raeumt numerisches Rauschen auf)."""
        return int(round(val(name)))

    produkte = list(daten.produkte())
    perioden = list(daten.perioden())
    wechsel = daten.wechsel_paare()
    eps = 1e-6

    # =========================================================================
    # (A) Periodentabelle als DataFrame
    # =========================================================================
    zeilen = []
    for t in perioden:
        zeile = {
            "t": t,
            "z_an": b01(f"z_an[{t}]"),
            "z_sb": b01(f"z_sb[{t}]"),
            "z_aus": b01(f"z_aus[{t}]"),
            "u": b01(f"u[{t}]"),
        }
        # je Produkt: Produktionsmenge x_k (gerundet) und Ruestzustand w_k = omega (0/1)
        for k in produkte:
            zeile[f"x_{k}"] = round(val(f"x[{k},{t}]"), 3)
        for k in produkte:
            zeile[f"w_{k}"] = b01(f"omega[{k},{t}]")

        # chi: aktive Ruestwechsel der Periode als "i->k" (PLSP: hoechstens einer je t)
        aktive = [f"{i}->{k}" for (i, k) in wechsel if val(f"chi[{i},{k},{t}]") > 0.5]
        zeile["chi"] = ", ".join(aktive)

        # Kapazitaet: genutzt INKL. Anlaufzeit-Term tau_an * u_t  vs.  verfuegbar b_t (C_t)
        kap_genutzt = (
            sum(daten.tb[k] * val(f"x[{k},{t}]") for k in produkte)
            + sum(daten.tr[i, k] * val(f"chi[{i},{k},{t}]") for (i, k) in wechsel)
            + daten.tau_an * val(f"u[{t}]")
        )
        zeile["kap_genutzt"] = round(kap_genutzt, 3)
        zeile["kap_verfuegbar"] = round(daten.b[t], 3)

        # Emission, Zertifikatskonto, Gratiszuteilung, Handel
        zeile["E_t"] = round(val(f"E[{t}]"), 3)
        zeile["J_t"] = round(val(f"J[{t}]"), 3)
        zeile["A_t"] = round(daten.A[t], 3)
        zeile["B_t"] = round(val(f"B[{t}]"), 3)
        zeile["S_t"] = round(val(f"S[{t}]"), 3)
        zeilen.append(zeile)

    spalten = (
        ["t", "z_an", "z_sb", "z_aus", "u"]
        + [f"x_{k}" for k in produkte]
        + [f"w_{k}" for k in produkte]
        + ["chi", "kap_genutzt", "kap_verfuegbar", "E_t", "J_t", "A_t", "B_t", "S_t"]
    )
    df = pd.DataFrame(zeilen)[spalten]

    print("=" * 78)
    print("(A) PERIODENTABELLE  (x_k = Produktion, w_k = Ruestzustand omega, chi = i->k)")
    print("=" * 78)
    display(df)

    # =========================================================================
    # (B) Kennzahlen-Uebersicht: Zielwert + Emissionsaufschluesselung nach Quelle
    # =========================================================================
    emi = {
        "Produktion (e^p * x)":
            sum(daten.e_p[k] * val(f"x[{k},{t}]") for k in produkte for t in perioden),
        "Ruesten fix (e^fix * chi)":
            sum(daten.e_fix[i, k] * val(f"chi[{i},{k},{t}]")
                for (i, k) in wechsel for t in perioden),
        "Ruesten variabel (e^var * tr * chi)":
            sum(daten.e_var * daten.tr[i, k] * val(f"chi[{i},{k},{t}]")
                for (i, k) in wechsel for t in perioden),
        "Lager (e^l * y)":
            sum(daten.e_l[k] * val(f"y[{k},{t}]") for k in produkte for t in perioden),
        "Anlauf (e^an * u)":
            sum(daten.e_an * val(f"u[{t}]") for t in perioden),
        "Standby (e^sb * z^sb)":
            sum(daten.e_sb * val(f"z_sb[{t}]") for t in perioden),
        "Laufender Betrieb (e^on * z^an)":
            sum(daten.e_on * val(f"z_an[{t}]") for t in perioden),
        "Aus (e^aus * z^aus)":
            sum(daten.e_aus * val(f"z_aus[{t}]") for t in perioden),
    }
    summe_quellen = sum(emi.values())
    summe_E = sum(val(f"E[{t}]") for t in perioden)

    print()
    print("=" * 78)
    print("(B) KENNZAHLEN-UEBERSICHT")
    print("=" * 78)
    print(f"Zielfunktionswert  Z = {model.ObjVal:.4f}")
    print(f"Gesamtemission Sum_t E_t = {summe_E:.4f}   "
          f"(Summe der Quellen = {summe_quellen:.4f})")
    print()
    em_zeilen = [{"Emissionsquelle": q, "Emission (Sum_t)": round(v, 4)}
                 for q, v in emi.items()]
    em_zeilen.append({"Emissionsquelle": "SUMME (= Sum_t E_t)",
                      "Emission (Sum_t)": round(summe_quellen, 4)})
    display(pd.DataFrame(em_zeilen))
    if abs(summe_quellen - summe_E) > 1e-4:
        print(f"  HINWEIS: Quellensumme {summe_quellen:.4f} weicht von Sum_t E_t "
              f"{summe_E:.4f} ab (numerische Pruefung).")

    # =========================================================================
    # (C) Automatische Konsistenz-Checks
    # =========================================================================
    print()
    print("=" * 78)
    print("(C) KONSISTENZ-CHECKS")
    print("=" * 78)

    bestanden = 0
    perioden0 = [0] + perioden  # fuer (t-1)-Bezuege ist t=0 fixiert vorhanden

    def melde(nr, beschreibung, betroffen, fmt=lambda x: f"t={x}"):
        nonlocal bestanden
        ok = (len(betroffen) == 0)
        if ok:
            bestanden += 1
            print(f"  [OK]     Check {nr}: {beschreibung}")
        else:
            detail = ", ".join(fmt(x) for x in betroffen)
            print(f"  [FEHLER] Check {nr}: {beschreibung}")
            print(f"           -> betroffene Perioden: {detail}")
        return ok

    # 1. Eindeutiger Leistungszustand: z_an + z_sb + z_aus == 1
    bad = [t for t in perioden
           if abs(val(f"z_an[{t}]") + val(f"z_sb[{t}]") + val(f"z_aus[{t}]") - 1.0) > eps]
    melde(1, "z_an + z_sb + z_aus == 1 fuer alle t", bad)

    # 2. (9) Produktion nur im An-Zustand: x_{k,t} > 0  ==>  z_an_t == 1
    bad = [t for t in perioden
           if sum(val(f"x[{k},{t}]") for k in produkte) > eps and val(f"z_an[{t}]") < 1 - eps]
    melde(2, "x_{k,t} > 0  ==>  z_an_t == 1 fuer alle k,t", bad)

    # 3. (11) Anlaufdetektion: u_t >= z_aus_{t-1} - z_aus_t
    bad = [t for t in perioden
           if val(f"u[{t}]") < val(f"z_aus[{t-1}]") - val(f"z_aus[{t}]") - eps]
    melde(3, "u_t >= z_aus_{t-1} - z_aus_t fuer alle t", bad)

    # 4. Ruestzustand aendert sich nur durch einen Ruestwechsel chi (nie durch z_aus)
    bad = []
    for t in perioden:
        geaendert = any(abs(val(f"omega[{k},{t}]") - val(f"omega[{k},{t-1}]")) > eps
                        for k in produkte)
        chi_aktiv = sum(val(f"chi[{i},{k},{t}]") for (i, k) in wechsel) > 0.5
        if geaendert and not chi_aktiv:
            bad.append(t)
    melde(4, "omega aendert sich nur durch einen Ruestwechsel chi (nie durch z_aus)", bad)

    # 5. Kein Self-Loop: chi_{k,k,t} == 0  (Variable ist nur fuer i != k definiert, A3)
    bad = []
    for k in produkte:
        for t in perioden:
            v = model.getVarByName(f"chi[{k},{k},{t}]")
            if v is not None and abs(v.X) > eps:
                bad.append((k, t))
    melde(5, "kein Self-Loop chi_{k,k,t} == 0 fuer alle k,t", bad,
          fmt=lambda kt: f"(k={kt[0]}, t={kt[1]})")

    # 6. (17) Zertifikatebilanz: J_t == J_{t-1} + A_t + B_t - S_t - E_t
    bad = [t for t in perioden
           if abs(val(f"J[{t}]")
                  - (val(f"J[{t-1}]") + daten.A[t] + val(f"B[{t}]")
                     - val(f"S[{t}]") - val(f"E[{t}]"))) > 1e-4]
    melde(6, "J_t == J_{t-1} + A_t + B_t - S_t - E_t fuer alle t", bad)

    # 7. Nichtnegativer Zertifikatssaldo: J_t >= 0
    bad = [t for t in perioden if val(f"J[{t}]") < -eps]
    melde(7, "J_t >= 0 fuer alle t", bad)

    print("-" * 78)
    print(f"{bestanden} von 7 Checks bestanden.")
    print("=" * 78)

    return df


def solve_und_report(daten, reduktion_basismodell=False, ausgabe=False):
    """Komfort-Wrapper: baut, loest und reportet eine Instanz in einem Schritt.

    Prueft den Solver-Status und gibt bei Infeasibility eine verstaendliche Meldung
    aus. Rueckgabe: (modell, periodentabelle_df_oder_None).
    """
    modell = build_model(daten, reduktion_basismodell=reduktion_basismodell)
    modell.Params.OutputFlag = 1 if ausgabe else 0
    modell.optimize()

    if modell.Status == GRB.OPTIMAL:
        df = report_loesung(modell, daten)
        return modell, df
    if modell.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        print("Das Modell ist INFEASIBLE (keine zulaessige Loesung). "
              "Pruefe Kapazitaeten b_t (inkl. Anlaufzeit tau_an), Bedarfe d_{kt} und "
              "Anfangsbedingungen. (Status INF_OR_UNBD laesst sich mit "
              "modell.Params.DualReductions = 0 eindeutig als INFEASIBLE/UNBOUNDED klaeren.)")
    else:
        print(f"Kein optimaler Status (Status={modell.Status}). "
              "Erwartet wird GRB.OPTIMAL (=2).")
    return modell, None


def demo_test_standby_aus():
    """Loest beide Trade-off-Instanzen nacheinander und reportet jede vollstaendig.

    Weist den Standby/Aus-Trade-off nach: eine einzelne Leerlaufperiode wird per
    Standby ueberbrueckt (test_standby), zwei aufeinanderfolgende per Abschalten mit
    Wiederanlauf (test_aus). Fuer jede Instanz: bauen, loesen, Status pruefen und
    report_loesung (Periodentabelle, Kennzahlen, 7 Checks). Modell bleibt unveraendert.
    """
    faelle = [
        ("test_standby", test_standby,
         "EIN Leerlauf   -> Erwartung: STANDBY (z_sb=1 in der Leerlaufperiode)"),
        ("test_aus", test_aus,
         "ZWEI Leerlaeufe -> Erwartung: AUS (z_aus=1) mit Wiederanlauf u=1"),
    ]
    for name, factory, erwartung in faelle:
        print("#" * 78)
        print(f"# INSTANZ {name}:  {erwartung}")
        print("#" * 78)
        daten = factory()
        modell = build_model(daten)          # bestehendes Modell UNVERAENDERT genutzt
        modell.Params.OutputFlag = 0
        modell.optimize()
        if modell.Status == GRB.OPTIMAL:
            report_loesung(modell, daten)
        elif modell.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            print(f"{name}: INFEASIBLE - keine zulaessige Loesung.")
        else:
            print(f"{name}: kein optimaler Status (Status={modell.Status}).")
        print()


# =============================================================================
# REDUKTIONSTEST: erweitertes Modell (reduktion_basismodell=True) == Basismodell
# (AUFGABE 1: Schalter pruefen | 2: Aequivalenztest | 3: alle Instanzen | 4: Anker)
# =============================================================================

# Registry ALLER Instanzen aus dem Instanzen-Abschnitt (fuer den Lauf ueber alles).
ALLE_INSTANZEN = [
    ("basis_instanz_k2_t4", basis_instanz_k2_t4),
    ("grosse_instanz_k6_t8", grosse_instanz_k6_t8),
    ("standby_instanz", standby_instanz),
    ("aus_instanz", aus_instanz),
    ("ruesten_im_aus_instanz", ruesten_im_aus_instanz),
    ("test_standby", test_standby),
    ("test_aus", test_aus),
    ("ruesten_im_aus_forced", ruesten_im_aus_forced),
]


def _mval(model, name):
    """Variablenwert per Name (0.0, falls Variable nicht existiert)."""
    v = model.getVarByName(name)
    return v.X if v is not None else 0.0


def _kostenanteile(model, daten):
    """Ruest- und Lagerkosten aus den Modellvariablen (gleiche Namen in beiden Modellen).

    Ruestkosten = Sum s_k * chi_{ikt};  Lagerkosten = Sum h_k * y_{kt} (t=1..T).
    h_k und s_k werden vom Reduktionsschalter NICHT genullt (nur Emissionen/Preise),
    daher fuer beide Modelle korrekt vergleichbar.
    """
    ruest = sum(daten.s[k] * _mval(model, f"chi[{i},{k},{t}]")
                for (i, k) in daten.wechsel_paare() for t in daten.perioden())
    lager = sum(daten.h[k] * _mval(model, f"y[{k},{t}]")
                for k in daten.produkte() for t in daten.perioden())
    return ruest, lager


def pruefe_reduktionsschalter(instanz, name=""):
    """AUFGABE 1: prueft empirisch, was reduktion_basismodell=True bewirkt (OK/FEHLT).

    (a) alle Emissionsparameter 0 -> E_t = 0 in jeder Periode.
    (b) pi^B = pi^S = 0 -> Zertifikatsterm faellt aus der Zielfunktion (Z = Lager+Ruest),
        J_t >= 0 bindet nicht kostenwirksam.
    (c) z_an_t = 1 (z_sb=z_aus=0, u=0, tau_an*u_t = 0). HINWEIS: der Schalter FIXIERT
        z_an NICHT - das Resultat u_t=0 wird vom Solver als zielfunktionsneutrales
        Optimum erreicht (hier gezeigt), weshalb die Aequivalenz dennoch haelt.
    """
    m = build_model(instanz, reduktion_basismodell=True)
    m.Params.OutputFlag = 0
    m.optimize()
    perioden = list(instanz.perioden())
    eps = 1e-7

    print(f"AUFGABE 1 - Reduktionsschalter auf '{name}' (Status "
          f"{'OPTIMAL' if m.Status == GRB.OPTIMAL else m.Status}):")

    # (a) E_t = 0 in jeder Periode
    E = [_mval(m, f"E[{t}]") for t in perioden]
    a_ok = all(abs(e) < eps for e in E)
    print(f"  [{'OK   ' if a_ok else 'FEHLT'}] (a) Emissionen neutralisiert: "
          f"max_t |E_t| = {max((abs(e) for e in E), default=0.0):.2e}")

    # (b) Zertifikatsterm aus Z: Z == Lager + Ruest (kein Zertifikatsanteil)
    ruest, lager = _kostenanteile(m, instanz)
    z_ohne_cert = m.ObjVal if m.Status == GRB.OPTIMAL else float("nan")
    b_ok = (m.Status == GRB.OPTIMAL and abs(z_ohne_cert - (ruest + lager)) < 1e-6)
    print(f"  [{'OK   ' if b_ok else 'FEHLT'}] (b) Zertifikatsterm faellt aus Z: "
          f"Z={z_ohne_cert:.4f} == Lager+Ruest={ruest + lager:.4f}")

    # (c) z_an_t = 1 / u_t = 0 -> tau_an*u_t = 0 (Kapazitaet wie Basis)
    z_an = [_mval(m, f"z_an[{t}]") for t in perioden]
    u = [_mval(m, f"u[{t}]") for t in perioden]
    c_loesung_ok = all(za > 0.5 for za in z_an) and all(abs(ui) < eps for ui in u)
    print("  [FEHLT] (c) z_an_t=1 wird vom Schalter NICHT erzwungen (struktureller Befund). "
          f"\n          -> In der Loesung dennoch u_t=0 ({'ja' if c_loesung_ok else 'NEIN'}), "
          f"tau_an*u_t=0, Kapazitaet == Basis. Aequivalenz bleibt gewahrt.\n"
          "          -> Optionale Loesung (nicht noetig, Modell bleibt unveraendert): im "
          "Reduktionsmodus z_an_t.lb=1 setzen.")
    return {"a": a_ok, "b": b_ok, "c_strukturell_fixiert": False, "c_loesung_ut0": c_loesung_ok}


def reduktionstest(instanz, name=""):
    """AUFGABE 2: Aequivalenz Basismodell (plsp_sd_basis) vs. erweitertes Modell
    im Reduktionsmodus auf DERSELBEN Instanz.

    Beide Modelle erhalten identische Anfangsbedingungen, da beide daten.i_0 fuer
    omega_{i0,0}=1 nutzen (verifiziert). Bestanden, wenn beide OPTIMAL und
    |Z_basis - Z_ext| < 1e-4. Gibt Zeile + Kostenaufschluesselung aus und liefert dict.
    """
    mb = build_plsp_sd_basis(instanz)
    mb.Params.OutputFlag = 0
    mb.optimize()
    me = build_model(instanz, reduktion_basismodell=True)
    me.Params.OutputFlag = 0
    me.optimize()

    status_ok = (mb.Status == GRB.OPTIMAL and me.Status == GRB.OPTIMAL)
    zb = mb.ObjVal if mb.Status == GRB.OPTIMAL else float("nan")
    ze = me.ObjVal if me.Status == GRB.OPTIMAL else float("nan")
    diff = abs(zb - ze) if status_ok else float("nan")
    bestanden = bool(status_ok and diff < 1e-4)

    rb, lb = _kostenanteile(mb, instanz)
    re_, le = _kostenanteile(me, instanz)

    # Anfangsbedingung verifizieren: gleiche Startruestung omega_{i0,0}=1 in beiden
    i0 = instanz.i_0
    start_gleich = (abs(_mval(mb, f"omega[{i0},0]") - 1.0) < 1e-6
                    and abs(_mval(me, f"omega[{i0},0]") - 1.0) < 1e-6)

    status_txt = "BESTANDEN" if bestanden else "FEHLER"
    if not status_ok:
        status_txt += f" (Status basis={mb.Status}, ext={me.Status})"
    print(f"{name:24s} | Z_basis={zb:9.4f} | Z_ext={ze:9.4f} | "
          f"diff={diff:.2e} | {status_txt}")
    print(f"     Kosten Basis: Ruest={rb:8.4f}  Lager={lb:8.4f}    "
          f"Ext: Ruest={re_:8.4f}  Lager={le:8.4f}    "
          f"Startruestung gleich: {'ja' if start_gleich else 'NEIN'}")

    return {
        "Instanz": name, "Z_basis": zb, "Z_ext": ze, "Differenz": diff,
        "Status": "BESTANDEN" if bestanden else "FEHLER",
        "Ruest_basis": rb, "Lager_basis": lb, "Ruest_ext": re_, "Lager_ext": le,
        "Start_gleich": start_gleich, "bestanden": bestanden,
    }


def run_reduktionstests():
    """AUFGABE 1-4 in einem Lauf: Schalter pruefen, Aequivalenztest ueber ALLE
    Instanzen, Zusammenfassung 'X von N', optionaler externer Kaczmarczyk-Anker."""
    print("=" * 78)
    print("AUFGABE 1: Reduktionsschalter verifizieren (repraesentativ, tau_an>0)")
    print("=" * 78)
    pruefe_reduktionsschalter(test_aus(), "test_aus")

    print()
    print("=" * 78)
    print("AUFGABE 2+3: Aequivalenztest extended-reduktion vs. Basismodell (alle Instanzen)")
    print("=" * 78)
    ergebnisse = [reduktionstest(fac(), name) for name, fac in ALLE_INSTANZEN]

    # Konsolidierte Uebersicht als DataFrame (Colab: display, alle Spalten/Zeilen)
    df = pd.DataFrame(ergebnisse)[
        ["Instanz", "Z_basis", "Z_ext", "Differenz", "Status",
         "Ruest_basis", "Lager_basis", "Ruest_ext", "Lager_ext", "Start_gleich"]
    ]
    print()
    display(df)

    bestanden = sum(1 for e in ergebnisse if e["bestanden"])
    print(f"\n{bestanden} von {len(ergebnisse)} Instanzen bestanden.")

    print()
    print("=" * 78)
    print("AUFGABE 4: Externer Anker (Kaczmarczyk)")
    print("=" * 78)
    if any("kaczmarczyk" in name.lower() for name, _ in ALLE_INSTANZEN):
        print("Kaczmarczyk-Benchmark gefunden - bitte Z_ext gegen publizierten Wert pruefen.")
    else:
        print("Externer Benchmark (Kaczmarczyk) noch nicht hinterlegt - Test validiert "
              "aktuell nur die interne Reduktionsaequivalenz.")
    return df


# =============================================================================
# RUESTWECHSEL WAEHREND AUS-PERIODE: Nachweis (AUFGABE 1-3)
# =============================================================================

def pruefe_wechsel_in_aus(model, daten, name=""):
    """Check: gibt es eine Periode t mit z_aus_t=1, in der ein chi_{i,k,t}>0 feuert?

    Gibt Periode(n) und i->k aus. Rueckgabe True (OK), wenn mindestens eine gefunden,
    sonst False (FEHLT).
    """
    if model.Status != GRB.OPTIMAL:
        print(f"[FEHLT] {name}: Modell nicht optimal (Status={model.Status}).")
        return False
    treffer = []
    for t in daten.perioden():
        if _mval(model, f"z_aus[{t}]") > 0.5:
            for (i, k) in daten.wechsel_paare():
                if _mval(model, f"chi[{i},{k},{t}]") > 0.5:
                    treffer.append((t, f"{i}->{k}"))
    if treffer:
        liste = ", ".join(f"t={t}: {wk}" for t, wk in treffer)
        print(f"[OK]    {name}: Ruestwechsel WAEHREND Aus-Periode (z_aus=1) -> {liste}")
        return True
    print(f"[FEHLT] {name}: kein chi>0 in einer Aus-Periode (z_aus=1) gefunden.")
    return False


def demo_ruesten_im_aus():
    """AUFGABE 1-3: Nachweis, dass ein Ruestwechsel waehrend z_aus=1 zulaessig ist.

    Beide Modelle bleiben UNVERAENDERT. Grundlage: chi ist nicht an z gekoppelt
    (nur (10) bindet die Produktion x an z_an; (6)/(7) binden chi an omega), und
    tr*chi steht auf der Verbrauchsseite der Kapazitaet (3) -> ein Wechsel laesst sich
    per Kapazitaet in eine Aus-Periode zwingen.
    """
    # ---- AUFGABE 1: bestehende Instanz im Vollmodell --------------------------
    print("#" * 78)
    print("# AUFGABE 1: ruesten_im_aus_instanz im VOLLMODELL (aktive Emissionen)")
    print("#" * 78)
    d1 = ruesten_im_aus_instanz()
    m1 = build_model(d1)                  # reduktion_basismodell=False
    m1.Params.OutputFlag = 0
    m1.optimize()
    if m1.Status == GRB.OPTIMAL:
        report_loesung(m1, d1)
        pruefe_wechsel_in_aus(m1, d1, "ruesten_im_aus_instanz")
    else:
        print(f"Kein optimaler Status (Status={m1.Status}).")

    # ---- AUFGABE 2: Kapazitaetsstruktur offenlegen (nur Bericht) --------------
    print()
    print("#" * 78)
    print("# AUFGABE 2: Kapazitaetsstruktur (nur Bericht, nichts geaendert)")
    print("#" * 78)
    print("Frage: Verbraucht ein Ruestwechsel Kapazitaet (steht tr*chi auf der "
          "Verbrauchsseite von (3))?")
    print("Antwort: JA. Codestelle modell.py, Restriktion (3) 'c3_kapazitaet':")
    print("    Sum_k tb_k*x_{kt} + Sum_{i,k} tr_{ik}*chi_{ikt} + tau_an*u_t <= b_t")
    print("Der Term Sum_{i,k} tr_{ik}*chi_{ikt} steht links (Verbrauchsseite). Zudem ist")
    print("chi NICHT an z gekoppelt: (10) 'c10_an_bei_produktion' bindet nur x an z_an")
    print("('keine Kopplung von chi an z'). -> Erzwingung ueber Kapazitaet moeglich;")
    print("AUFGABE 3 wird ausgefuehrt.")

    # ---- AUFGABE 3: Erzwingungs-Instanz ---------------------------------------
    print()
    print("#" * 78)
    print("# AUFGABE 3: ruesten_im_aus_forced (Wechsel in Aus-Periode erzwungen)")
    print("#" * 78)
    d3 = ruesten_im_aus_forced()
    print("ERWARTUNG (dokumentiert VOR dem Loesen):")
    print(f"  Berechnete Kapazitaeten aus echten Parametern (tb_1={d3.tb[1]:g}, "
          f"tb_2={d3.tb[2]:g}, tr_12={d3.tr[1, 2]:g}, tau_an={d3.tau_an:g}):")
    print(f"    C_1={d3.b[1]:g} (=tb_1*5, exakt P1, kein Platz fuer tr)")
    print(f"    C_2={d3.b[2]:g}, C_3={d3.b[3]:g} (grosszuegig, Platz fuer den Wechsel)")
    print(f"    C_4={d3.b[4]:g} (=tb_2*5 + tau_an, exakt P2+Anlauf, kein Platz fuer tr)")
    print("  => Wechsel 1->2 muss in t in {2,3} liegen; diese sind im Optimum Aus.")
    print("  => Erwartet: chi_{1,2,t}=1 fuer ein t in {2,3} bei gleichzeitig z_aus_t=1.")
    print()
    m3 = build_model(d3)
    m3.Params.OutputFlag = 0
    m3.optimize()
    if m3.Status == GRB.OPTIMAL:
        report_loesung(m3, d3)
        print("ERGEBNIS vs. Erwartung:")
        pruefe_wechsel_in_aus(m3, d3, "ruesten_im_aus_forced")
    elif m3.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        print("INFEASIBLE - Kapazitaeten zu knapp gewaehlt (Erwartung verfehlt).")
    else:
        print(f"Kein optimaler Status (Status={m3.Status}).")


# =============================================================================
# 6) BEISPIELBLOCK  (in Colab als letzte Zelle ausfuehren)
# =============================================================================
# Laedt eine kleine Toy-Instanz, baut/loest das Modell, prueft den Status und ruft
# report_loesung auf. Klein genug fuer die groessenbeschraenkte pip-Gurobi-Lizenz.
if __name__ == "__main__":
    # --- 1) Toy-Instanz aus den Instanzen laden -------------------------------
    daten = basis_instanz_k2_t4()   # K=2, T=4 (klein -> pip-Lizenz genuegt)
    # Hinweis: Anlaufzeit demonstrieren? -> daten.tau_an = 1.0 setzen (Default 0.0).

    # --- 2) Modell bauen und loesen -------------------------------------------
    modell = build_model(daten)     # Vollmodell PLSP-SD-E
    modell.Params.OutputFlag = 0    # Solver-Log aus (auf 1 fuer Gurobi-Ausgabe)
    modell.optimize()

    # --- 3) Status pruefen, dann auswerten ------------------------------------
    if modell.Status == GRB.OPTIMAL:
        report_loesung(modell, daten)
    elif modell.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        print("Das Modell ist INFEASIBLE - keine zulaessige Loesung gefunden. "
              "Pruefe Kapazitaeten b_t (inkl. tau_an), Bedarfe d_{kt} und Anfangsbedingungen.")
    else:
        print(f"Kein optimaler Status (Status={modell.Status}); erwartet GRB.OPTIMAL (=2).")

    # Alternative Einzeiler (baut + loest + reportet + Status-/Infeasibility-Handling):
    #   modell, df = solve_und_report(basis_instanz_k2_t4())

    # --- 4) Standby/Aus-Trade-off-Nachweis (beide neuen Instanzen) -------------
    print()
    demo_test_standby_aus()

    # --- 5) Reduktionstest: extended(reduktion) == Basismodell (alle Instanzen) -
    print()
    run_reduktionstests()

    # --- 6) Ruestwechsel waehrend Aus-Periode: Nachweis (AUFGABE 1-3) ----------
    print()
    demo_ruesten_im_aus()
