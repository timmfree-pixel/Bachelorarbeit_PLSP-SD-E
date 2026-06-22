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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


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
