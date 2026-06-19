"""Konkrete Probleminstanzen (Daten) fuer Tests und Validierung.

Strikte Trennung: Hier liegen ausschliesslich Zahlenwerte (Instanzdaten), keine
Modelllogik. Jede Factory-Funktion liefert eine fertige :class:`instanz.Instanz`.

Enthalten sind:
- ``basis_instanz_k2_t4``  : kleine, gut konditionierte Instanz (K=2, T=4) fuer den
                             Bau-/Loesungstest und den Reduktionstest.
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
