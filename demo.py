"""Demonstration: PLSP-SD-E loesen, auswerten und grafisch darstellen.

Ausfuehren mit::

    python3 demo.py

Erzeugt fuer mehrere Instanzen je eine PNG-Uebersicht (5 Panels) im aktuellen
Verzeichnis und gibt zusaetzlich die textuelle Loesungstabelle aus.
"""

from __future__ import annotations

from instanzen import (
    basis_instanz_k2_t4,
    standby_instanz,
    aus_instanz,
    ruesten_im_aus_instanz,
)
from modell import build_model
from loesung import loese, zeige_loesung
from visualisierung import visualisiere


# Instanzname -> (Factory, Anzeigetitel, Ausgabedatei)
DEMOS = {
    "basis": (basis_instanz_k2_t4, "PLSP-SD-E – Basisinstanz (K=2, T=4)",
              "uebersicht_basis.png"),
    "3a_standby": (standby_instanz, "Verhaltenstest 3a – Standby (K=1, T=3)",
                   "uebersicht_3a_standby.png"),
    "3b_aus": (aus_instanz, "Verhaltenstest 3b – Abschalten (K=1, T=5)",
               "uebersicht_3b_aus.png"),
    "3c_ruesten_aus": (ruesten_im_aus_instanz,
                       "Verhaltenstest 3c – Rüstwechsel im Aus-Modus (K=2, T=5)",
                       "uebersicht_3c_ruesten_aus.png"),
}


def main() -> None:
    for schluessel, (factory, titel, datei) in DEMOS.items():
        print("=" * 72)
        print(f"Instanz: {schluessel}")
        print("=" * 72)
        daten = factory()
        modell = build_model(daten)
        loese(modell, ausgabe=False)
        zeige_loesung(modell, daten)
        visualisiere(modell, daten, titel=titel, dateipfad=datei)
        print(f"-> Grafik gespeichert: {datei}\n")


if __name__ == "__main__":
    main()
