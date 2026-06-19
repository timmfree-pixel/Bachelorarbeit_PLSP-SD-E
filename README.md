# PLSP-SD-E — MILP-Implementierung (gurobipy)

Implementierung des **Proportional Lot-Sizing and Scheduling Problem mit
Sequenzabhängigen Rüstzeiten und Emissionshandel** (PLSP-SD-E) als gemischt-ganzzahliges
lineares Programm (MILP) mit `gurobipy`.

Die Implementierung bildet die mathematische Spezifikation **(1)–(15) exakt** ab:
keine Restriktion hinzugefügt, keine weggelassen, keine Notation verändert.

## Architektur (strikte Trennung)

| Datei | Verantwortung |
|-------|---------------|
| `instanz.py` | Datenschema (`@dataclass Instanz`) – Mengen/Parameter, **keine** Zahlen, **keine** Modelllogik |
| `instanzen.py` | Konkrete Probleminstanzen (Daten) für Tests/Validierung |
| `modell.py` | **Modelllogik**: `build_model(daten, reduktion_basismodell=False) -> gp.Model` |
| `plsp_sd_basis.py` | Reines PLSP-SD (nur (1)–(8)) als Referenz für den Reduktionstest |
| `loesung.py` | Lösung & Auswertung (Optimieren, Solution-Auslese, Anzeige) |
| `test_modell.py` | Verifikation: Bau/Lösung, Reduktionstest, Verhaltenstests 3a–3c |

Im Modellcode sind **keine Zahlenwerte hartkodiert**; die Big-M-Werte
`M_{kt} = min(b_t/tb_k, Σ_{τ≥t} d_{kτ})` und `M_t = b_t / min_k tb_k` werden in
`build_model` aus den Instanzdaten berechnet.

## Verwendung

```python
from instanzen import basis_instanz_k2_t4
from modell import build_model
from loesung import loese, zeige_loesung

daten = basis_instanz_k2_t4()
modell = build_model(daten)                 # Vollmodell PLSP-SD-E
# modell = build_model(daten, reduktion_basismodell=True)  # reines PLSP-SD reproduzieren
loese(modell, ausgabe=False)
zeige_loesung(modell, daten)
modell.write("plsp_sd_e.lp")                # optionaler .lp-Export
```

## Spezifikations-Abgleich (1)–(15)

| Nr. | Bedeutung | Code (`modell.py`) |
|-----|-----------|--------------------|
| (1) | Zielfunktion (Lager + Rüstkosten `s_k` je Wechsel **in** k + Zertifikatssaldo) | `setObjective(...)` |
| (2) | Lagerbilanz | `c2_lagerbilanz` |
| (3) | Kapazität (Produktion + sequenzabh. Rüsten, **kein** Anschaltterm) | `c3_kapazitaet` |
| (4) | Eindeutiger Rüstzustand | `c4_ruestzustand_eindeutig` |
| (5) | Produktion nur bei (vor-)gerüstetem Zustand | `c5_produktion_nur_geruestet` |
| (6) | Rüstzustandsfortschreibung `Σ_i χ_{ikt}` (Wechsel **in** k) | `c6_wechsel_in_k` |
| (7) | Spiegelbild `Σ_i χ_{kit}` (Wechsel **aus** k) | `c7_wechsel_aus_k` |
| (8) | Anfangsbedingungen `ω_{i0,0}=1`, `ω_{k,0}=0`, `y_{k,0}` | `c8_*` |
| (9) | Eindeutiger Leistungszustand An/Standby/Aus | `c9_leistungszustand_eindeutig` |
| (10) | An-Zustand nur bei Produktion (Rüsten von z entkoppelt) | `c10_an_bei_produktion` |
| (11) | Anschaltvorgang Aus→Ein erkennen | `c11_anschaltvorgang` |
| (12) | Gesamtemission der Periode | `c12_emission` |
| (13) | Zertifikatssaldo | `c13_*` |
| (14) | `J_t ≥ 0` | `c14_saldo_nichtnegativ` |
| (15) | `z^{aus}_0 = 0` | `c15_z_aus0` |

## Modellannahmen / Default-Entscheidungen

Vollständig dokumentiert im Kopf-Docstring von `modell.py` (Block **A1–A8**). Kernpunkte:

- **A3** `χ_{ikt}` nur für `i ≠ k` (kein i=k-Self-Loop, gemäß Logik-Hinweis); inkludierte
  Self-Loops wären im Optimum stets 0 → identische Lösung.
- **A2** Anfangsperiode `t=0` als fixierte Variablen für `ω, y, z^{aus}, J`, sodass alle
  `(t−1)`-Bezüge bei `t=1` einheitlich auf (8)/(15)/Parameter zugreifen.
- **A5** `χ, u` als `CONTINUOUS` deklariert (binäres Verhalten induziert); `ω, z` `BINARY`.
- **A6** `J_t` frei; Nichtnegativität ausschließlich über (14).
- **A7** `reduktion_basismodell=True` nullt alle Emissionsparameter und `π^B, π^S`;
  (9)–(15) bleiben strukturell, wirken aber zielfunktionsneutral.

## Tests ausführen

```bash
python3 test_modell.py        # eigener Runner (ohne pytest-Abhängigkeit)
# alternativ, falls pytest installiert:
python3 -m pytest test_modell.py -q
```

Abgedeckt:
1. **Bau & Lösung** der Basisinstanz (K=2, T=4).
2. **Reduktionstest**: `build_model(..., reduktion_basismodell=True)` liefert denselben
   Zielwert wie das separat gebaute reine PLSP-SD (nur (1)–(8)).
3. **Verhaltenstests** (Netto-Verkäufer-Regime, `e^{aus}=0 ≤ e^{sb} ≤ e^{on}`):
   - **3a** Standby überbrückt kurzen Leerlauf (`∃ t: z^{sb}_t > 0.5`, konkret t=2).
   - **3b** Abschalten überbrückt langen Leerlauf (`∃ t: z^{aus}_t > 0.5` und `Σ_t u_t ≥ 1`).
   - **3c** Rüstwechsel im Aus-Modus (`∃ t∈{2,3,4}: χ_{1,2,t} > 0.5` und `z^{aus}_t > 0.5`).
