"""Verifikationstests fuer das PLSP-SD-E.

Abgedeckt:
  Test 1  Bau & Loesung der Basisinstanz (K=2, T=4).
  Test 2  Reduktionstest: build_model(reduktion_basismodell=True) liefert denselben
          Zielwert wie das separat gebaute reine PLSP-SD (nur (1)-(8)).
  Test 3a Standby ueberbrueckt kurzen Leerlauf  -> ex. t mit z^{sb}_t > 0.5 (t=2).
  Test 3b Abschalten ueberbrueckt langen Leerlauf -> ex. t mit z^{aus}_t > 0.5 und Sum u_t >= 1.
  Test 3c Ruestwechsel im Aus-Modus -> ex. t in {2,3,4} mit chi_{1,2,t} > 0.5 und z^{aus}_t > 0.5.

Lauffaehig per ``python test_modell.py`` (eigener Runner) oder via pytest.
"""

from __future__ import annotations

from gurobipy import GRB

from instanzen import (
    basis_instanz_k2_t4,
    grosse_instanz_k6_t8,
    standby_instanz,
    aus_instanz,
    ruesten_im_aus_instanz,
)
from modell import build_model
from plsp_sd_basis import build_plsp_sd_basis


def _val(modell, name: str) -> float:
    """Variablenwert per Name (0.0, falls Variable nicht existiert)."""
    var = modell.getVarByName(name)
    return var.X if var is not None else 0.0


def _loese_still(modell):
    """Optimiert ohne Solver-Log."""
    modell.Params.OutputFlag = 0
    modell.optimize()
    return modell


# ---------------------------------------------------------------------------
# Test 1: Bau & Loesung
# ---------------------------------------------------------------------------
def test_baut_und_loest():
    daten = basis_instanz_k2_t4()
    modell = _loese_still(build_model(daten))
    assert modell.Status == GRB.OPTIMAL, f"Status {modell.Status}, erwartet OPTIMAL"
    assert modell.SolCount >= 1
    return modell.ObjVal


# ---------------------------------------------------------------------------
# Test 1b: Grosse Instanz (K=6, T=8) baut, loest optimal und uebt alle Mechanismen
# ---------------------------------------------------------------------------
def test_grosse_instanz_k6_t8():
    daten = grosse_instanz_k6_t8()
    modell = _loese_still(build_model(daten))
    assert modell.Status == GRB.OPTIMAL, f"Status {modell.Status}, erwartet OPTIMAL"

    perioden = list(daten.perioden())
    summe_aus = sum(_val(modell, f"z_aus[{t}]") for t in perioden)
    summe_u = sum(_val(modell, f"u[{t}]") for t in perioden)
    summe_B = sum(_val(modell, f"B[{t}]") for t in perioden)
    summe_S = sum(_val(modell, f"S[{t}]") for t in perioden)
    # familienuebergreifende Ruestwechsel (A={1,2,3}, B={4,5,6}) muessen auftreten
    cross = sum(
        _val(modell, f"chi[{i},{k},{t}]")
        for i in daten.produkte() for k in daten.produkte()
        for t in perioden
        if i != k and (i <= 3) != (k <= 3)
    )

    # Die Instanz ist so kalibriert, dass das Vollmodell folgende Mechanismen nutzt:
    assert summe_aus >= 1.0 - 1e-6, "Abschalten (Aus) wird nicht genutzt"
    assert summe_u >= 1.0 - 1e-6, "kein Anschaltvorgang (Wiederanlauf)"
    assert summe_B > 1e-6, "kein Zertifikat-Zukauf (Netto-Kaeufer erwartet)"
    assert cross >= 1.0 - 1e-6, "kein familienuebergreifender Ruestwechsel"
    return modell.ObjVal, summe_aus, summe_u, summe_B, summe_S


# ---------------------------------------------------------------------------
# Test 2: Reduktionstest gegen reines PLSP-SD
# ---------------------------------------------------------------------------
def test_reduktion_entspricht_basis():
    daten = basis_instanz_k2_t4()

    voll_red = _loese_still(build_model(daten, reduktion_basismodell=True))
    basis = _loese_still(build_plsp_sd_basis(daten))

    assert voll_red.Status == GRB.OPTIMAL
    assert basis.Status == GRB.OPTIMAL
    assert abs(voll_red.ObjVal - basis.ObjVal) < 1e-6, (
        f"Reduktion {voll_red.ObjVal} != Basis {basis.ObjVal}"
    )
    return voll_red.ObjVal, basis.ObjVal


# ---------------------------------------------------------------------------
# Test 3a: Standby
# ---------------------------------------------------------------------------
def test_3a_standby():
    daten = standby_instanz()
    modell = _loese_still(build_model(daten))
    assert modell.Status == GRB.OPTIMAL

    z_sb = [_val(modell, f"z_sb[{t}]") for t in daten.perioden()]
    assert any(v > 0.5 for v in z_sb), f"Kein Standby genutzt: z_sb={z_sb}"
    # Erwartet konkret in t=2 (der Leerlaufperiode)
    assert _val(modell, "z_sb[2]") > 0.5, "Standby nicht in t=2"
    return z_sb


# ---------------------------------------------------------------------------
# Test 3b: Abschalten
# ---------------------------------------------------------------------------
def test_3b_aus():
    daten = aus_instanz()
    modell = _loese_still(build_model(daten))
    assert modell.Status == GRB.OPTIMAL

    z_aus = [_val(modell, f"z_aus[{t}]") for t in daten.perioden()]
    summe_u = sum(_val(modell, f"u[{t}]") for t in daten.perioden())
    assert any(v > 0.5 for v in z_aus), f"Kein Abschalten genutzt: z_aus={z_aus}"
    assert summe_u >= 1.0 - 1e-6, f"Kein Wiederanlauf: Sum u_t = {summe_u}"
    return z_aus, summe_u


# ---------------------------------------------------------------------------
# Test 3c: Ruestwechsel im Aus-Modus
# ---------------------------------------------------------------------------
def test_3c_ruesten_im_aus():
    daten = ruesten_im_aus_instanz()
    modell = _loese_still(build_model(daten))
    assert modell.Status == GRB.OPTIMAL

    treffer = [
        t for t in (2, 3, 4)
        if _val(modell, f"chi[1,2,{t}]") > 0.5 and _val(modell, f"z_aus[{t}]") > 0.5
    ]
    assert treffer, (
        "Wechsel 1->2 nicht im Aus-Modus einer Leerlaufperiode t in {2,3,4}. "
        f"chi_12={[_val(modell, f'chi[1,2,{t}]') for t in (2,3,4)]}, "
        f"z_aus={[_val(modell, f'z_aus[{t}]') for t in (2,3,4)]}"
    )
    return treffer


if __name__ == "__main__":
    print("== Test 1: Bau & Loesung (K=2, T=4) ==")
    z1 = test_baut_und_loest()
    print(f"   OK  -> Zielwert Z = {z1:.4f}\n")

    print("== Test 1b: Grosse Instanz (K=6, T=8), alle Mechanismen ==")
    z1b, s_aus, s_u, s_B, s_S = test_grosse_instanz_k6_t8()
    print(f"   OK  -> Z = {z1b:.2f}; Aus-Perioden={s_aus:.0f}, Anschalt={s_u:.0f}, "
          f"Zukauf={s_B:.1f}, Verkauf={s_S:.1f}\n")

    print("== Test 2: Reduktionstest (== reines PLSP-SD) ==")
    z_red, z_basis = test_reduktion_entspricht_basis()
    print(f"   OK  -> Z(reduktion) = {z_red:.4f} == Z(PLSP-SD) = {z_basis:.4f}\n")

    print("== Test 3a: Standby ueberbrueckt kurzen Leerlauf ==")
    z_sb = test_3a_standby()
    print(f"   OK  -> z_sb (t=1..3) = {[round(v, 2) for v in z_sb]} (Standby in t=2)\n")

    print("== Test 3b: Abschalten ueberbrueckt langen Leerlauf ==")
    z_aus, su = test_3b_aus()
    print(f"   OK  -> z_aus (t=1..5) = {[round(v, 2) for v in z_aus]}, Sum u_t = {su:.2f}\n")

    print("== Test 3c: Ruestwechsel im Aus-Modus ==")
    treffer = test_3c_ruesten_im_aus()
    print(f"   OK  -> Wechsel 1->2 im Aus-Modus in Periode(n) {treffer}\n")

    print("Alle Tests bestanden.")
