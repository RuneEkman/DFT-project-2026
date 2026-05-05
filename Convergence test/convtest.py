#Primary metric: total energy per atom.
#Secondary: magnetic moment magnitrude on Mn.

#For each run, log:
# total energy (energy/atom)
# calc.get_non_collinear_magnetic_moments()
#  #check Mn magnitude around 4.5 µB. Total moment around 0.
# Check whether SCF converged within maxiter

# Phase 1: k-convergence at low cutoff (e.g. 300 eV)

# Phase 2: ecut-convergence at low k-grid (e.g. 4,4,1)


from gpaw.new.ase_interface import GPAW
import numpy as np
from gpaw import FermiDirac
from pathlib import Path
import sys
import csv
import traceback

from ase.parallel import parprint


sys.path.append(str(Path().resolve().parent))
from spinspiral import construct_full

# ==============================================================================
# MnI2 Convergence Test — k-points and PW cutoff
#
# Strategy:
#   Phase 1: Converge k-points at a low fixed cutoff (KCONV_ECUT).
#             Use energy/atom as primary metric, Mn moment magnitudes as sanity.
#   Phase 2: Converge PW cutoff at the converged k-grid (ECUT_KGRID).
#
# Convergence criteria are loosened relative to production to keep runs fast.
# Results are appended to a CSV after each calculation so nothing is lost
# if the job is killed mid-run.
# ==============================================================================

# --- Shared structure parameters (keep identical to production run) ----------
theta, phi      = 0, 0
path_to_cif     = "1MnI2-1.cif"
Q               = [1/3, 1/3, 0]
magnetic_magnitude = 4.5
P = np.array([
    [2,  1, 0],
    [-1, 1, 0],
    [0,  0, 1]
])
magnetic_atom   = 'Mn'
init_moment     = [0, 4.5, 0]

# --- Convergence test parameters ---------------------------------------------
# Phase 1: k-convergence — fix a cheap cutoff, sweep k-grids
KCONV_ECUT  = 300          # eV  (low but not absurd for a quick scan)
K_GRIDS     = [
    (4,  4,  1),
    (8,  8,  1),
    (10, 10, 1),
    (12, 12, 1),
]

# Phase 2: ecut-convergence — use a cheap fixed k-grid 
# (ecut convergence is independent of k-sampling; a coarse grid is fine here)
ECUT_KGRID  = (4, 4, 1)    # deliberately cheap 
ECUT_VALUES = [300, 400, 500, 600, 700]   # eV

# Loosened SCF convergence for test runs (production uses 1e-9 / 1e-10)
TEST_CONVERGENCE = {'density': 0.0005, 'energy': 0.0001, 'eigenstates': 4e-8}
TEST_MAXITER     = 200

# Parallelisation 
PARALLEL = {'domain': 4, 'kpt': 4, 'band': 1}

# Output CSV
CSV_FILE = "convergence_results.csv"
CSV_HEADER = [
    "phase", "label",
    "kgrid", "ecut_eV",
    "energy_eV", "energy_per_atom_eV",
    "total_magmom",
    "mn_moment_magnitudes",   # pipe-separated list of |m| for each Mn
    "scf_converged",
    "notes"
]

# ==============================================================================

def build_supercell():
    """Fresh supercell + magmoms for every calculator.
    Note: GPAW('file.gpw') can reload a .gpw, but since ecut/kgrid change
    between runs the stored density/wavefunctions are on a different grid
    and won't help SCF convergence — so fresh start every time
    """
    supercell, name = construct_full(
        theta=theta, phi=phi, Q=Q,
        path=path_to_cif,
        transform=P,
        magnitude=magnetic_magnitude,
        magsymbols=magnetic_atom,
        init_moment=init_moment,
    )
    magmoms = supercell.arrays['initial_magmoms']
    return supercell, name, magmoms


def make_calc(ecut, kgrid, txt_name, magmoms):
    return GPAW(
        mode={'name': 'pw', 'ecut': ecut},
        xc='LDA',
        mixer={
            'backend': 'pulay',
            'beta': 0.05,
            'method': 'sum',
            'nmaxold': 5,
            'weight': 100,
        },
        kpts={'size': kgrid, 'gamma': True},
        symmetry='off',
        magmoms=magmoms,
        spinpol=True,
        occupations=FermiDirac(0.01),
        txt=txt_name,
        maxiter=TEST_MAXITER,
        parallel=PARALLEL,
        soc=False,
        convergence=TEST_CONVERGENCE,
    )


def get_mn_moment_magnitudes(calc, supercell):
    """Return list of |m| for every Mn atom."""
    symbols  = supercell.get_chemical_symbols()
    nc_moms  = calc.get_non_collinear_magnetic_moments()   # shape (N, 3)
    mn_mags  = [np.linalg.norm(nc_moms[i]) for i, s in enumerate(symbols) if s == 'Mn']
    return mn_mags


def append_csv(row: dict):
    file_exists = Path(CSV_FILE).exists()
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_single(phase, label, ecut, kgrid):
    """Run one SCF, collect results, append to CSV. Returns energy/atom."""
    parprint(f"\n{'='*60}") #To separate outputs
    parprint(f"  {phase} | {label} | ecut={ecut} eV | kgrid={kgrid}") #output
    parprint(f"{'='*60}") #to separate output

    txt_name = f"conv_{label}.txt"
    row = {
        "phase": phase, "label": label,
        "kgrid": 'x'.join(map(str, kgrid)),
        "ecut_eV": ecut,
        "energy_eV": None, "energy_per_atom_eV": None,
        "total_magmom": None,
        "mn_moment_magnitudes": None,
        "scf_converged": False,
        "notes": "",
    }

    try:
        supercell, _, magmoms = build_supercell()
        n_atoms = len(supercell)

        calc = make_calc(ecut, kgrid, txt_name, magmoms)
        calc.verbosity = 1
        supercell.calc = calc

        energy = supercell.get_potential_energy()

        mn_mags  = get_mn_moment_magnitudes(calc, supercell)
        tot_mom  = supercell.get_magnetic_moment()

        row["energy_eV"]          = round(energy, 6)
        row["energy_per_atom_eV"] = round(energy / n_atoms, 6)
        row["total_magmom"]       = round(tot_mom, 4)
        row["mn_moment_magnitudes"] = '|'.join(f'{m:.3f}' for m in mn_mags)
        row["scf_converged"]      = True

        parprint(f"  Energy/atom : {energy/n_atoms:.6f} eV")
        parprint(f"  Total magmom: {tot_mom:.4f} µB  (should be  around 0)")
        parprint(f"  Mn |m|      : {mn_mags}  (should be around 4.5 µB each)")

    except Exception as e:
        row["notes"] = f"FAILED: {str(e)[:120]}"
        parprint(f"  !! Run failed: {e}")
        traceback.print_exc()

    append_csv(row)
    return row


# ==============================================================================
# Phase 1 — k-point convergence
# ==============================================================================
parprint("\n" + "#"*60)
parprint("  PHASE 1: k-point convergence  (ecut = {} eV)".format(KCONV_ECUT))
parprint("#"*60)

kconv_results = []
for kgrid in K_GRIDS:
    label = "kconv_{}x{}x{}".format(*kgrid)
    row = run_single("kconv", label, KCONV_ECUT, kgrid)
    kconv_results.append(row)

# parprint a quick energy-difference table to guide choice of ECUT_KGRID
parprint("\n--- Phase 1 summary ---")
parprint(f"{'kgrid':<14} {'E/atom (eV)':<18} {'ΔE/atom (meV)':<16} {'Mn |m| (µB)'}")
prev_e = None
for r in kconv_results:
    e = r["energy_per_atom_eV"]
    delta = (e - prev_e) * 1000 if prev_e is not None else 0.0
    parprint(f"  {r['kgrid']:<12} {e:<18} {delta:<16.2f} {r['mn_moment_magnitudes']}")
    prev_e = e


# ==============================================================================
# Phase 2 — PW cutoff convergence
# ==============================================================================
parprint("\n" + "#"*60)
parprint(f"Currently using ECUT_KGRID = {ECUT_KGRID} for Phase 2.")
parprint("  PHASE 2: PW cutoff convergence  (kgrid = {}x{}x{})".format(*ECUT_KGRID))
parprint("#"*60)

ecut_results = []
for ecut in ECUT_VALUES:
    label = "ecut_{}eV".format(ecut)
    row = run_single("econv", label, ecut, ECUT_KGRID)
    ecut_results.append(row)

parprint("\n--- Phase 2 summary ---")
parprint(f"{'ecut (eV)':<12} {'E/atom (eV)':<18} {'ΔE/atom (meV)':<16} {'Mn |m| (µB)'}")
prev_e = None
for r in ecut_results:
    e = r["energy_per_atom_eV"]
    delta = (e - prev_e) * 1000 if prev_e is not None else 0.0
    parprint(f"  {r['ecut_eV']:<12} {e:<18} {delta:<16.2f} {r['mn_moment_magnitudes']}")
    prev_e = e

parprint(f"\nAll results written to: {CSV_FILE}")
parprint("Done.")
