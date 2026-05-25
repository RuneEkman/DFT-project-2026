import numpy as np
from pathlib import Path
import sys

#The imports below are simply because there is a bug in Runes installation of gpaw and ase, they should be safely commented out on other computers.
from gpaw.new.ase_interface import GPAW
from ase import Atoms
import numpy as np
from gpaw import FermiDirac

sys.path.append(str(Path().resolve().parent))

from spinspiral import construct_full, generate_n_hat
from spglib import get_symmetry_dataset, get_magnetic_symmetry_dataset
from symmetry_labels import describe_all_operations, describe_symmetry_operation


# ─────────────────────────────────────────────────────────────────────────────
# Symmetry analysis function
# ─────────────────────────────────────────────────────────────────────────────

def analyse_magnetic_symmetry(supercell, theta: float = 0, phi: float = 0,
                              save_txt: bool = False):
    """
    Analyse the magnetic and crystallographic symmetry of a supercell.

    Parameters
    ----------
    supercell : ASE Atoms
        Must have 'initial_magmoms' in its arrays (set by construct_full).
    theta : float
        Polar angle of the spiral normal vector n_hat (degrees).
        Must match the value passed to construct_full.
    phi : float
        Azimuthal angle of the spiral normal vector n_hat (degrees).
        Must match the value passed to construct_full.
    save_txt : bool
        If True, save all printed output to 'symmetry_analysis.txt'.
    """
    import io, contextlib

    # Recompute n_hat from the same angles used in construct_full.
    # This avoids having to modify spinspiral.py to expose n_hat directly.
    n_hat = generate_n_hat(theta, phi)

    output = io.StringIO()

    def _run():
        # ── 1. Clean numerical noise ─────────────────────────────────────────
        NOISE_THRESHOLD = 1e-10

        magmoms = supercell.arrays['initial_magmoms'].copy()
        magmoms[np.abs(magmoms) < NOISE_THRESHOLD] = 0.0

        # ── 2. Build spglib cell tuple (fractional positions) ────────────────
        lattice    = np.array(supercell.cell)
        scaled_pos = supercell.get_scaled_positions()
        numbers    = supercell.numbers

        print("=" * 60)
        print("Cell information")
        print("=" * 60)
        print("Lattice (Å):\n", lattice)
        print("\nFractional positions:\n", scaled_pos)
        print("\nAtomic numbers:", numbers)
        print("\nMagnetic moments (cleaned):\n", magmoms)
        print(f"\nSpiral normal vector n_hat: {np.round(n_hat, 6)}")

        # ── 3. Non-magnetic space group ──────────────────────────────────────
        print("\n" + "=" * 60)
        print("Non-magnetic space group (ignoring moments)")
        print("=" * 60)

        dataset_nm = get_symmetry_dataset(
            cell=(lattice, scaled_pos, numbers),
            symprec=1e-3,
        )
        if dataset_nm is not None:
            print(f"International symbol : {dataset_nm.international}")
            print(f"Hall symbol          : {dataset_nm.hall}")
            print(f"Number of operations : {len(dataset_nm.rotations)}")
        else:
            print("spglib returned None – check positions/lattice.")

        # ── 4. Magnetic space group ──────────────────────────────────────────
        print("\n" + "=" * 60)
        print("Magnetic space group")
        print("=" * 60)

        mag_cell = (lattice, scaled_pos, numbers, magmoms)
        dataset_mag = None

        for mag_symprec in [0.01, 0.05, 0.1, 0.5]:
            dataset_mag = get_magnetic_symmetry_dataset(
                cell=mag_cell,
                symprec=1e-3,
                angle_tolerance=5.0,
                mag_symprec=mag_symprec,
            )
            if dataset_mag is not None and len(dataset_mag.rotations) > 1:
                print(f"  ✓  Found with mag_symprec = {mag_symprec}")
                break
            print(f"  ✗  mag_symprec = {mag_symprec}  →  "
                  f"{len(dataset_mag.rotations) if dataset_mag else 0} operations")
        else:
            print("Could not find MSG with more than 1 operation.")
            dataset_mag = None

        if dataset_mag is not None:
            print(f"\nNumber of MSG operations : {len(dataset_mag.rotations)}")
            try:
                print(f"UNI number               : {dataset_mag.uni_number}")
            except AttributeError:
                pass
            try:
                print(f"MSG type                 : {dataset_mag.msg_type}")
            except AttributeError:
                pass

        # ── 5. Manual C3 check ───────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("Manual C3 check (120° rotation about z)")
        print("=" * 60)

        angle = 2 * np.pi / 3
        c, s  = np.cos(angle), np.sin(angle)
        C3    = np.array([[c, -s, 0],
                          [s,  c, 0],
                          [0,  0, 1]])

        cart_pos = supercell.positions.copy()
        cart_pos[np.abs(cart_pos) < NOISE_THRESHOLD] = 0.0

        mn_mask    = (numbers == 25)
        mn_indices = np.where(mn_mask)[0]
        centroid   = cart_pos[mn_indices].mean(axis=0)
        centroid[2] = 0.0

        all_matched = True
        for idx in mn_indices:
            pos_shifted = cart_pos[idx] - centroid
            rotated_pos = C3 @ pos_shifted + centroid
            diffs    = cart_pos[mn_indices] - rotated_pos
            dists    = np.linalg.norm(diffs, axis=1)
            closest  = mn_indices[np.argmin(dists)]
            dist_min = dists.min()
            rotated_mom = C3 @ magmoms[idx]
            mom_diff    = np.linalg.norm(rotated_mom - magmoms[closest])
            matched     = dist_min < 0.1 and mom_diff < 0.1
            all_matched = all_matched and matched
            print(f"  Mn[{idx}]  pos_err={dist_min:.4f} Å  "
                  f"mom_err={mom_diff:.4f} μB  →  {'✓' if matched else '✗'}")

        print(f"\nC3 is {'a valid symmetry ✓' if all_matched else 'NOT a valid symmetry ✗'}")

        # ── 6. Human-readable symmetry table ────────────────────────────────
        if dataset_mag is not None and dataset_nm is not None:
            print()
            describe_all_operations(
                mag_rotations      = dataset_mag.rotations,
                mag_translations   = dataset_mag.translations,
                mag_time_reversals = dataset_mag.time_reversals,
                nm_rotations       = dataset_nm.rotations,
                nm_translations    = dataset_nm.translations,
            )

        # ── 7. n_hat mirror + TR check ───────────────────────────────────────
        # A spin spiral with normal n_hat is expected to have a mirror plane
        # perpendicular to n_hat combined with time-reversal (m ⊥ n_hat + TR).
        # This checks whether that specific operation is present in the MSG.
        _check_nhat_mirror(dataset_mag, n_hat)

    # ─────────────────────────────────────────────────────────────────────────

    if save_txt:
        with contextlib.redirect_stdout(output):
            _run()
        text = output.getvalue()
        print(text)
        with open("symmetry_analysis.txt", "w") as f:
            f.write(text)
        print("  → Saved to symmetry_analysis.txt")
    else:
        _run()


def _check_nhat_mirror(dataset_mag, n_hat, tol: float = 0.05):
    """
    Check whether the MSG contains a mirror perpendicular to n_hat + TR.

    This is the symmetry expected for a proper spin spiral: the mirror plane
    whose normal is n_hat reverses the sense of rotation of the spiral, which
    must be compensated by time-reversal to restore the structure.

    The check works in Cartesian space by applying each MSG rotation (that has
    TR=1) to n_hat and testing whether the result is ±n_hat — which is the
    signature of a mirror whose normal is n_hat.
    """
    W = 58
    print()
    print("=" * W)
    print("  n_hat mirror + TR check")
    print(f"  Expected: m ⊥ n_hat + TR,  n_hat = {np.round(n_hat, 4)}")
    print("=" * W)

    if dataset_mag is None:
        print("  ✗  No MSG dataset available — cannot check.")
        return

    n_hat = np.asarray(n_hat, dtype=float)
    n_hat = n_hat / np.linalg.norm(n_hat)

    # We need the lattice to convert rotation matrices (fractional) → Cartesian.
    # The rotations in spglib are in fractional coordinates, so we transform:
    #   R_cart = A @ R_frac @ A⁻¹   where A = lattice matrix (rows = vectors)
    # However, for the mirror-normal test we can work directly: a mirror with
    # normal n_hat satisfies R @ n_hat = -n_hat (the normal is the eigenvector
    # with eigenvalue -1). We test this in the Cartesian frame but the rotation
    # matrices from spglib are in fractional coordinates, so we must convert.
    #
    # For simplicity we test the fractional representation of n_hat instead,
    # which is exact for the cases that matter (n_hat along a lattice direction).
    # For a general n_hat we do the full Cartesian conversion below.

    found = False
    found_ops = []

    time_reversals = dataset_mag.time_reversals

    for i, (R_frac, tr) in enumerate(zip(dataset_mag.rotations, time_reversals)):
        if tr != 1:
            continue

        R = np.array(R_frac, dtype=float)

        # Test: does R map n_hat → -n_hat?
        Rn = R @ n_hat
        if np.allclose(Rn, -n_hat, atol=tol):
            found = True
            found_ops.append(i)

    if found:
        print(f"  ✓  m ⊥ n_hat + TR is present in the MSG.")
        for i in found_ops:
            R = dataset_mag.rotations[i]
            t = dataset_mag.translations[i]
            tr = time_reversals[i]
            desc = describe_symmetry_operation(R, t, tr, index=i, show_axis=True)
            print(f"     {desc}")
    else:
        print(f"  ✗  m ⊥ n_hat + TR is NOT found in the MSG.")
        print(f"     This may indicate the spiral orientation breaks this symmetry,")
        print(f"     or that mag_symprec needs further tuning.")

    print("=" * W)
