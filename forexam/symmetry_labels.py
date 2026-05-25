import numpy as np
from fractions import Fraction


# ─────────────────────────────────────────────────────────────────────────────
# Rotation recognition
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (trace, det) → (symbol, order)
# det = +1  proper rotation
# det = -1  improper (includes inversion, mirror, roto-inversion)
_ROT_TABLE = {
    ( 3,  1): ("E",   1),   # identity
    (-3, -1): ("i",   1),   # inversion
    (-1,  1): ("C2",  2),
    ( 0,  1): ("C3",  3),
    ( 1,  1): ("C4",  4),
    ( 2,  1): ("C6",  6),
    (-1, -1): ("S6",  6),   # trace -1, det -1 → S6 (roto-inversion 6̄)
    ( 0, -1): ("S4",  4),   # trace  0, det -1 → S4
    ( 1, -1): ("m",   2),   # trace  1, det -1 → mirror (R²=I confirmed)
    ( 2, -1): ("S3",  3),   # trace  2, det -1 → S3
}

def _rotation_symbol(R: np.ndarray) -> str:
    """Return a crystallographic symbol for a 3×3 integer rotation matrix."""
    R = np.array(R)
    tr  = int(round(np.trace(R)))
    det = int(round(np.linalg.det(R)))

    # Use (trace, det) lookup first
    key = (tr, det)
    if key in _ROT_TABLE:
        sym, order = _ROT_TABLE[key]
        # Distinguish C3 / C3² by checking R² == E
        if sym == "C3" and det == 1:
            R2 = R @ R
            if np.allclose(R2, np.eye(3)):
                sym = "C3²"
        # Distinguish C4 / C4³ similarly
        if sym == "C4" and det == 1:
            R3 = R @ R @ R
            if np.allclose(R3, np.eye(3)):
                sym = "C4³"
        # Mirror vs S4: already separated by trace
        return sym

    # Fallback: describe by angle
    cos_theta = (tr - det) / 2.0   # works for proper; approximate otherwise
    cos_theta = np.clip(cos_theta, -1, 1)
    angle_deg = round(np.degrees(np.arccos(cos_theta)))
    kind = "C" if det > 0 else "S"
    return f"{kind}({angle_deg}°)"


# ─────────────────────────────────────────────────────────────────────────────
# Rotation axis
# ─────────────────────────────────────────────────────────────────────────────

def _rotation_axis(R: np.ndarray) -> str:
    """Return the principal axis direction as a string, e.g. '[0 0 1]'."""
    R = np.array(R, dtype=float)
    det = round(np.linalg.det(R))
    M = R if det > 0 else -R          # for improper, axis of underlying proper part

    # Axis = eigenvector of M with eigenvalue +1
    vals, vecs = np.linalg.eig(M)
    for val, vec in zip(vals, vecs.T):
        if abs(val - 1.0) < 1e-6:
            v = np.real(vec)
            v = v / (np.max(np.abs(v)) + 1e-30)   # normalise largest component to ±1
            v = np.round(v).astype(int)
            # canonical form: first non-zero component positive
            for c in v:
                if c != 0:
                    if c < 0:
                        v = -v
                    break
            return f"[{v[0]} {v[1]} {v[2]}]"
    return "?"


# ─────────────────────────────────────────────────────────────────────────────
# Translation formatting
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_translation(t: np.ndarray, tol: float = 1e-4) -> str | None:
    """
    Convert a fractional translation vector to a readable string.
    Returns None if the translation is zero.
    """
    t = np.array(t)
    # Bring components into [0, 1)
    t = t % 1.0
    t[t > 1 - tol] = 0.0           # wrap near-1 to 0

    if np.all(np.abs(t) < tol):
        return None

    parts = []
    for x in t:
        if abs(x) < tol:
            parts.append("0")
        else:
            frac = Fraction(x).limit_denominator(12)
            parts.append(str(frac))

    return "(" + ", ".join(parts) + ")"


# ─────────────────────────────────────────────────────────────────────────────
# Main helper
# ─────────────────────────────────────────────────────────────────────────────

def describe_symmetry_operation(
    rotation,
    translation,
    time_reversal: int = 0,
    index: int | None = None,
    show_axis: bool = True,
) -> str:
    """
    Convert a single MSG operation into a one-line human-readable string.

    Parameters
    ----------
    rotation      : (3, 3) array-like  – integer rotation matrix
    translation   : (3,) array-like    – fractional translation vector
    time_reversal : int                – 0 = no TR, 1 = with TR
    index         : int or None        – operation number for the label
    show_axis     : bool               – append axis direction for non-trivial rots

    Returns
    -------
    str  e.g. "Op 2:  C3 [0 0 1]  +  τ(2/3, 1/3, 0)  +  TR"
    """
    R = np.array(rotation)
    t = np.array(translation, dtype=float)

    sym  = _rotation_symbol(R)
    axis = _rotation_axis(R) if show_axis else None
    trans_str = _fmt_translation(t)
    tr_str    = "TR" if time_reversal else None

    # Build the pieces
    pieces = []

    rot_part = sym
    if show_axis and sym not in ("E", "i") and axis is not None:
        prefix = "⊥" if sym == "m" else ""   # mirror: axis is the plane normal
        rot_part += f" {prefix}{axis}"
    pieces.append(rot_part)

    if trans_str:
        pieces.append(f"τ{trans_str}")
    if tr_str:
        pieces.append(tr_str)

    label = f"Op {index}:  " if index is not None else ""
    return label + "  +  ".join(pieces)


def _ops_match(R1, t1, R2, t2, tol=1e-4) -> bool:
    """Return True if two (rotation, translation) pairs are the same operation."""
    t1 = np.array(t1) % 1.0
    t2 = np.array(t2) % 1.0
    return np.array_equal(R1, R2) and np.allclose(t1, t2, atol=tol)


def describe_all_operations(
    mag_rotations,
    mag_translations,
    mag_time_reversals,
    nm_rotations,
    nm_translations,
    show_axis: bool = True,
) -> None:
    """
    Pretty-print symmetry operations in three sections:

      1. Purely spatial (crystallographic) — present in the non-magnetic dataset
         but BROKEN by the magnetic order (absent from MSG).
      2. Shared spatial — present in both datasets (TR=0); survive magnetic ordering.
      3. Combined spin+space — MSG operations with TR=1; only valid when moments
         are included and time-reversal is applied together with the spatial part.

    Parameters
    ----------
    mag_rotations     : rotations from get_magnetic_symmetry_dataset
    mag_translations  : translations from get_magnetic_symmetry_dataset
    mag_time_reversals: time_reversals from get_magnetic_symmetry_dataset
    nm_rotations      : rotations from get_symmetry_dataset (no moments)
    nm_translations   : translations from get_symmetry_dataset (no moments)
    show_axis         : bool
    """
    W = 58

    # ── Classify non-magnetic operations ────────────────────────────────────
    shared_spatial  = []   # in both nm and msg (TR=0)
    broken_spatial  = []   # in nm only — broken by magnetic order

    for i, (R_nm, t_nm) in enumerate(zip(nm_rotations, nm_translations)):
        in_msg = any(
            _ops_match(R_nm, t_nm, R_m, t_m)
            for R_m, t_m, tr in zip(mag_rotations, mag_translations, mag_time_reversals)
            if tr == 0
        )
        if in_msg:
            shared_spatial.append((i, R_nm, t_nm))
        else:
            broken_spatial.append((i, R_nm, t_nm))

    # ── Combined (TR=1) operations from MSG ─────────────────────────────────
    combined = [
        (i, R, t, tr)
        for i, (R, t, tr) in enumerate(zip(mag_rotations, mag_translations, mag_time_reversals))
        if tr == 1
    ]

    # ── Print ────────────────────────────────────────────────────────────────
    print("─" * W)
    print("  1. Purely spatial operations  (no magnetic moments)")
    print(f"     {len(nm_rotations)} total  |  "
          f"{len(shared_spatial)} survive magnetic ordering  |  "
          f"{len(broken_spatial)} broken")
    print("─" * W)

    if shared_spatial:
        print("  Survive (present in MSG, TR=0):")
        for i, R, t in shared_spatial:
            print("   ", describe_symmetry_operation(R, t, 0, index=i, show_axis=show_axis))

    if broken_spatial:
        print("  Broken by magnetic order:")
        for i, R, t in broken_spatial:
            print("   ", describe_symmetry_operation(R, t, 0, index=i, show_axis=show_axis))

    print()
    print("─" * W)
    print("  2. Combined spin+space operations  (spatial part + TR)")
    print(f"     {len(combined)} operations")
    print("─" * W)
    for i, R, t, tr in combined:
        print("   ", describe_symmetry_operation(R, t, tr, index=i, show_axis=show_axis))

    print("─" * W)


# ─────────────────────────────────────────────────────────────────────────────
# Quick self-test  (matches the 6 operations from your output)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Magnetic dataset (6 operations, as returned by get_magnetic_symmetry_dataset)
    mag_ops = [
        ([[1, 0, 0], [0, 1, 0], [0, 0, 1]],   [0., 0., 0.],                    0),
        ([[0, 1, 0], [1, 0, 0], [0, 0, 1]],   [1.29e-16, -1.29e-16, 0.],       1),
        ([[0, -1, 0], [1, -1, 0], [0, 0, 1]], [0.66666667, 0.33333333, 0.],    0),
        ([[-1, 0, 0], [-1, 1, 0], [0, 0, 1]], [0.66666667, 0.33333333, 0.],    1),
        ([[-1, 1, 0], [-1, 0, 0], [0, 0, 1]], [0.33333333, 0.66666667, 0.],    0),
        ([[1, -1, 0], [0, -1, 0], [0, 0, 1]], [0.33333333, 0.66666667, 0.],    1),
    ]

    # Non-magnetic dataset — same spatial ops as above PLUS extra ones
    # broken by the magnetic order (e.g. a C6 and additional mirrors)
    nm_ops = [
        ([[1, 0, 0], [0, 1, 0], [0, 0, 1]],    [0., 0., 0.]),           # E        (shared)
        ([[0, -1, 0], [1, -1, 0], [0, 0, 1]],  [0.66666667, 0.33333333, 0.]),  # C3 (shared)
        ([[-1, 1, 0], [-1, 0, 0], [0, 0, 1]],  [0.33333333, 0.66666667, 0.]),  # C3²(shared)
        ([[-1, 0, 0], [0, -1, 0], [0, 0, 1]],  [0., 0., 0.]),           # C2 z     (broken)
        ([[0, 1, 0], [-1, 1, 0], [0, 0, 1]],   [0., 0., 0.]),           # C6       (broken)
        ([[1, -1, 0], [1, 0, 0], [0, 0, 1]],   [0., 0., 0.]),           # C6⁻¹     (broken)
    ]

    describe_all_operations(
        mag_rotations    = [o[0] for o in mag_ops],
        mag_translations = [o[1] for o in mag_ops],
        mag_time_reversals=[o[2] for o in mag_ops],
        nm_rotations     = [o[0] for o in nm_ops],
        nm_translations  = [o[1] for o in nm_ops],
    )
