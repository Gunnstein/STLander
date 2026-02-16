from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pyvista as pv
from scipy import linalg


@dataclass(frozen=True)
class AlignmentResult:
    original: pv.PolyData
    translated: pv.PolyData
    aligned: pv.PolyData
    center_of_mass: np.ndarray  # (3,)
    axes_matrix: np.ndarray     # (3,3) columns are principal axes in original coords
    eigenvalues: np.ndarray     # (3,)


def load_mesh(path: str) -> pv.PolyData:
    """Load an STL (or any PyVista-supported file) and ensure triangles."""
    mesh = pv.read(path)
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface().triangulate()
    else:
        mesh = mesh.triangulate()
    mesh.clean(inplace=True)
    return mesh


def _triangles_from_polydata(mesh: pv.PolyData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tri = mesh
    if tri.n_cells == 0:
        raise ValueError("Mesh has no faces/cells.")
    if not tri.is_all_triangles:
        tri = tri.triangulate()

    faces = tri.faces.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError("Expected triangulated faces (all triangles).")

    ids = faces[:, 1:4]
    pts = tri.points
    a = pts[ids[:, 0]]
    b = pts[ids[:, 1]]
    c = pts[ids[:, 2]]
    return a, b, c


def compute_surface_com_and_axes(mesh: pv.PolyData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Area-weighted surface COM and principal axes (thin shell assumption)."""
    a, b, c = _triangles_from_polydata(mesh)

    ab = b - a
    ac = c - a
    cross = np.cross(ab, ac)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    if not np.isfinite(areas).all() or np.allclose(areas.sum(), 0.0):
        raise ValueError("Degenerate mesh: total triangle area is zero or invalid.")

    centroids = (a + b + c) / 3.0
    wsum = areas.sum()
    com = (centroids * areas[:, None]).sum(axis=0) / wsum

    r = centroids - com[None, :]
    cov = (areas[:, None, None] * (r[:, :, None] * r[:, None, :])).sum(axis=0) / wsum

    evals, evecs = linalg.eigh(cov)  # ascending
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]  # columns

    # Deterministic signs
    for i in range(3):
        j = int(np.argmax(np.abs(evecs[:, i])))
        if evecs[j, i] < 0:
            evecs[:, i] *= -1.0

    # Right-handed
    if np.linalg.det(evecs) < 0:
        evecs[:, 2] *= -1.0

    return com.astype(float), evecs.astype(float), evals.astype(float)


def _ensure_right_handed(axes: np.ndarray) -> np.ndarray:
    """Ensure a 3x3 basis matrix is right-handed (determinant +1)."""
    ax = axes.copy()
    if np.linalg.det(ax) < 0:
        ax[:, 2] *= -1.0
    return ax


def align_to_principal_axes(mesh: pv.PolyData, pa2_target: str = "Y") -> AlignmentResult:
    """Translate mesh to COM=0 and rotate so principal axes align with XYZ.

    Parameters
    ----------
    pa2_target:
        Where the **2nd** principal axis (intermediate eigenvector) should land.
        - "Y" (default): [PA1 -> X, PA2 -> Y, PA3 -> Z]
        - "Z":           [PA1 -> X, PA2 -> Z, PA3 -> Y] (swap Y/Z)

    Notes
    -----
    Uses:
        p_aligned = (p_original - COM) @ axes_matrix
    with row-vector convention.
    """
    original = mesh.copy(deep=True)
    com, axes, evals = compute_surface_com_and_axes(original)

    pa2_target = (pa2_target or "Y").upper().strip()
    if pa2_target not in {"Y", "Z"}:
        raise ValueError("pa2_target must be 'Y' or 'Z'.")

    # axes columns are [PA1, PA2, PA3]
    if pa2_target == "Z":
        # Make PA2 land on Z, and PA3 land on Y -> swap columns 2 and 3
        axes = axes[:, [0, 2, 1]]

    axes = _ensure_right_handed(axes)

    translated = original.copy(deep=True)
    translated.points = translated.points - com[None, :]

    aligned = translated.copy(deep=True)
    aligned.points = aligned.points @ axes

    return AlignmentResult(
        original=original,
        translated=translated,
        aligned=aligned,
        center_of_mass=com,
        axes_matrix=axes,
        eigenvalues=evals,
    )


def save_mesh(mesh: pv.PolyData, path: str) -> None:
    mesh.save(path)


def rotate_alignment_result(res: AlignmentResult, axis: str, degrees: float = 180.0) -> AlignmentResult:
    """Rotate the aligned mesh by `degrees` about `axis` around the mesh COM.

    Rotation is performed in the aligned coordinate frame (i.e. about the
    current X/Y/Z axes of the aligned mesh). The function updates the
    `axes_matrix` by post-multiplying with the requested rotation matrix and
    recomputes the `aligned` mesh from the stored `translated` points.

    Parameters
    ----------
    res:
        The existing :class:`AlignmentResult` to transform.
    axis:
        One of 'X', 'Y', or 'Z' indicating the rotation axis in the aligned
        coordinate frame.
    degrees:
        Rotation angle in degrees (default 180.0).
    """
    axis = (axis or "").strip().upper()
    if axis not in {"X", "Y", "Z"}:
        raise ValueError("axis must be one of 'X','Y','Z'")

    theta = float(degrees) * np.pi / 180.0
    c = np.cos(theta)
    s = np.sin(theta)

    if axis == "X":
        R = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)
    elif axis == "Y":
        R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)
    else:  # Z
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    axes = res.axes_matrix.copy()
    axes = axes @ R
    axes = _ensure_right_handed(axes)

    translated = res.translated
    new_aligned = translated.copy(deep=True)
    new_aligned.points = translated.points @ axes

    return AlignmentResult(
        original=res.original,
        translated=res.translated,
        aligned=new_aligned,
        center_of_mass=res.center_of_mass,
        axes_matrix=axes,
        eigenvalues=res.eigenvalues,
    )
